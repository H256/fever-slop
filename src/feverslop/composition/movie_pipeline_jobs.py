from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from feverslop.application.movie import build_movie_actor_reference_prompt, build_movie_actor_visual_description
from feverslop.composition.movie_workflow import patch_movie_msr_workflow
from feverslop.application.movie_artifacts import (
    ensure_movie_planning_artifacts,
    write_movie_reference_manifest_from_bible,
)
from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts
from feverslop.config.project_config import ProjectConfig


JobHandler = Callable[[Callable[[str], None]], Any]


class ProjectStorePort(Protocol):
    """Minimal project-store surface needed by CLI movie jobs."""

    def resolve_project_path(self, project_id: str, relative: str) -> Path: ...


def build_movie_full_auto_handler(*, store: ProjectStorePort, project_id: str, render_plan_path: Path, movie_config: dict[str, Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        planning = ensure_movie_planning_artifacts(project_dir)
        log(f"[MoviePipeline] Stage: Movie Bible Ready: {planning.bible_path}")
        log(f"[MoviePipeline] Stage: Movie Story Design Ready: {planning.story_design_path}")
        log(f"[MoviePipeline] Stage: Movie Screenplay Ready: {planning.screenplay_path}")
        log(f"[MoviePipeline] Stage: Movie Narrative Ready: {planning.narrative_plan_path}")
        log(f"[MoviePipeline] Stage: Movie Scene Cards Ready: {planning.scene_cards_path}")
        log(f"[MoviePipeline] Stage: Movie Shot Cards Ready: {planning.shot_cards_path}")
        log(f"[MoviePipeline] Stage: Movie Continuity Ready: {planning.continuity_plan_path}")
        log(f"[MoviePipeline] Stage: Render Plan Ready: {planning.render_plan_path}")
        config = movie_runtime_config(movie_config)
        log(f"[MoviePipeline] Planner backend: {config['planner_backend']}")
        log(f"[Krea2_Adapter] Preparing visual consistency references via {config['reference_backend']}")
        manifest_path = ensure_movie_references(project_dir, movie_config=config)
        log(f"[Krea2_Adapter] Reference sheets ready: {manifest_path}")
        if config["movie_video_workflow"] in {"minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"}:
            workflow_key = {
                "minimax-h3-r2v": "r2v_workflow",
                "minimax-h3-t2v": "t2v_workflow",
                "minimax-h3-i2v": "i2v_workflow",
            }[config["movie_video_workflow"]]
            log(f"[MoviePipeline] Stage: MiniMax {config['movie_video_workflow']} render")
            adapter = build_movie_visual_adapter(
                project_dir,
                Path(config[workflow_key]),
                movie_config=config,
            )
            final_video = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=planning.render_plan_path,
                on_clip_rendered=lambda completed, total, scene_number: log(
                    f"[MoviePipeline] Rendered MiniMax clip {completed}/{total}: scene {scene_number}"
                ),
            )
            log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
            return final_video
        if config["movie_video_workflow"] == "i2v-edit":
            from feverslop.adapters.movie_i2v_visual import LocalMovieI2VEditVisualAdapter
            from feverslop.application.movie_i2v_render_plan import write_movie_i2v_render_plan
            from feverslop.application.movie_visual_plan import build_movie_visual_plan
            from feverslop.tools.movie_storyboard_page import generate_movie_storyboard_page

            log("[MoviePipeline] Stage: Movie visual plan")
            visual_plan_path = build_movie_visual_plan(project_dir=project_dir)
            log(f"[MoviePipeline] Stage: Movie visual plan ready: {visual_plan_path}")
            log("[MoviePipeline] Stage: Movie I2V render plan")
            render_plan_i2v_path = write_movie_i2v_render_plan(project_dir=project_dir)
            log(f"[MoviePipeline] Stage: Movie I2V render plan ready: {render_plan_i2v_path}")
            log(f"[MoviePipeline] Stage: Movie I2V/edit render via {config['render_backend']}")
            adapter = (
                LocalMovieI2VEditVisualAdapter()
                if config["render_backend"] == "local"
                else build_movie_i2v_edit_visual_adapter(project_dir, config)
            )
            final_video = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_i2v_path,
                on_startframe_step=lambda event: log(f"[MoviePipeline] {_format_movie_startframe_step(event)}"),
                on_clip_rendered=lambda completed, total, scene_number: log(f"[MoviePipeline] Rendered I2V clip {completed}/{total}: scene {scene_number}"),
            )
            storyboard_page = generate_movie_storyboard_page(project_dir=project_dir)
            log(f"[MoviePipeline] Stage: Storyboard review page: {storyboard_page}")
            log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
            return final_video
        if config["movie_video_workflow"] == "startframe-director":
            from feverslop.adapters.startframe_director_visual import LocalStartframeDirectorVisualAdapter
            from feverslop.application.startframe_director_prompts import build_startframe_director_prompts
            from feverslop.application.startframe_i2v_render_plan import write_startframe_i2v_render_plan
            from feverslop.application.startframe_identity import build_startframe_identity_ledger
            from feverslop.application.startframe_plan import build_startframe_plan
            from feverslop.application.startframe_validation import write_local_startframe_validation

            log("[MoviePipeline] Stage: Movie identity ledger")
            identity_ledger_path = build_startframe_identity_ledger(project_dir=project_dir)
            log(f"[MoviePipeline] Stage: Movie identity ledger ready: {identity_ledger_path}")
            log("[MoviePipeline] Stage: Movie startframe plan")
            startframe_plan_path = build_startframe_plan(project_dir=project_dir)
            log(f"[MoviePipeline] Stage: Movie startframe plan ready: {startframe_plan_path}")
            log(f"[MoviePipeline] Stage: Movie director prompts ({config['startframe_director_backend']})")
            project_config = ProjectConfig.load(project_dir / "config.json")
            prompts_path = build_startframe_director_prompts(
                project_dir=project_dir,
                director_backend=config["startframe_director_backend"],
                reference_image_size=project_config.reference_images.resolve(project_config.video),
            )
            log(f"[MoviePipeline] Stage: Movie director prompts ready: {prompts_path}")
            render_plan_i2v_path = write_startframe_i2v_render_plan(project_dir=project_dir)
            log(f"[MoviePipeline] Stage: Movie I2V render plan ready: {render_plan_i2v_path}")
            adapter = (
                LocalStartframeDirectorVisualAdapter()
                if config["render_backend"] == "local"
                else build_movie_startframe_director_visual_adapter(project_dir, config)
            )
            final_video = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_i2v_path,
                on_startframe_step=lambda event: log(f"[MoviePipeline] {_format_movie_startframe_step(event)}"),
                on_clip_rendered=lambda completed, total, scene_number: log(f"[MoviePipeline] Rendered I2V clip {completed}/{total}: scene {scene_number}"),
            )
            validation_path = (
                write_local_startframe_validation(project_dir=project_dir)
                if config["render_backend"] == "local"
                else project_dir / "movie" / "startframe_validation.json"
            )
            log(f"[MoviePipeline] Stage: Movie startframe validation ready: {validation_path}")
            log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
            return final_video
        render_plan_msr_path = enrich_movie_render_plan_with_msr_prompts(project_dir=project_dir, keyframe_mode=config["keyframe_mode"])
        log(f"[MoviePipeline] Stage: Movie MSR Prompt Enrichment Ready: {render_plan_msr_path}")
        patched_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        patched_i2v_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_i2v_workflow"])) if config.get("msr_i2v_workflow") else None
        log(f"[WorkflowPatcher] Movie MSR workflow patched in memory for LTX native audio from: {config['msr_workflow']}")
        log(f"[LTX_MSR_Movie_Adapter] Rendering via {config['render_backend']} with LTX 2.3 native audio; no custom audio track supplied")
        final_video = build_movie_visual_adapter(
            project_dir,
            Path(config["msr_workflow"]),
            movie_config=config,
            workflow=patched_workflow,
            i2v_workflow=patched_i2v_workflow,
        ).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_msr_path if render_plan_msr_path.exists() else planning.render_plan_path,
            continuity_keyframes=config["continuity_keyframes"],
            on_clip_rendered=lambda completed, total, scene_number: log(f"[MoviePipeline] Rendered clip {completed}/{total}: scene {scene_number}"),
        )
        log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
        return final_video

    return run


def build_movie_render_handler(
    *,
    store: ProjectStorePort,
    project_id: str,
    render_plan_path: Path,
    movie_config: dict[str, Any] | None = None,
    selected_scenes: list[int] | None = None,
    concat_only: bool = False,
) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        config = movie_runtime_config(movie_config)
        log(f"[MoviePipeline] Stage: Existing Movie MSR plan: {render_plan_path}")
        if config["movie_video_workflow"] in {"minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"}:
            workflow_key = {
                "minimax-h3-r2v": "r2v_workflow",
                "minimax-h3-t2v": "t2v_workflow",
                "minimax-h3-i2v": "i2v_workflow",
            }[config["movie_video_workflow"]]
            adapter = build_movie_visual_adapter(
                project_dir,
                Path(config[workflow_key]),
                movie_config=config,
            )
            final_video = adapter.render_movie(
                project_dir=project_dir,
                render_plan_path=render_plan_path,
                selected_scenes=selected_scenes,
                concat_only=concat_only,
                on_clip_rendered=lambda completed, total, scene_number: log(
                    f"[MoviePipeline] Rendered MiniMax clip {completed}/{total}: scene {scene_number}"
                ),
            )
            log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
            return final_video
        patched_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        patched_i2v_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_i2v_workflow"])) if config.get("msr_i2v_workflow") else None
        log(f"[WorkflowPatcher] Movie MSR workflow patched in memory for LTX native audio from: {config['msr_workflow']}")
        if concat_only:
            log("[MoviePipeline] Stage: Final movie concat from existing scene clips")
        else:
            log(f"[LTX_MSR_Movie_Adapter] Rendering via {config['render_backend']} with LTX 2.3 native audio; no custom audio track supplied")
        final_video = build_movie_visual_adapter(
            project_dir,
            Path(config["msr_workflow"]),
            movie_config=config,
            workflow=patched_workflow,
            i2v_workflow=patched_i2v_workflow,
        ).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_path,
            selected_scenes=selected_scenes,
            concat_only=concat_only,
            continuity_keyframes=config["continuity_keyframes"],
            on_clip_rendered=lambda completed, total, scene_number: log(f"[MoviePipeline] Rendered clip {completed}/{total}: scene {scene_number}"),
        )
        log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
        return final_video

    return run


def ensure_movie_references(project_dir: Path, *, movie_config: dict[str, Any] | None = None) -> Path:
    config = movie_runtime_config(movie_config)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    ensure_movie_planning_artifacts(project_dir)
    write_movie_reference_manifest_from_bible(project_dir)
    if movie_references_ready(manifest_path, backend=config["reference_backend"]):
        return manifest_path
    return mark_movie_reference_backend(
        build_movie_reference_generator(movie_config=config).generate(project_dir=project_dir),
        config["reference_backend"],
    )


def _format_movie_startframe_step(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "step")
    completed = int(event.get("completed") or 0)
    total = int(event.get("total") or 0)
    scene = int(event.get("scene") or 0)
    actor_id = str(event.get("actor_id") or "").strip()
    actor_suffix = f" actor {actor_id}" if actor_id else ""
    return f"Rendered startframe {kind} {completed}/{total}: scene {scene}{actor_suffix}"


def movie_references_ready(manifest_path: Path, *, backend: str | None = None) -> bool:
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actors = manifest.get("actors") or []
    locations = manifest.get("locations") or []
    if not actors or not locations:
        return False
    if not all(str(item.get("msr_sheet_path") or "").strip() for item in [*actors, *locations]):
        return False
    return not backend or manifest.get("generator_backend") == backend


def build_movie_references_handler(*, store: ProjectStorePort, project_id: str, movie_config: dict[str, Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        config = movie_runtime_config(movie_config)
        log(f"[MoviePipeline] Stage: Movie references via {config['reference_backend']}")
        ensure_movie_planning_artifacts(project_dir)
        write_movie_reference_manifest_from_bible(project_dir)
        manifest_path = mark_movie_reference_backend(
            build_movie_reference_generator(movie_config=config).generate(project_dir=project_dir),
            config["reference_backend"],
        )
        log(f"[Krea2_Adapter] Reference sheets ready: {manifest_path}")
        return manifest_path

    return run


def sync_movie_manifest_with_render_plan(project_dir: Path) -> Path:
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not manifest_path.exists() or not render_plan_path.exists():
        return manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shots = _movie_plan_shots(json.loads(render_plan_path.read_text(encoding="utf-8")))
    actors = manifest.setdefault("actors", [])
    locations = manifest.setdefault("locations", [])
    actor_map = {str(actor.get("id")): actor for actor in actors if isinstance(actor, dict)}
    location_map = {str(location.get("id")): location for location in locations if isinstance(location, dict)}
    actor_shots = _movie_actor_shots_by_id(shots)
    changed = False
    for shot in shots:
        refs = shot.get("reference_ids") or {}
        for actor_id in refs.get("actors") or []:
            actor_id = str(actor_id or "").strip()
            if not actor_id:
                continue
            if actor_id not in actor_map:
                actor_map[actor_id] = _movie_manifest_ref(actor_id, kind="actor", shots=actor_shots.get(actor_id) or [shot])
                actors.append(actor_map[actor_id])
                changed = True
            elif _needs_movie_prompt_repair(actor_map[actor_id]):
                _repair_movie_manifest_ref(actor_map[actor_id], kind="actor", shots=actor_shots.get(actor_id) or [shot])
                changed = True
        location_id = str(refs.get("location") or "").strip()
        if not location_id:
            continue
        if location_id not in location_map:
            location_map[location_id] = _movie_manifest_ref(location_id, kind="location", name=str(shot.get("location") or ""), shot=shot)
            locations.append(location_map[location_id])
            changed = True
        elif _needs_movie_prompt_repair(location_map[location_id]):
            _repair_movie_manifest_ref(location_map[location_id], kind="location", name=str(shot.get("location") or ""), shot=shot)
            changed = True
    if changed:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _movie_plan_shots(plan: Any) -> list[dict[str, Any]]:
    if isinstance(plan, list):
        return [shot for shot in plan if isinstance(shot, dict)]
    if isinstance(plan, dict):
        shots = plan.get("shots") or plan.get("scenes") or []
        return [shot for shot in shots if isinstance(shot, dict)]
    return []


def _movie_manifest_ref(
    ref_id: str,
    *,
    kind: str,
    name: str = "",
    shot: dict[str, Any] | None = None,
    shots: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    display_name = name.strip() or ref_id.replace("_", " ").title()
    visual_description, prompt = _movie_reference_fields(display_name, kind=kind, shot=shot, shots=shots)
    return {
        "id": ref_id,
        "name": display_name,
        "role": "",
        "visual_description": visual_description,
        "image_prompt": prompt,
        "prompt": prompt,
        "status": "required",
        "msr_sheet_path": "",
    }


def _repair_movie_manifest_ref(
    ref: dict[str, Any],
    *,
    kind: str,
    name: str = "",
    shot: dict[str, Any] | None = None,
    shots: list[dict[str, Any]] | None = None,
) -> None:
    display_name = name.strip() or str(ref.get("name") or ref.get("id") or "").replace("_", " ").title()
    visual_description, prompt = _movie_reference_fields(display_name, kind=kind, shot=shot, shots=shots)
    ref["name"] = display_name
    ref["visual_description"] = visual_description
    ref["image_prompt"] = prompt
    ref["prompt"] = prompt
    ref["msr_sheet_path"] = ""
    ref.pop("sheet_path", None)


def _needs_movie_prompt_repair(ref: dict[str, Any]) -> bool:
    prompt = str(ref.get("prompt") or "").strip().lower()
    visual_description = str(ref.get("visual_description") or "").strip().lower()
    combined = f"{visual_description}\n{prompt}"
    return (
        not prompt
        or (prompt.startswith("consistent cinematic") and "drawn from the story premise" in prompt)
        or "visual identity" in combined
        or "full-body cinematic character reference sheet" in visual_description
        or "environment reference sheet" in combined
        or "four vertical panels" in visual_description
        or any(
            token in combined
            for token in (
                "jump cut",
                "lunges",
                "bellows",
                "low-angle shot",
                "medium shot",
                "wide shot",
                "close-up",
                "eye fluttering",
                "eyes roll back",
                "screen fades",
                "mesmerized",
                "gaze",
                "reaches",
                "'s;",
            )
        )
    )


def _movie_reference_fields(
    display_name: str,
    *,
    kind: str,
    shot: dict[str, Any] | None = None,
    shots: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    if kind == "actor":
        actor_shots = shots or ([shot] if shot else [])
        visual_description = build_movie_actor_visual_description(_movie_actor_shot_cues(actor_shots))
        return visual_description, build_movie_actor_reference_prompt(display_name, visual_description)
    cues = _movie_shot_cues(shot or {})
    prompt = f"Cinematic wide establishing photograph of {display_name}"
    if cues:
        prompt += f". Environment and mood from scenes: {cues}"
    prompt = (
        f"{prompt}. Wide establishing view, production design, lighting, atmosphere, "
        "single continuous image, no collage, no split screen, no panels, no people, no text."
    )
    return prompt, prompt


def _movie_actor_shots_by_id(shots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    actor_shots: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        actor_ids = [str(actor_id or "").strip() for actor_id in (shot.get("reference_ids") or {}).get("actors") or []]
        for actor_id in actor_ids:
            if actor_id:
                actor_shots.setdefault(actor_id, []).append(shot)
    for actor_id, items in list(actor_shots.items()):
        solo = [shot for shot in items if len((shot.get("reference_ids") or {}).get("actors") or []) == 1]
        if solo:
            actor_shots[actor_id] = solo
    return actor_shots


def _movie_actor_shot_cues(shots: list[dict[str, Any]]) -> str:
    parts = []
    for shot in shots[:4]:
        for key in ("description", "expression"):
            value = str(shot.get(key) or "").strip()
            if value and value not in parts:
                parts.append(value)
    return "; ".join(parts)[:700]


def _movie_shot_cues(shot: dict[str, Any]) -> str:
    parts = []
    for key in ("description", "action", "expression", "location", "dialogue"):
        value = str(shot.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    return "; ".join(parts)[:700]


def build_movie_reference_generator(movie_config: dict[str, Any] | None = None):
    backend = movie_runtime_config(movie_config)["reference_backend"]
    if backend == "local":
        from feverslop.adapters.movie_references import LocalMovieImageBackend
        from feverslop.application.movie_references import MovieReferenceSheetGenerator

        local = LocalMovieImageBackend()
        return MovieReferenceSheetGenerator(backend=local, edit_backend=local)

    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
    from feverslop.adapters.sequence_to_sheet_backend import ComfyUISequenceToSheetBackend
    from feverslop.application.movie_references import MovieReferenceSheetGenerator
    from feverslop.application.reference_sheet_planning import ReferenceSheetPlanner
    from feverslop.adapters.llm_client import LocalOpenAIClient
    from feverslop.config.app_config import AppConfig
    from feverslop.ports.rendering import WorkflowAnchorConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    hero = ComfyUIImageBackend(
        client=client,
        workflow_path=backend_config_path(movie_runtime_config(movie_config)["hero_workflow"]),
        output_dir=Path("."),
        model_resolver=resolver,
    )
    edit = ComfyUIImageBackend(
        client=client,
        workflow_path=backend_config_path(movie_runtime_config(movie_config)["edit_workflow"]),
        output_dir=Path("."),
        model_resolver=resolver,
    )
    sequence_backend = None
    sequence_planner = None
    if movie_runtime_config(movie_config)["reference_generation"] == "sequence_sheet":
        sequence_backend = ComfyUISequenceToSheetBackend(
            client=client,
            workflow_path=backend_config_path(movie_runtime_config(movie_config)["sequence_to_sheet_workflow"]),
            backend="minimax",
            model_resolver=resolver,
        )
        sequence_planner = ReferenceSheetPlanner(
            llm=LocalOpenAIClient(
                base_url=app_config.llm.base_url,
                api_key=app_config.llm.api_key,
                model=app_config.llm.model_for("structured"),
                temperature=app_config.llm.temperature,
                dspy_temperature=app_config.llm.dspy_temperature,
                max_tokens=app_config.llm.max_tokens,
                request_timeout_seconds=app_config.llm.request_timeout_seconds,
                dspy_cache=app_config.llm.dspy_cache,
                max_concurrent_requests=app_config.llm.max_concurrent_requests,
            )
        )
    return MovieReferenceSheetGenerator(
        backend=hero,
        edit_backend=edit,
        sequence_backend=sequence_backend,
        sequence_planner=sequence_planner,
        hero_anchors=WorkflowAnchorConfig(positive_prompt_title="#PROMPT_POSITIVE"),
        edit_anchors=WorkflowAnchorConfig(positive_prompt_title="#PROMPT_POSITIVE", reference_image_title="#IMAGE_1"),
    )


def build_movie_visual_adapter(
    project_dir: Path,
    workflow_path: Path,
    movie_config: dict[str, Any] | None = None,
    workflow: dict | None = None,
    i2v_workflow: dict | None = None,
):
    config = movie_runtime_config(movie_config)
    if config["movie_video_workflow"] in {"minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"}:
        from feverslop.adapters.movie_minimax_visual import ComfyUIMiniMaxMovieVisualAdapter

        workflow_key = {
            "minimax-h3-r2v": "r2v_workflow",
            "minimax-h3-t2v": "t2v_workflow",
            "minimax-h3-i2v": "i2v_workflow",
        }[config["movie_video_workflow"]]
        return ComfyUIMiniMaxMovieVisualAdapter(
            project_dir=project_dir,
            workflow_path=config[workflow_key],
            video_pipeline=config["movie_video_workflow"],
            app_config_path=config.get("app_config_path", "app_config.json"),
        )
    backend = config["render_backend"]
    if backend == "local":
        from feverslop.adapters.movie_visual import LocalMovieVisualAdapter

        return LocalMovieVisualAdapter()

    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
    from feverslop.adapters.postprocessor_frame_extractor import (
        PostprocessorFrameExtractor,
    )
    from feverslop.application.continuity_handoff import (
        ContinuityHandoffUseCase,
    )
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    return ComfyUIMovieVisualAdapter(
        client=client,
        workflow_path=workflow_path,
        workflow=workflow,
        i2v_workflow_path=Path(config["msr_i2v_workflow"]) if config.get("msr_i2v_workflow") else None,
        i2v_workflow=i2v_workflow,
        continuity_keyframes=config["continuity_keyframes"],
        continuity_handoff_factory=lambda postprocessor, root, selected: (
            ContinuityHandoffUseCase(
                PostprocessorFrameExtractor(
                    postprocessor,
                    project_dir=root,
                    selected_rerender=selected,
                )
            )
        ),
        model_resolver=ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides),
    )


def build_movie_i2v_edit_visual_adapter(project_dir: Path, config: dict[str, Any]):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
    from feverslop.adapters.local_artifacts import JsonArtifactStore
    from feverslop.adapters.movie_edit_image_backend import MovieTwoRefEditImageBackend
    from feverslop.adapters.movie_i2v_visual import ComfyUIMovieI2VEditVisualAdapter
    from feverslop.adapters.video_postprocessor import VideoPostProcessor
    from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=str(config.get("startframe_comfyui_base_url") or app_config.comfyui.base_url),
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    ltx_dir = project_dir / "output" / "movie" / "ltx_i2v"
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            workflow_path=config["i2v_workflow"],
            single_prompt_workflow_path=config["i2v_workflow"],
            output_dir=ltx_dir,
            video_pipeline="ltx_i2v",
        )
    )
    return ComfyUIMovieI2VEditVisualAdapter(
        base_image_backend=ComfyUIImageBackend(
            client=client,
            workflow_path=backend_config_path(config["hero_workflow"]),
            output_dir=project_dir / "output" / "movie" / "storyboard" / "base",
            model_resolver=model_resolver,
        ),
        edit_backend=MovieTwoRefEditImageBackend(
            client=client,
            workflow_path=backend_config_path(config["edit_workflow"]),
            model_resolver=model_resolver,
        ),
        artifact_store=JsonArtifactStore(),
        video_use_case=video_use_case,
        workflow_path=Path(config["hero_workflow"]),
        edit_workflow_path=Path(config["edit_workflow"]),
        i2v_workflow_path=Path(config["i2v_workflow"]),
        postprocessor=VideoPostProcessor(),
    )


def build_movie_startframe_director_visual_adapter(project_dir: Path, config: dict[str, Any]):
    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.gemma4_startframe_validator import Gemma4StartframeValidator
    from feverslop.adapters.movie_workflow import MovieWorkflowPatcher
    from feverslop.adapters.startframe_director_comfyui import ComfyUIStartframeDirectorVisualAdapter
    from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    ltx_dir = project_dir / "output" / "movie" / "ltx_startframe_director"
    i2v_workflow_path = write_startframe_i2v_empty_audio_workflow(
        project_dir=project_dir,
        workflow_path=Path(config["i2v_workflow"]),
        patcher=MovieWorkflowPatcher(),
    )
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            workflow_path=i2v_workflow_path,
            single_prompt_workflow_path=i2v_workflow_path,
            output_dir=ltx_dir,
            video_pipeline="ltx_i2v",
            debug_workflows_dir=_startframe_debug_workflows_dir(project_dir, config),
        )
    )
    return ComfyUIStartframeDirectorVisualAdapter(
        client=client,
        director_workflow_path=backend_config_path(config["director_workflow"]),
        mask_workflow_path=backend_config_path(config["mask_workflow"]),
        identity_repair_workflow_path=backend_config_path(config["identity_repair_workflow"]),
        detail_workflow_path=backend_config_path(config["detail_workflow"]),
        i2v_workflow_path=i2v_workflow_path,
        video_use_case=video_use_case,
        validator=Gemma4StartframeValidator(
            base_url=config["startframe_validator_base_url"],
            model=config["startframe_validator_model"],
        ),
        debug_workflows_dir=_startframe_debug_workflows_dir(project_dir, config),
    )


def movie_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return movie_runtime_config(dict(metadata.get("movie") or {}))


def write_startframe_i2v_empty_audio_workflow(*, project_dir: Path, workflow_path: Path, patcher) -> Path:
    workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8-sig"))
    stripped = patcher.strip_audio_inputs(workflow)
    output = project_dir / "output" / "movie" / "startframes" / "workflows" / "ltx_i2v_empty_audio.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stripped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def _startframe_debug_workflows_dir(project_dir: Path, config: dict[str, Any]) -> Path | None:
    if not config.get("startframe_write_debug_workflows"):
        return None
    raw = str(config.get("startframe_debug_workflows_dir") or "").strip()
    if not raw:
        return project_dir / "output" / "movie" / "startframes" / "debug_workflows"
    path = Path(raw)
    if path.is_absolute():
        return path
    return project_dir / path


def movie_runtime_config(config: dict[str, Any] | None = None) -> dict[str, str]:
    raw = dict(config or {})
    planner_backend = _movie_backend(raw.get("planner_backend"), default="llm", supported={"llm", "deterministic", "local"})
    if planner_backend == "local":
        planner_backend = "deterministic"
    movie_video_workflow = _movie_backend(raw.get("movie_video_workflow"), default="msr", supported={"msr", "msr-i2v-startframe", "i2v-edit", "startframe-director", "ingredients", "minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"})
    msr_i2v_default = "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json" if movie_video_workflow == "msr-i2v-startframe" else ""
    i2v_default = "workflows/video_ltxv_i2v_native_audio_v2.json" if movie_video_workflow == "startframe-director" else "workflows/video_ltxv_i2v_v2.json"
    if movie_video_workflow in {"minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"}:
        i2v_default = "workflows/video_minimax_h3_t2v.json"
    edit_workflow_default = "workflows/image_edit_flux2_klein_2ref_v1.json" if movie_video_workflow == "i2v-edit" else "workflows/image_edit_flux2_klein_1ref_v1.json"
    ingredients_default = "workflows/video_ltxv_ingredients_2stage_v6.json" if movie_video_workflow == "ingredients" else ""
    return {
        "planner_backend": planner_backend,
        "reference_backend": _movie_backend(raw.get("reference_backend"), default="comfyui", supported={"comfyui", "local"}),
        "reference_generation": _movie_backend(raw.get("reference_generation"), default="image_views", supported={"image_views", "sequence_sheet"}),
        "render_backend": _movie_backend(raw.get("render_backend"), default="comfyui", supported={"comfyui", "local"}),
        "hero_workflow": _movie_workflow_path(raw.get("hero_workflow"), "workflows/image_t2i_startframe_krea_v1.json"),
        "edit_workflow": _movie_workflow_path(raw.get("edit_workflow"), edit_workflow_default),
        "startframe_director_backend": _movie_backend(raw.get("startframe_director_backend"), default="krea2", supported={"krea2", "ideogram"}),
        "director_workflow": _movie_workflow_path(
            raw.get("director_workflow"),
            _default_startframe_director_workflow(raw.get("startframe_director_backend")),
        ),
        "mask_workflow": _movie_workflow_path(raw.get("mask_workflow"), "workflows/image_mask_sam3_actor_regions_v1.json"),
        "identity_repair_workflow": _movie_workflow_path(raw.get("identity_repair_workflow"), "workflows/image_repair_sdxl_ipadapter_identity_v1.json"),
        "detail_workflow": _movie_workflow_path(raw.get("detail_workflow"), "workflows/image_detail_easyuse_startframe_v1.json"),
        "startframe_comfyui_base_url": str(raw.get("startframe_comfyui_base_url") or "http://localhost:8188").rstrip("/"),
        "startframe_write_debug_workflows": bool(raw.get("startframe_write_debug_workflows", False)),
        "startframe_debug_workflows_dir": str(raw.get("startframe_debug_workflows_dir") or ""),
        "startframe_validator_base_url": str(raw.get("startframe_validator_base_url") or "http://your-llm-server.local/v1").rstrip("/"),
        "startframe_validator_model": str(raw.get("startframe_validator_model") or "gemma4-26b-a4b:vision"),
        "msr_workflow": _movie_workflow_path(raw.get("msr_workflow"), "workflows/video_default_ltxv_msr_1actor_1background_v4.json"),
        "msr_i2v_workflow": _movie_workflow_path(raw.get("msr_i2v_workflow"), msr_i2v_default) if msr_i2v_default or raw.get("msr_i2v_workflow") else "",
        "i2v_workflow": _movie_workflow_path(raw.get("i2v_workflow"), i2v_default),
        "r2v_workflow": _movie_workflow_path(raw.get("r2v_workflow"), "workflows/video_minimax_h3_r2v.json"),
        "t2v_workflow": _movie_workflow_path(raw.get("t2v_workflow"), "workflows/video_minimax_h3_t2v.json"),
        "sequence_to_sheet_workflow": _movie_workflow_path(raw.get("sequence_to_sheet_workflow"), "workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"),
        "ingredients_workflow": _movie_workflow_path(raw.get("ingredients_workflow"), ingredients_default) if ingredients_default or raw.get("ingredients_workflow") else "",
        "movie_video_workflow": movie_video_workflow,
        "keyframe_mode": _movie_backend(raw.get("keyframe_mode"), default="none", supported={"none", "start", "start-end"}),
        "continuity_keyframes": _movie_continuity_keyframes(raw.get("continuity_keyframes"), movie_video_workflow=raw.get("movie_video_workflow")),
        "refine_location_prompts": bool(raw.get("refine_location_prompts", False)),
    }


def _movie_continuity_keyframes(value: object, *, movie_video_workflow: object = None) -> str:
    mode = _movie_backend(value, default="none", supported={"none", "last-to-start"})
    workflow = _movie_backend(movie_video_workflow, default="msr", supported={"msr", "msr-i2v-startframe", "i2v-edit", "startframe-director", "ingredients", "minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"})
    if mode == "last-to-start" and workflow != "msr-i2v-startframe":
        raise ValueError("continuity_keyframes=last-to-start requires movie_video_workflow=msr-i2v-startframe")
    return mode


def mark_movie_reference_backend(manifest_path: Path, backend: str) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest["generator_backend"] = backend
    Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return Path(manifest_path)


def _movie_backend(value: object, *, default: str, supported: set[str]) -> str:
    backend = str(value or default).strip().lower()
    if backend == "placeholder":
        backend = "local"
    if backend not in supported:
        raise ValueError(f"movie backend must be one of: {', '.join(sorted(supported))}")
    return backend


def _movie_workflow_path(value: object, default: str) -> str:
    raw = str(value or default).strip()
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("movie workflow paths must be repository-relative")
    return path.as_posix()


def _default_startframe_director_workflow(backend: object) -> str:
    if _movie_backend(backend, default="krea2", supported={"krea2", "ideogram"}) == "ideogram":
        return "workflows/image_t2i_startframe_ideogram_director_v1.json"
    return "workflows/image_t2i_startframe_krea_v1.json"


def backend_config_path(value: str) -> str:
    return Path(value).as_posix()
