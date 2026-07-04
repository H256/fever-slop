from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Protocol

from feverslop.studio.jobs import (
    PIPELINE_ACTIONS,
    JobHandler,
    JobRegistry,
    build_pipeline_handler,
    build_pipeline_options,
    build_recut_scene_handler,
    build_reference_rerender_handler,
    run_with_stream_logging,
)
from feverslop.application.movie import build_movie_actor_reference_prompt, build_movie_actor_visual_description
from feverslop.application.movie_artifacts import ensure_movie_bible, write_movie_reference_manifest_from_bible
from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts
from feverslop.studio.logging import render_log_lines
from feverslop.studio.projects import ProjectStore


FullAutoHandlerFactory = Callable[..., Any]
PipelineHandlerFactory = Callable[..., Any]


@dataclass(frozen=True)
class StudioJobRequest:
    action: str
    scenes: list[int] | None = None
    pipeline_mode: str | None = None
    thumbnails: list[dict[str, Any]] | None = None
    reference_kind: str | None = None
    reference_id: str | None = None
    raw_clip: str | None = None
    output_clip: str | None = None
    raw_in_seconds: float | None = None
    raw_out_seconds: float | None = None
    exact: bool = False


class ActionHandler(Protocol):
    action: str

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        """Build a runnable job handler."""


class ActionRegistry:
    def __init__(self, handlers: list[ActionHandler], fallback: ActionHandler):
        self._handlers = {handler.action: handler for handler in handlers}
        self._fallback = fallback

    def resolve(self, action: str) -> ActionHandler:
        return self._handlers.get(action, self._fallback)


class StudioJobService:
    def __init__(
        self,
        *,
        store: ProjectStore,
        jobs: JobRegistry,
        full_auto_handler: FullAutoHandlerFactory | None = None,
        pipeline_handler: PipelineHandlerFactory | None = None,
    ):
        self.store = store
        self.jobs = jobs
        self.registry = ActionRegistry(
            [
                ReferenceRerenderAction(store),
                RecutSceneAction(store),
                ThumbnailPrebuildAction(store),
                ThumbnailCleanupAction(store),
                FullAutoAction(store, full_auto_handler),
                MovieReferencesAction(store),
                MovieRenderAction(store),
                MovieFullAutoAction(store),
            ],
            PipelineAction(store, pipeline_handler),
        )

    def start_job(self, project_id: str, request: StudioJobRequest) -> dict[str, Any]:
        metadata = self.store.project_metadata(project_id)
        handler = self.registry.resolve(request.action).build(project_id, request, metadata)
        return self.jobs.get(
            self.jobs.start(
                project_id,
                request.action,
                handler,
                project_type=str(metadata.get("project_type", "standard_music_video")),
                reject_if_project_active=request.action in PIPELINE_ACTIONS,
            )
        )


class ReferenceRerenderAction:
    action = "reference-rerender"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        config_path = self.store.resolve_project_path(project_id, "config.json")
        if request.reference_kind not in {"actor", "location"} or not request.reference_id:
            raise ValueError("reference-rerender requires reference_kind and reference_id")
        return build_reference_rerender_handler(
            config_path,
            reference_kind=request.reference_kind,
            reference_id=request.reference_id,
        )


class RecutSceneAction:
    action = "recut-scene"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if (
            not request.raw_clip
            or not request.output_clip
            or request.raw_in_seconds is None
            or request.raw_out_seconds is None
        ):
            raise ValueError("recut-scene requires raw_clip, output_clip, raw_in_seconds, and raw_out_seconds")
        return build_recut_scene_handler(
            self.store.resolve_project_path(project_id, request.raw_clip),
            self.store.resolve_project_path(project_id, request.output_clip),
            raw_in_seconds=request.raw_in_seconds,
            raw_out_seconds=request.raw_out_seconds,
            exact=request.exact,
        )


class ThumbnailPrebuildAction:
    action = "thumbnail-prebuild"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        thumbnails = request.thumbnails or []

        def run(log: Callable[[str], None]) -> int:
            count = 0
            for thumbnail in thumbnails:
                path = str(thumbnail.get("path") or "")
                for second in thumbnail.get("times") or []:
                    thumbnail_path(self.store, project_id, path, float(second))
                    count += 1
            log(f"Generated {count} thumbnails")
            return count

        return run


class ThumbnailCleanupAction:
    action = "thumbnail-cleanup"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        def run(log: Callable[[str], None]) -> int:
            log("Clearing thumbnail cache")
            return self.store.clear_thumbnail_cache(project_id)

        return run


class FullAutoAction:
    action = "full-auto"

    def __init__(self, store: ProjectStore, factory: FullAutoHandlerFactory | None):
        self.store = store
        self.factory = factory

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "full_auto":
            raise ValueError("full-auto jobs require a full_auto project")
        factory = self.factory or build_full_auto_handler
        return factory(
            store=self.store,
            project_id=project_id,
            payload={**dict(metadata.get("full_auto") or {}), "silent_mode": bool(metadata.get("silent_mode", False))},
        )


class MovieFullAutoAction:
    action = "movie-full-auto"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-full-auto jobs require a movie project")
        render_plan_path = self.store.resolve_project_path(project_id, "movie/render_plan.json")
        if not render_plan_path.exists():
            raise ValueError("movie-full-auto requires movie/render_plan.json; create the movie scaffold first")
        handler = build_movie_full_auto_handler(
            store=self.store,
            project_id=project_id,
            render_plan_path=render_plan_path,
            movie_config=movie_config_from_metadata(metadata),
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class MovieReferencesAction:
    action = "movie-references"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-references jobs require a movie project")
        manifest_path = self.store.resolve_project_path(project_id, "movie/references/manifest.json")
        if not manifest_path.exists():
            raise ValueError("movie-references requires movie/references/manifest.json; create the movie scaffold first")
        handler = build_movie_references_handler(
            store=self.store,
            project_id=project_id,
            movie_config=movie_config_from_metadata(metadata),
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class MovieRenderAction:
    action = "movie-render"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-render jobs require a movie project")
        render_plan_msr_path = self.store.resolve_project_path(project_id, "movie/render_plan_msr.json")
        if not render_plan_msr_path.exists():
            raise ValueError("movie-render requires movie/render_plan_msr.json; run Movie full-auto or CLI MSR enrichment first")
        manifest_path = self.store.resolve_project_path(project_id, "movie/references/manifest.json")
        config = movie_config_from_metadata(metadata)
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("movie-render requires ready movie references; run Movie references first")
        handler = build_movie_render_handler(
            store=self.store,
            project_id=project_id,
            render_plan_path=render_plan_msr_path,
            movie_config=config,
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class PipelineAction:
    action = "*"

    def __init__(self, store: ProjectStore, factory: PipelineHandlerFactory | None):
        self.store = store
        self.factory = factory

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        config_path = self.store.resolve_project_path(project_id, "config.json")
        factory = self.factory or build_pipeline_handler
        pipeline_mode = request.pipeline_mode or pipeline_mode_from_config(config_path)
        handler = factory(config_path, request.action, scenes=request.scenes, pipeline_mode=pipeline_mode)
        return record_pipeline_state(
            self.store,
            project_id,
            request.action,
            handler,
            scenes=request.scenes,
            pipeline_mode=pipeline_mode,
        )


def record_pipeline_state(
    store: ProjectStore,
    project_id: str,
    action: str,
    handler: JobHandler,
    *,
    scenes: list[int] | None = None,
    pipeline_mode: str | None = None,
) -> JobHandler:
    if action not in PIPELINE_ACTIONS:
        return handler

    def run(log: Callable[[str], None]) -> Any:
        try:
            result = handler(log)
        except Exception:
            store.record_pipeline_run(project_id, action=action, stages=pipeline_state_stages(action, scenes=scenes, pipeline_mode=pipeline_mode), status="failed")
            raise
        store.record_pipeline_run(project_id, action=action, stages=pipeline_state_stages(action, scenes=scenes, pipeline_mode=pipeline_mode), status="succeeded")
        return result

    return run


def pipeline_state_stages(action: str, *, scenes: list[int] | None = None, pipeline_mode: str | None = None) -> list[str]:
    try:
        options = build_pipeline_options(action, scenes=scenes, pipeline_mode=pipeline_mode)
    except ValueError:
        return [action]
    stages = options.get("stages")
    if isinstance(stages, list):
        return [str(stage) for stage in stages]
    return [action]


class StudioFullAutoConsole:
    def __init__(self, log: Callable[[str], None]):
        self.log = log

    def print(self, *values: object, **_kwargs: object) -> None:
        for line in render_log_lines(*values):
            self.log(line)

    def rule(self, title: str) -> None:
        for line in render_log_lines(str(title)):
            self.log(line)


def build_full_auto_handler(*, store: ProjectStore, project_id: str, payload: dict[str, Any], use_case_factory: Callable[[Any], Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        from feverslop.application.full_auto import FullAutoRequest
        from feverslop.composition.full_auto import build_full_auto_use_case

        console = StudioFullAutoConsole(log)
        use_case = use_case_factory(console) if use_case_factory else build_full_auto_use_case(console=console)
        pipeline_mode = str(payload.get("pipeline_mode") or "classic")
        request = FullAutoRequest(
            idea=str(payload.get("idea") or ""),
            style=str(payload.get("song_style") or ""),
            project_name=project_id,
            projects_dir=store.projects_root,
            duration_seconds=float(payload.get("duration_seconds") or 120.0),
            width=int(payload.get("width") or 1280),
            height=int(payload.get("height") or 704),
            fps=int(payload.get("fps") or 24),
            silent_mode=bool(payload.get("silent_mode", False)),
            run_video_pipeline=True,
            runner_options={"skip_tests": True, "video_pipeline": "ltx_msr" if pipeline_mode == "msr" else "ltx_i2v"},
        )
        return run_with_stream_logging(lambda: use_case.execute(request), log).project_config_path

    return run


def build_movie_full_auto_handler(*, store: ProjectStore, project_id: str, render_plan_path: Path, movie_config: dict[str, Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        bible_path = ensure_movie_bible(project_dir)
        log(f"[MoviePipeline] Stage: Movie Bible Ready: {bible_path}")
        log("[MoviePipeline] Stage: Render Plan Ready")
        config = movie_runtime_config(movie_config)
        log(f"[MoviePipeline] Planner backend: {config['planner_backend']}")
        log(f"[Krea2_Adapter] Preparing visual consistency references via {config['reference_backend']}")
        manifest_path = ensure_movie_references(project_dir, movie_config=config)
        log(f"[Krea2_Adapter] Reference sheets ready: {manifest_path}")
        render_plan_msr_path = enrich_movie_render_plan_with_msr_prompts(project_dir=project_dir)
        log(f"[MoviePipeline] Stage: Movie MSR Prompt Enrichment Ready: {render_plan_msr_path}")
        patched_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        log(f"[WorkflowPatcher] Movie MSR workflow patched in memory for LTX native audio from: {config['msr_workflow']}")
        log(f"[LTX_MSR_Movie_Adapter] Rendering via {config['render_backend']} with LTX 2.3 native audio; no custom audio track supplied")
        final_video = build_movie_visual_adapter(project_dir, Path(config["msr_workflow"]), movie_config=config, workflow=patched_workflow).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_msr_path if render_plan_msr_path.exists() else render_plan_path,
        )
        log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
        return final_video

    return run


def build_movie_render_handler(*, store: ProjectStore, project_id: str, render_plan_path: Path, movie_config: dict[str, Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        config = movie_runtime_config(movie_config)
        log(f"[MoviePipeline] Stage: Existing Movie MSR plan: {render_plan_path}")
        patched_workflow = patch_movie_msr_workflow(template_path=Path(config["msr_workflow"]))
        log(f"[WorkflowPatcher] Movie MSR workflow patched in memory for LTX native audio from: {config['msr_workflow']}")
        log(f"[LTX_MSR_Movie_Adapter] Rendering via {config['render_backend']} with LTX 2.3 native audio; no custom audio track supplied")
        final_video = build_movie_visual_adapter(project_dir, Path(config["msr_workflow"]), movie_config=config, workflow=patched_workflow).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_path,
        )
        log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
        return final_video

    return run


def ensure_movie_references(project_dir: Path, *, movie_config: dict[str, Any] | None = None) -> Path:
    config = movie_runtime_config(movie_config)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    ensure_movie_bible(project_dir)
    write_movie_reference_manifest_from_bible(project_dir)
    if movie_references_ready(manifest_path, backend=config["reference_backend"]):
        return manifest_path
    return mark_movie_reference_backend(
        build_movie_reference_generator(movie_config=config).generate(project_dir=project_dir),
        config["reference_backend"],
    )


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


def build_movie_references_handler(*, store: ProjectStore, project_id: str, movie_config: dict[str, Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        config = movie_runtime_config(movie_config)
        log(f"[MoviePipeline] Stage: Movie references via {config['reference_backend']}")
        ensure_movie_bible(project_dir)
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
    prompt = f"Cinematic environment reference sheet for {display_name}"
    if cues:
        prompt += f". Environment and mood from scenes: {cues}"
    prompt = f"{prompt}. Wide establishing view, production design, lighting, atmosphere, no people, no text."
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
    from feverslop.application.movie_references import MovieReferenceSheetGenerator
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
    return MovieReferenceSheetGenerator(
        backend=hero,
        edit_backend=edit,
        hero_anchors=WorkflowAnchorConfig(positive_prompt_title="#PROMPT_POSITIVE"),
        edit_anchors=WorkflowAnchorConfig(positive_prompt_title="#PROMPT_POSITIVE", reference_image_title="#IMAGE_1"),
    )


def build_movie_visual_adapter(project_dir: Path, workflow_path: Path, movie_config: dict[str, Any] | None = None, workflow: dict | None = None):
    backend = movie_runtime_config(movie_config)["render_backend"]
    if backend == "local":
        from feverslop.adapters.movie_visual import LocalMovieVisualAdapter

        return LocalMovieVisualAdapter()

    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
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
        model_resolver=ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides),
    )


def movie_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return movie_runtime_config(dict(metadata.get("movie") or {}))


def movie_runtime_config(config: dict[str, Any] | None = None) -> dict[str, str]:
    raw = dict(config or {})
    planner_backend = _movie_backend(raw.get("planner_backend"), default="llm", supported={"llm", "deterministic", "local"})
    if planner_backend == "local":
        planner_backend = "deterministic"
    return {
        "planner_backend": planner_backend,
        "reference_backend": _movie_backend(raw.get("reference_backend"), default="comfyui", supported={"comfyui", "local"}),
        "render_backend": _movie_backend(raw.get("render_backend"), default="comfyui", supported={"comfyui", "local"}),
        "hero_workflow": _movie_workflow_path(raw.get("hero_workflow"), "workflows/image_t2i_startframe_krea_v1.json"),
        "edit_workflow": _movie_workflow_path(raw.get("edit_workflow"), "workflows/image_edit_flux2_klein_1ref_v1.json"),
        "msr_workflow": _movie_workflow_path(raw.get("msr_workflow"), "workflows/video_default_ltxv_msr_1actor_1background_v1.json"),
    }


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


def backend_config_path(value: str) -> str:
    return Path(value).as_posix()


def patch_movie_msr_workflow(*, template_path: Path = Path("workflows") / "video_default_ltxv_msr_1actor_1background_v1.json") -> dict[str, Any]:
    from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

    if not template_path.exists():
        raise FileNotFoundError(f"Movie MSR workflow template not found: {template_path}")
    workflow = json.loads(template_path.read_text(encoding="utf-8"))
    return MovieWorkflowPatcher().strip_audio_inputs(workflow)


def pipeline_mode_from_config(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = str(config.get("video_pipeline") or "")
    if value == "ltx_msr":
        return "msr"
    if value == "ltx_i2v":
        return "classic"
    return None


def thumbnail_path(store: ProjectStore, project_id: str, path: str, at: float) -> Path:
    video_path = store.resolve_video_path(project_id, path)
    seconds = max(0.0, float(at))
    key = hashlib.sha1(f"{path}:{seconds:.2f}".encode()).hexdigest()
    thumbnail = store.thumbnail_cache_path(project_id, key)
    if thumbnail.exists():
        return thumbnail
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seconds:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(thumbnail),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Could not generate thumbnail with ffmpeg") from exc
    return thumbnail
