from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from feverslop.application.global_cast_resolver import materialize_global_assets
from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.application.story_plan_service import (
    SegmentDescriptor,
    StoryPlanError,
    StoryPlanRequest,
    compute_source_fingerprint,
)
from feverslop.domain.prompt_constraints import build_location_constraint
from feverslop.domain.story_plan import (
    PLANNER_REVISION,
    SUPPORTED_SCHEMA_VERSIONS,
)
from feverslop.domain.story_plan_artifacts import (
    ArtifactClass,
    RegenerationPolicy,
    StoryPlanArtifactManifest,
    manifest_is_stale,
    manifest_matches_story_plan,
    read_manifest,
    read_story_plan,
    story_plan_fingerprint,
    write_manifest,
    write_story_plan,
)
from feverslop.errors import FeverSlopValidationError
from feverslop.ports.generate_pipeline import (
    ConceptBatcherFactory,
    LLMFactory,
    PromptPipelineFactory,
    ScenePromptBuilderFactory,
)
from feverslop.prompting.subject_directive_planning import (
    DspySubjectDirectivePlanner,
    build_shared_staging_plan,
)
from feverslop.prompting.semantic_intent_extraction import (
    SemanticIntentExtractor,
)
from feverslop.prompting.planning_payload import compact_planning_payload
from feverslop.prompting.concept_prompt_batcher import (
    validate_and_annotate_concept_chronology,
)
from feverslop.prompting.semantic_intent_review import review_and_repair
from feverslop.utils.sub_step_progress import SubStepProgress


def join_notes(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def get_steering_value(config: Any, name: str, default: str = "") -> str:
    steering = getattr(config, "steering", None)
    return str(getattr(steering, name, default) or "")


def get_config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _report_subject_staging_retry(
    reporter: Any,
    *,
    attempt: int,
    max_attempts: int,
    segment_id: str,
    error: Exception,
) -> None:
    """Render a visible, markup-safe retry diagnostic for the console reporter."""
    title = f"Subject staging · Retry {attempt}/{max_attempts} · {segment_id}"
    text = (
        "The previous staging plan was rejected; retrying with repair feedback.\n"
        f"Reason: {type(error).__name__}: {error}"
    )
    warning = getattr(reporter, "warning", None)
    if callable(warning):
        warning(text, title=title)
    else:
        panel = getattr(reporter, "panel", None)
        if callable(panel):
            panel(text, title=title)
        else:
            reporter.message(f"{title}\n{text}")


def _write_subject_directive_debug_artifact(
    *, planner: Any, scene: dict[str, Any], error: Exception, output_path: Path, model: str,
) -> Path:
    debug_dir = output_path.parent / "subject_directive_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    segment_id = str(scene.get("segment_id") or scene.get("shot_id") or scene.get("scene") or "unknown")
    debug_path = debug_dir / f"{segment_id}.json"
    payload = {
        "segment_id": segment_id,
        "model": model,
        "effective_max_tokens": 4096,
        "error": {"type": type(error).__name__, "message": str(error)},
        "planner_input": getattr(planner, "last_scene", None) or scene,
        "raw_dspy_output": getattr(planner, "last_output", None),
        "lm_history": getattr(planner, "last_lm_history", None),
    }
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return debug_path


def config_items_as_dicts(items: Any) -> list[dict]:
    output = []
    for item in items or []:
        if is_dataclass(item):
            output.append(asdict(item))
        elif isinstance(item, dict):
            output.append(dict(item))
    return output


def _actor_needs_llm_enrichment(actor: dict[str, Any]) -> bool:
    return any(not str(actor.get(field) or "").strip() for field in (
        "role", "gender", "visual_description", "image_prompt",
    ))


def _merge_configured_actors(
    configured: list[dict], generated: Any, *, mode: str = "extend", target_size: int | None = None,
) -> list[dict]:
    generated_items = config_items_as_dicts(generated)
    by_key = {
        key: item
        for item in generated_items
        for key in (
            str(item.get("id") or "").strip().casefold(),
            str(item.get("name") or "").strip().casefold(),
        )
        if key
    }
    merged = []
    for configured_actor in configured:
        actor = dict(configured_actor)
        key = str(actor.get("id") or actor.get("name") or "").strip().casefold()
        generated_actor = by_key.get(key, {})
        for field in ("role", "gender", "visual_description", "image_prompt"):
            if not str(actor.get(field) or "").strip():
                value = str(generated_actor.get(field) or "").strip()
                if value:
                    actor[field] = value
        merged.append(actor)
    if mode == "extend":
        configured_keys = {
            str(actor.get("id") or actor.get("name") or "").strip().casefold()
            for actor in configured
        }
        for actor in generated_items:
            key = str(actor.get("id") or actor.get("name") or "").strip().casefold()
            if key and key not in configured_keys:
                merged.append(dict(actor))
                configured_keys.add(key)
    if target_size is not None and len(merged) != target_size:
        raise FeverSlopValidationError(
            f"Resolved cast has {len(merged)} actors, expected {target_size} from cast policy",
        )
    return merged


_GENDER_WORDS = ("female", "male")
_GENDER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_GENDER_WORDS) + r")\b",
    re.IGNORECASE,
)
_PERSON_WORD_RE = re.compile(r"\b(?:person|individual)\b", re.IGNORECASE)


def _find_explicit_gender(text: str) -> str | None:
    """Return the first explicit gender word in text, or None when absent."""
    match = _GENDER_WORD_RE.search(text)
    return match.group(0).lower() if match else None


def _insert_gender_word(text: str, gender: str) -> str:
    """Insert a gender word before the first person word, or prepend it."""
    match = _PERSON_WORD_RE.search(text)
    if match is None:
        return f"{gender} {text}"
    word = match.group(0)
    replacement = f"{gender} {word.lower()}"
    if word[:1].isupper():
        replacement = replacement.capitalize()
    return text[: match.start()] + replacement + text[match.end():]


def enforce_explicit_cast_attributes(
    actors: list[dict],
    configured_actors: list[dict],
) -> None:
    """Keep explicit cast attributes in LLM-origin actor free text.

    Repairs a dropped gender word in generated visual_description/image_prompt
    text and raises FeverSlopValidationError when the generated text states a
    gender that contradicts the actor's gender. Configured non-empty free text
    is authoritative user data and is never touched.
    """
    configured_by_key = {}
    for configured_actor in configured_actors:
        for key in (
            str(configured_actor.get("id") or "").strip().casefold(),
            str(configured_actor.get("name") or "").strip().casefold(),
        ):
            if key:
                configured_by_key[key] = configured_actor
    for actor in actors:
        gender = str(actor.get("gender") or "").strip().lower()
        if not gender or gender in ("none", "not_applicable"):
            continue
        key = str(actor.get("id") or actor.get("name") or "").strip().casefold()
        configured = configured_by_key.get(key)
        for field in ("visual_description", "image_prompt"):
            if configured is not None and str(configured.get(field) or "").strip():
                continue
            text = str(actor.get(field) or "").strip()
            if not text:
                continue
            explicit_gender = _find_explicit_gender(text)
            if explicit_gender and explicit_gender != gender:
                label = str(actor.get("name") or actor.get("id") or "actor")
                raise FeverSlopValidationError(
                    f"Cast constraint conflict for '{label}': explicit gender "
                    f"'{gender}' contradicts '{explicit_gender}' in {field}. "
                    "Fix the configured cast or the story idea."
                )
            if explicit_gender is None:
                actor[field] = _insert_gender_word(text, gender)


def normalize_location_names(items: Any) -> list[str]:
    names = []
    for item in items or []:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item.get("id") or "").strip())
        else:
            names.append(str(item).strip())
    return [name for name in names if name]


def reporter_message(reporter: Any, message: str) -> None:
    if reporter is not None:
        reporter.message(message)


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs):
    signature = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported)


def _contract_entry_id(raw: Any) -> str:
    """Return the id of a narrative-contract entry (``{id, source}`` or string)."""
    value = raw.get("id") if isinstance(raw, dict) else raw
    return str(value or "").strip()


def resolve_text_override(
    *,
    configured_value: Any,
    generated_value_factory: Callable[[], str],
    reporter: Any,
    message: str,
) -> str:
    configured = str(configured_value or "").strip()
    if configured:
        reporter_message(reporter, message)
        return configured
    return generated_value_factory()


def resolve_locations_override(*, configured_locations: Any, generated_locations: Any, reporter: Any) -> list[str]:
    if configured_locations:
        reporter_message(reporter, "[yellow]Using locations override from project config.[/yellow]")
        return configured_locations
    return normalize_location_names(generated_locations)


def validate_and_order_concept_prompts(stage1_segments: list[dict], concept_prompts: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    expected_ids = {seg["segment_id"] for seg in stage1_segments}
    missing = [seg["segment_id"] for seg in stage1_segments if seg["segment_id"] not in concept_prompts]
    if missing:
        raise ValueError(f"Missing concept prompts: {missing}")
    extra = [segment_id for segment_id in concept_prompts if segment_id not in expected_ids]
    ordered = {seg["segment_id"]: concept_prompts[seg["segment_id"]] for seg in stage1_segments}
    return ordered, extra


class PromptGenerationPipeline:
    """Application service for resolved context, concept prompts, and scene prompts."""

    required_keys = {"config", "app_config", "request", "stage1_segments"}
    produced_keys = {
        "global_context",
        "concept_prompts",
        "scene_details",
        "scene_prompts_json",
    }

    def __init__(
        self,
        *,
        llm_factory: LLMFactory,
        prompt_pipeline_factory: PromptPipelineFactory,
        concept_batcher_factory: ConceptBatcherFactory,
        scene_prompt_builder_factory: ScenePromptBuilderFactory,
        global_library_factory: Callable[[Any], Any] | None = None,
        intent_review_factory: Callable[[Any], Any] | None = None,
        intent_extractor_factory: Callable[[Any], Any] | None = None,
        story_plan_service_factory: Callable[[Any], Any] | None = None,
    ):
        self.llm_factory = llm_factory
        self.prompt_pipeline_factory = prompt_pipeline_factory
        self.concept_batcher_factory = concept_batcher_factory
        self.scene_prompt_builder_factory = scene_prompt_builder_factory
        self.global_library_factory = global_library_factory
        self.intent_review_factory = intent_review_factory
        self.intent_extractor_factory = intent_extractor_factory
        self.story_plan_service_factory = story_plan_service_factory

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        config = context["config"]
        app_config = context["app_config"]
        request = context["request"]
        resume = bool(getattr(request, "resume", False))
        stage1_segments = context["stage1_segments"]
        resolved_context_json: Path = context["resolved_context_json"]
        semantic_intent_json: Path | None = getattr(context, "semantic_intent_json", None)
        concept_prompts_json: Path = context["concept_prompts_json"]
        scene_details_json: Path = context["scene_details_json"]
        scene_prompts_json: Path = context["scene_prompts_json"]
        log_step = context["log_step"]
        log_file = context["log_file"]
        run_spinner = context["run_spinner"]
        reporter = context["reporter"]
        artifact_store = context["artifact_store"]

        log_step("7. LLM Prompt Pipeline")
        llm = self.llm_factory(app_config)
        prompt_pipeline = self.prompt_pipeline_factory(llm)
        if resume and resolved_context_json.is_file():
            reporter.message("[yellow]Resuming resolved context; using existing resolved context.[/yellow]")
            global_context = artifact_store.read_json(resolved_context_json)
            log_file("Resolved Context JSON", resolved_context_json)
            self._report_global_context(reporter, global_context)
        else:
            global_context = self._prepare_global_context(
                config=config,
                app_config=app_config,
                prompt_pipeline=prompt_pipeline,
                stage1_segments=stage1_segments,
                run_spinner=run_spinner,
                reporter=reporter,
                resolved_context_json=resolved_context_json,
                artifact_store=artifact_store,
                log_file=log_file,
                llm=llm,
                semantic_intent_json=semantic_intent_json,
            )
            self._review_semantic_intent(
                llm=llm,
                story_idea=global_context["story_idea"],
                semantic_intent_json=getattr(context, "semantic_intent_json", None),
                semantic_intent_review_json=getattr(
                    context, "semantic_intent_review_json", None
                ),
                artifact_store=artifact_store,
                log_file=log_file,
                reporter=reporter,
            )

        _paths = getattr(context, "paths", None)
        if _paths is None and isinstance(context, dict):
            _paths = context.get("paths")
        segment_briefs, gate_stopped = self._resolve_story_plan(
            config=config,
            app_config=app_config,
            request=request,
            global_context=global_context,
            resume=resume,
            stage1_segments=stage1_segments,
            paths=_paths,
            reporter=reporter,
            artifact_store=artifact_store,
            log_file=log_file,
        )
        if gate_stopped:
            context.update({"story_plan_gate": True})
            return context

        concept_story_input = join_notes(
            global_context["story_idea"],
            "STEERING:",
            get_steering_value(config, "concepts"),
        )
        if resume and concept_prompts_json.is_file():
            reporter.message("[yellow]Resuming concept prompts; using existing concept prompts.[/yellow]")
            concept_prompts = artifact_store.read_json(concept_prompts_json)
        else:
            concept_prompts = self._generate_concept_prompts(
                config=config,
                llm=llm,
                app_config=app_config,
                prompt_pipeline=prompt_pipeline,
                request=request,
                stage1_segments=stage1_segments,
                concept_story_input=concept_story_input,
                global_context=global_context,
                segment_briefs=segment_briefs,
                concept_prompts_json=concept_prompts_json,
                artifact_store=artifact_store,
                reporter=reporter,
            )
        concept_prompts = self._finalize_concept_prompts(
            prompt_pipeline=prompt_pipeline,
            reporter=reporter,
            stage1_segments=stage1_segments,
            concept_prompts=concept_prompts,
            global_context=global_context,
            concept_prompts_json=concept_prompts_json,
            artifact_store=artifact_store,
            log_file=log_file,
        )
        if resume and scene_details_json.is_file():
            reporter.message("[yellow]Resuming scene details; using existing scene details.[/yellow]")
            scene_details = artifact_store.read_json(scene_details_json)
        else:
            skip_scene_details = get_config_value(config, "video_pipeline") == "minimax-h3-r2v"
            if skip_scene_details:
                reporter.message(
                    "[yellow]Scene details skipped for MiniMax H3 R2V; "
                    "H3 structured prompts are generated after reference sheets.[/yellow]",
                )
            else:
                reporter.message(
                    f"[cyan]Scene details started: {len(stage1_segments)} scenes; "
                    "camera and character motion per scene[/cyan]",
                )
            scene_details = self._generate_scene_details(
                config=config,
                prompt_pipeline=prompt_pipeline,
                concept_prompts=concept_prompts,
                stage1_segments=stage1_segments,
                global_context=global_context,
                reporter=reporter,
            )
            prompt_pipeline.save_json(
                scene_details_json,
                scene_details,
                artifact_store=artifact_store,
            )
        log_file("Scene Details JSON", scene_details_json)

        log_step("8. Scene Prompt Pack (Startframe + Base Motion Prompts)")
        if resume and scene_prompts_json.is_file():
            reporter.message("[yellow]Resuming scene prompt pack; using existing scene prompts.[/yellow]")
        else:
            self._build_scene_prompt_pack(
                config=config,
                llm=llm,
                stage1_segments=stage1_segments,
                concept_prompts=concept_prompts,
                scene_details=scene_details,
                global_context=global_context,
                scene_prompts_json=scene_prompts_json,
                artifact_store=artifact_store,
                reporter=reporter,
            )
            self._attach_subject_directives(
                stage1_segments=stage1_segments,
                scene_prompts_json=scene_prompts_json,
                artifact_store=artifact_store,
                reporter=reporter,
            )
            self._generate_subject_directives(
                llm=llm,
                stage1_segments=stage1_segments,
                concept_prompts=concept_prompts,
                scene_details=scene_details,
                global_context=global_context,
                scene_prompts_json=scene_prompts_json,
                artifact_store=artifact_store,
                reporter=reporter,
            )
        log_file("Scene Prompts JSON", scene_prompts_json)

        context.update(
            {
                "global_context": global_context,
                "concept_prompts": concept_prompts,
                "scene_details": scene_details,
            },
        )
        return context

    def _resolve_story_plan(
        self,
        *,
        config: Any,
        app_config: Any,
        request: Any,
        global_context: dict[str, Any] | None = None,
        resume: bool,
        stage1_segments: list[dict],
        paths: Any,
        reporter: Any,
        artifact_store: Any,
        log_file: Any,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Build or reuse the story plan; return (segment_briefs, gate_stopped)."""
        if str(getattr(config, "content_mode", "music_video")) != "music_video":
            return None, False
        if self.story_plan_service_factory is None:
            return None, False
        if paths is None:
            return None, False

        llm = self.llm_factory(app_config)
        factory = self.story_plan_service_factory
        story_planning = getattr(getattr(app_config, "llm", None), "story_planning", None)
        acting_batch_size = int(getattr(story_planning, "acting_batch_size", 8))
        # Preserve compatibility with injected one-argument factories, while
        # giving the production service its configured bound and live reporter.
        parameters = inspect.signature(factory).parameters
        factory_kwargs: dict[str, Any] = {}
        if "acting_batch_size" in parameters:
            factory_kwargs["acting_batch_size"] = acting_batch_size
        if "reporter" in parameters:
            factory_kwargs["reporter"] = reporter
        service = factory(llm, **factory_kwargs)

        global_context = global_context or {}
        effective_direction = (
            str(getattr(request, "story_direction", "") or "").strip()
            or str(getattr(config, "story_direction", "") or "").strip()
        )

        song_id = str(
            getattr(config, "song_id", "")
            or getattr(config, "project_name", "")
            or "song"
        )
        song_language = str(getattr(config, "language", "") or "unknown")
        song_style = str(
            getattr(config, "music_style", "")
            or getattr(config, "style", "")
            or ""
        )
        lyrics = str(getattr(config, "lyrics", "") or "")
        story_idea = str(global_context.get("story_idea", "") or getattr(config, "story_idea", "") or "")

        context_characters = global_context.get("actors") or getattr(config, "actors", ()) or ()
        characters = [
            {
                "id": str(actor.get("id") or actor.get("name") or "")
                if isinstance(actor, dict) else str(actor.id),
                "name": str(actor.get("name") or actor.get("id") or "")
                if isinstance(actor, dict) else str(actor.name),
                "description": str(actor.get("description", ""))
                if isinstance(actor, dict) else str(getattr(actor, "description", "")),
                "is_singer": bool(actor.get("is_singer", False))
                if isinstance(actor, dict) else bool(getattr(actor, "is_singer", False)),
            }
            for actor in context_characters
        ]
        locations = tuple(
            item for item in (global_context.get("structured_locations") or ())
            if isinstance(item, dict)
        )
        props = tuple(
            item for item in (global_context.get("props") or ())
            if isinstance(item, dict)
        )

        segments = tuple(
            SegmentDescriptor.from_mapping(seg)
            for seg in stage1_segments
            if "segment_id" in seg and SegmentDescriptor.has_stage1_timing(seg)
        )
        if stage1_segments and not segments:
            sample_keys = sorted({key for segment in stage1_segments[:3] for key in segment})
            raise StoryPlanError(
                "no usable Stage 1 segments for story planning; expected "
                "segment_id plus start/end or start_seconds/end_seconds, found keys: "
                f"{', '.join(sample_keys) or '(none)'}"
            )

        max_end = max((seg.end_seconds for seg in segments), default=0.0)
        terminal_window_seconds = max_end * 0.9 if max_end > 0 else 0.0

        source_evidence = {
            "creative_direction": effective_direction,
            "story_idea": story_idea,
            "locations": [dict(item) for item in locations],
            "props": [dict(item) for item in props],
            "narrative_contract": global_context.get("narrative_contract", {}),
            "semantic_intent": global_context.get("semantic_intent", {}),
        }

        fingerprint = compute_source_fingerprint(
            song_title=song_id,
            song_language=song_language,
            song_style=song_style,
            lyrics=lyrics,
            sections=(),
            segments=[seg.to_compact_dict() for seg in segments],
            characters=characters,
            creative_direction=effective_direction,
            story_idea=story_idea,
            locations=locations,
            props=props,
        )

        plan_path = paths.prompts_dir / f"story_plan_{song_id}.json"
        manifest_path = paths.prompts_dir / f"story_plan_{song_id}.manifest.json"

        expected_revision = str(
            getattr(
                getattr(app_config.llm, "story_planning", None),
                "planner_revision",
                PLANNER_REVISION,
            )
        )

        reused = False
        if resume and manifest_path.is_file():
            manifest = read_manifest(manifest_path)
            if not manifest_is_stale(manifest, input_fingerprint=fingerprint):
                if manifest.artifact_class == ArtifactClass.authoritative:
                    plan = read_story_plan(plan_path)
                    if (
                        manifest_matches_story_plan(manifest, plan)
                        and
                        plan.schema_version in SUPPORTED_SCHEMA_VERSIONS
                        and plan.planner_revision == expected_revision
                    ):
                        reporter.message(
                            "[yellow]Reusing story plan (fingerprint match).[/yellow]"
                        )
                        reused = True

        if not reused:
            reporter.message("[cyan]Story plan build started.[/cyan]")
            plan_request = StoryPlanRequest(
                source_fingerprint=fingerprint,
                song_title=song_id,
                song_language=song_language,
                song_style=song_style,
                lyrics=lyrics,
                sections=(),
                segments=segments,
                characters=tuple(characters),
                source_evidence=source_evidence,
                guide="",
                terminal_window_seconds=terminal_window_seconds,
                story_idea=story_idea,
                locations=locations,
                props=props,
            )
            try:
                result = service.build_plan(plan_request)
            except StoryPlanError as exc:
                if exc.diagnostics:
                    reporter.table(
                        "Story plan diagnostics",
                        ["Code", "Boundary", "What needs attention"],
                        [
                            [
                                str(diagnostic.get("code", "unknown")),
                                str(diagnostic.get("subject_id", "")),
                                str(diagnostic.get("message", "")),
                            ]
                            for diagnostic in exc.diagnostics
                        ],
                    )
                story_planning_config = getattr(
                    getattr(app_config, "llm", None), "story_planning", None
                )
                failure_policy = str(
                    getattr(story_planning_config, "failure_policy", "warn")
                ).strip().lower()
                message = (
                    "Story plan unavailable; continuing without optional story bindings: "
                    f"{exc}"
                )
                if failure_policy == "block":
                    raise
                reporter.message(f"[yellow]{message}[/yellow]")
                return {}, False
            plan = result.plan

            if plan.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
                raise StoryPlanError(
                    f"unsupported story plan schema: {plan.schema_version}"
                )
            if plan.planner_revision != expected_revision:
                raise StoryPlanError(
                    f"planner revision mismatch: {plan.planner_revision}"
                )

            write_story_plan(plan_path, plan)
            manifest = StoryPlanArtifactManifest(
                artifact_class=ArtifactClass.authoritative,
                regeneration_policy=RegenerationPolicy.on_input_change,
                input_fingerprint=fingerprint,
                plan_fingerprint=story_plan_fingerprint(plan),
            )
            write_manifest(manifest_path, manifest)
            log_file("Story Plan JSON", plan_path)
            log_file("Story Plan Manifest", manifest_path)
            reporter.message("[green]Story plan build complete.[/green]")

            if (
                getattr(
                    getattr(app_config.llm, "story_planning", None),
                    "require_approval",
                    False,
                )
                and not getattr(request, "story_plan_approve", False)
            ):
                export_md, export_manifest = service.write_review_export(
                    result, plan_path.parent, name=f"story_plan_{song_id}"
                )
                log_file("Story Plan Review Export", export_md)
                reporter.message(
                    "[yellow]Story plan approval required. "
                    "Re-run with --story-plan-approve to continue.[/yellow]"
                )
                return None, True

        segment_briefs = {
            brief.target: brief.model_dump(mode="json")
            for brief in plan.segments
        }
        return segment_briefs, False

    def _report_global_context(self, reporter: Any, global_context: dict[str, Any]) -> None:
        reporter.panel(global_context["story_idea"], title="Story Idea")
        reporter.panel(global_context["style"], title="Style Block")
        reporter.table(
            "Resolved Subject / Locations",
            ["Field", "Value"],
            [
                ["Subject", global_context["subject"]],
                ["Locations", "\n".join(global_context["locations"])],
            ],
        )

    def _generate_concept_prompts(
        self,
        *,
        config: Any,
        llm: Any,
        app_config: Any,
        prompt_pipeline: Any,
        request: Any,
        stage1_segments: list[dict],
        concept_story_input: str,
        global_context: dict[str, Any],
        segment_briefs: dict[str, Any] | None = None,
        concept_prompts_json: Path,
        artifact_store: Any,
        reporter: Any,
    ) -> dict[str, Any]:
        if request.concept_batch_size > 0:
            return self._generate_concept_prompts_batched(
                config=config,
                llm=llm,
                app_config=app_config,
                request=request,
                stage1_segments=stage1_segments,
                concept_story_input=concept_story_input,
                global_context=global_context,
                segment_briefs=segment_briefs,
                concept_prompts_json=concept_prompts_json,
                artifact_store=artifact_store,
                reporter=reporter,
            )
        reporter.message(
            f"[cyan]Concept generation started for "
            f"{len(stage1_segments)} scenes[/cyan]",
        )
        concept_prompts = call_with_supported_kwargs(
            prompt_pipeline.create_concept_prompts,
            stage1_segments=stage1_segments,
            story_idea=concept_story_input,
            global_context=global_context,
            notes=get_steering_value(config, "concepts"),
            segment_briefs=segment_briefs or {},
        )
        reporter.message("[green]Concept generation finished.[/green]")
        return concept_prompts

    def _generate_concept_prompts_batched(
        self,
        *,
        config: Any,
        llm: Any,
        app_config: Any,
        request: Any,
        stage1_segments: list[dict],
        concept_story_input: str,
        global_context: dict[str, Any],
        segment_briefs: dict[str, Any] | None = None,
        concept_prompts_json: Path,
        artifact_store: Any,
        reporter: Any,
    ) -> dict[str, Any]:
        reporter.message(
            f"[cyan]Using batched concept generation: "
            f"{request.concept_batch_size} segments per batch[/cyan]",
        )
        enforcement = (
            config.narrative_contract_enforcement
            if config.narrative_contract_enforcement is not None
            else app_config.llm.narrative_contract_enforcement
        )
        concept_batcher = self.concept_batcher_factory(
            llm,
            request.concept_batch_size,
            request_timeout_seconds=app_config.llm.request_timeout_seconds,
            semantic_enforcement=enforcement,
        )
        # Optional seam: capable batchers checkpoint accepted concepts per
        # batch so a mid-stage failure does not discard the whole stage.
        enable_checkpoint = getattr(concept_batcher, "enable_checkpoint", None)
        if callable(enable_checkpoint):
            enable_checkpoint(
                path=concept_prompts_json.with_name(
                    concept_prompts_json.stem.replace("concept_prompts", "concept_checkpoint", 1)
                    + concept_prompts_json.suffix
                ),
                artifact_store=artifact_store,
            )
        reporter.message(
            f"[cyan]Concept generation started: "
            f"{len(stage1_segments)} scenes, batches of "
            f"{request.concept_batch_size}[/cyan]",
        )
        concept_prompts = call_with_supported_kwargs(
            concept_batcher.create_concept_prompts_batched,
            stage1_segments=stage1_segments,
            story_idea=concept_story_input,
            global_context=global_context,
            notes=get_steering_value(config, "concepts"),
            progress_callback=lambda message: reporter.message(
                f"[cyan]{message}[/cyan]",
            ),
            segment_briefs=segment_briefs or {},
        )
        reporter.message("[green]Concept generation finished.[/green]")
        return concept_prompts

    def _prepare_global_context(
        self,
        *,
        config: Any,
        app_config: Any,
        prompt_pipeline: Any,
        stage1_segments: list[dict],
        run_spinner: Callable[[str, Callable[[], Any]], Any],
        reporter: Any,
        resolved_context_json: Path,
        artifact_store: Any,
        log_file: Callable[[str, Path], None],
        llm: Any = None,
        semantic_intent_json: Path | None = None,
    ) -> dict:
        all_lyrics = " ".join(
            seg.get("lyrics", "")
            for seg in stage1_segments
            if seg.get("lyrics")
        ).strip()
        global_context = self.build_resolved_global_context(
            config=config,
            app_config=app_config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics=all_lyrics,
            run_spinner=run_spinner,
            reporter=reporter,
            llm=llm,
            semantic_intent_json=semantic_intent_json,
            artifact_store=artifact_store,
            log_file=log_file,
        )
        prompt_pipeline.save_json(
            resolved_context_json,
            global_context,
            artifact_store=artifact_store,
        )
        log_file("Resolved Context JSON", resolved_context_json)
        self._report_global_context(reporter, global_context)
        return global_context

    def _review_semantic_intent(
        self,
        *,
        llm: Any,
        story_idea: str,
        semantic_intent_json: Path | None,
        semantic_intent_review_json: Path | None,
        artifact_store: Any,
        log_file: Callable[[str, Path], None],
        reporter: Any,
    ) -> None:
        """Independently review and repair the extracted semantic intent ledger.

        This is the review/repair stage (#545). It reads the ledger persisted
        by the extraction stage (#544), runs an independent judge against the
        original story idea, applies bounded targeted repair, and persists a
        reviewed artifact. It is tolerant and bounded: with no extraction
        artifact (tests, non-LLM runs, or before #544 lands) it is a no-op and
        never invents entities. A review/repair failure is logged as a warning
        and never blocks the pipeline.
        """
        if semantic_intent_json is None or not semantic_intent_json.is_file():
            return
        try:
            from feverslop.prompting.semantic_intent_review import (
                DspySemanticIntentReviewer,
                IntentLedger,
            )

            payload = artifact_store.read_json(semantic_intent_json)
            ledger = IntentLedger.from_dict(payload.get("ledger", {}))
            reviewer = (
                self.intent_review_factory(llm)
                if self.intent_review_factory is not None
                else DspySemanticIntentReviewer(llm)
            )
            review = reviewer.review(story_idea, ledger)
            result = review_and_repair(
                story_idea=story_idea,
                ledger=ledger,
                reviewer=reviewer,
                review=review,
            )
        except Exception as exc:  # noqa: BLE001 - bounded, never blocks
            if reporter is not None:
                reporter.message(
                    f"[yellow]Semantic intent review skipped: {exc}[/yellow]"
                )
            return
        out_payload = {
            "status": result.status,
            "findings": [f.model_dump() for f in review.findings],
            "applied": result.applied,
            "skipped": result.skipped,
            "warnings": result.warnings,
            "ledger": result.ledger.to_dict(),
        }
        if semantic_intent_review_json is not None and artifact_store is not None:
            artifact_store.write_json(semantic_intent_review_json, out_payload)
            if log_file is not None:
                log_file("Semantic Intent Review JSON", semantic_intent_review_json)
        if reporter is not None:
            if review.findings or result.warnings:
                reporter.message(
                    f"[yellow]Semantic intent review result: {result.status} "
                    f"({len(review.findings)} findings, "
                    f"{len(result.applied)} applied, {len(result.skipped)} unresolved, "
                    f"{len(result.warnings)} warnings)[/yellow]"
                )
                self._report_warning_details(
                    reporter,
                    result.warnings,
                    title="Semantic intent review",
                )
            else:
                reporter.message("[green]Semantic intent review: clean[/green]")

    def _finalize_concept_prompts(
        self,
        *,
        prompt_pipeline: Any,
        reporter: Any,
        stage1_segments: list[dict],
        concept_prompts: dict[str, Any],
        global_context: dict[str, Any],
        concept_prompts_json: Path,
        artifact_store: Any,
        log_file: Callable[[str, Path], None],
    ) -> dict[str, Any]:
        concept_prompts, extra_concepts = validate_and_order_concept_prompts(stage1_segments, concept_prompts)
        if extra_concepts:
            reporter.message(f"[yellow]Ignoring extra concept prompt keys: {extra_concepts}[/yellow]")
        concept_prompts = validate_and_annotate_concept_chronology(
            concept_prompts,
            global_context.get("narrative_contract") or {},
        )
        prompt_pipeline.save_json(
            concept_prompts_json,
            concept_prompts,
            artifact_store=artifact_store,
        )
        log_file("Concept Prompts JSON", concept_prompts_json)
        return concept_prompts

    def _generate_scene_details(
        self,
        *,
        config: Any,
        prompt_pipeline: Any,
        concept_prompts: dict[str, Any],
        stage1_segments: list[dict],
        global_context: dict[str, Any],
        reporter: Any,
    ) -> Any:
        skip_llm = get_config_value(config, "video_pipeline") == "minimax-h3-r2v"
        scene_details_progress = (
            None if skip_llm else SubStepProgress(reporter, "Scene details", len(stage1_segments))
        )
        scene_details = call_with_supported_kwargs(
            prompt_pipeline.create_scene_details,
            concept_prompts=concept_prompts,
            stage1_segments=stage1_segments,
            global_context=global_context,
            progress_callback=(
                None
                if scene_details_progress is None
                else lambda current, total: scene_details_progress.update(current)
            ),
            skip_llm=skip_llm,
        )
        if not skip_llm:
            reporter.message("[green]Scene details finished.[/green]")
        return scene_details

    def _build_scene_prompt_pack(
        self,
        *,
        config: Any,
        llm: Any,
        stage1_segments: list[dict],
        concept_prompts: dict[str, Any],
        scene_details: Any,
        global_context: dict[str, Any],
        scene_prompts_json: Path,
        artifact_store: Any,
        reporter: Any,
    ) -> None:
        reporter.message(
            f"[cyan]Scene prompt pack started: {len(stage1_segments)} scenes; "
            "building still-image startframe and backend-neutral base motion prompts[/cyan]",
        )
        if get_config_value(config, "video_pipeline") == "minimax-h3-r2v":
            reporter.message(
                "[cyan]MiniMax H3 R2V selected: H3 structured prompts will be "
                "generated after reference sheets.[/cyan]",
            )
        scene_prompt_builder = self.scene_prompt_builder_factory(llm)
        scene_prompts_progress = SubStepProgress(reporter, "Scene prompts", len(stage1_segments))
        scene_prompt_builder.build_scene_prompts(
            stage1_segments=stage1_segments,
            concept_prompts=concept_prompts,
            scene_details=scene_details,
            global_context=global_context,
            output_json_path=scene_prompts_json,
            zimage_instructions=get_steering_value(config, "zimage"),
            ltx_instructions=get_steering_value(config, "ltx"),
            trigger_word=str(get_config_value(config, "trigger_word", "") or ""),
            artifact_store=artifact_store,
            progress_callback=lambda current, total: scene_prompts_progress.update(current),
            status_callback=reporter.message,
        )
        reporter.message("[green]Scene prompt pack finished.[/green]")

    @staticmethod
    def _attach_subject_directives(
        *,
        stage1_segments: list[dict],
        scene_prompts_json: Path,
        artifact_store: Any,
        reporter: Any,
    ) -> None:
        """Persist shared staging when upstream scene input provides subjects."""
        try:
            prompts = artifact_store.read_json(scene_prompts_json)
        except (FileNotFoundError, KeyError):
            # Lightweight dependency fakes may only capture the builder call;
            # there is no persisted scene pack to enrich in that mode.
            return
        by_segment = {str(item.get("segment_id")): item for item in stage1_segments}
        changed = 0
        for scene in prompts:
            source = by_segment.get(str(scene.get("segment_id")))
            if not source or not (source.get("subject_directives") or source.get("subjects")):
                continue
            plan = build_shared_staging_plan({
                "shot_id": source.get("segment_id") or scene.get("scene"),
                "duration_seconds": source.get("duration") or source.get("duration_seconds") or scene.get("duration"),
                "subjects": source.get("subjects") or [],
                "subject_directives": source.get("subject_directives"),
                "spatial_relations": source.get("spatial_relations") or [],
            }) if not source.get("subject_directives") else None
            if plan is not None:
                scene["subject_directives"] = plan.to_dict()
                changed += 1
            elif source.get("subject_directives"):
                scene["subject_directives"] = source["subject_directives"]
                changed += 1
        if changed:
            artifact_store.write_json(scene_prompts_json, prompts)
            reporter.message(f"[green]Subject staging persisted for {changed} scenes.[/green]")

    @staticmethod
    def _generate_subject_directives(
        *,
        llm: Any,
        stage1_segments: list[dict],
        concept_prompts: dict[str, Any],
        scene_details: dict[str, Any],
        global_context: dict[str, Any],
        scene_prompts_json: Path,
        artifact_store: Any,
        reporter: Any,
    ) -> None:
        """Generate one shared DSPy staging plan per scene in production runs."""
        if not getattr(llm, "model", None) or getattr(llm, "client", None) is None:
            return
        try:
            prompts = artifact_store.read_json(scene_prompts_json)
        except (FileNotFoundError, KeyError):
            return
        planner = DspySubjectDirectivePlanner(llm)
        by_segment = {str(item.get("segment_id")): item for item in stage1_segments}
        changed = 0
        failures: list[str] = []
        for scene in prompts:
            if scene.get("subject_directives") is not None:
                continue
            source = by_segment.get(str(scene.get("segment_id")))
            if source is None:
                continue
            segment_id = str(source.get("segment_id") or scene.get("scene"))
            concept = concept_prompts.get(segment_id, "")
            if isinstance(concept, dict):
                concept = concept.get("concept") or concept.get("prompt") or ""
            planner_input = {
                "shot_id": segment_id,
                "scene": source.get("scene") or scene.get("scene"),
                "duration_seconds": source.get("duration") or source.get("duration_seconds") or scene.get("duration_seconds"),
                # Subject staging only needs semantic scene facts. Keep the
                # authoritative timing in artifacts, but do not resend the
                # expanded whisper/alignment evidence to the LLM.
                "segment": compact_planning_payload(source),
                "concept": str(concept),
                "scene_details": scene_details.get(segment_id, {}),
                "global_context": compact_planning_payload(global_context),
                "allowed_subject_ids": list(
                    (scene.get("references") or {}).get("actor_ids") or [],
                ),
                "allowed_environment_ids": [
                    str(location_id)
                    for location_id in [
                        (scene.get("references") or {}).get("location_id"),
                    ]
                    if location_id
                ],
                "allowed_prop_ids": [
                    str(item.get("id"))
                    for item in (global_context.get("props") or [])
                    if isinstance(item, dict) and item.get("id")
                ],
            }
            plan = None
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    plan = planner.plan(planner_input)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 3:
                        break
                    planner_input = {
                        **planner_input,
                        "repair_feedback": (
                            f"Previous staging plan was rejected: {type(exc).__name__}: {exc}. "
                            "Return a complete valid plan. Every temporal scope must have "
                            "end_seconds greater than start_seconds and cover the shot duration. "
                            "Return only the structured staging plan."
                        ),
                    }
                    _report_subject_staging_retry(
                        reporter,
                        attempt=attempt + 1,
                        max_attempts=3,
                        segment_id=segment_id,
                        error=exc,
                    )
            if plan is None:
                assert last_error is not None
                debug_path = _write_subject_directive_debug_artifact(
                    planner=planner,
                    scene=planner_input,
                    error=last_error,
                    output_path=scene_prompts_json,
                    model=str(getattr(llm, "model", "unknown")),
                )
                failures.append(f"{segment_id}: {last_error} (trace: {debug_path})")
                continue
            for repair in getattr(planner, "last_repairs", []):
                reporter.message(
                    f"[yellow]Subject staging repaired for {segment_id}: {repair}[/yellow]",
                )
            scene["subject_directives"] = plan.to_dict()
            changed += 1
            reporter.message(f"[cyan]Subject staging generated: {changed}/{len(prompts)} scenes[/cyan]")
        if changed:
            artifact_store.write_json(scene_prompts_json, prompts)
        if failures:
            reporter.message(
                "[yellow]Subject staging warning: "
                f"{len(failures)} scene(s) could not be normalized; "
                "their debug traces were saved and the pipeline continues. "
                + " | ".join(failures)
                + "[/yellow]",
            )

    @staticmethod
    def _collect_config_values(config: Any) -> dict:
        config_story_idea = str(get_config_value(config, "story_idea", "") or "").strip()
        config_style = str(get_config_value(config, "style", "") or "").strip()
        config_subject = str(get_config_value(config, "subject", "") or "").strip()
        config_locations = get_config_value(config, "locations", []) or []
        config_actors = get_config_value(config, "actors", []) or []
        cast_idea = str(get_config_value(config, "cast_idea", "") or "").strip()
        cast_policy = get_config_value(config, "cast_policy", None)
        cast_mode = str(getattr(cast_policy, "mode", "extend") or "extend").strip().lower()
        cast_target_size = getattr(cast_policy, "target_size", None)
        config_structured_locations = get_config_value(config, "structured_locations", []) or []
        subject_mode = str(get_config_value(config, "subject_mode", "multi") or "multi")
        max_scene_actors = int(get_config_value(config, "max_scene_actors", 1 if subject_mode == "single" else 4) or 4)
        silent_mode = bool(get_config_value(config, "silent_mode", False))
        audio_config = getattr(config, "audio", None)
        language = str(getattr(audio_config, "language", "") or "").strip()
        return {
            "story_idea": config_story_idea,
            "style": config_style,
            "subject": config_subject,
            "locations": config_locations,
            "actors": config_actors,
            "cast_idea": cast_idea,
            "cast_mode": cast_mode,
            "cast_target_size": cast_target_size,
            "structured_locations": config_structured_locations,
            "subject_mode": subject_mode,
            "max_scene_actors": max_scene_actors,
            "silent_mode": silent_mode,
            "language": language,
        }

    def _resolve_actors_and_locations(
        self,
        *,
        config: Any,
        app_config: Any,
        subject_locations: dict,
        configured_actor_items: list[dict],
    ) -> tuple:
        config_actors = get_config_value(config, "actors", []) or []
        config_structured_locations = get_config_value(config, "structured_locations", []) or []
        cast_policy = get_config_value(config, "cast_policy", None)
        cast_mode = str(getattr(cast_policy, "mode", "extend") or "extend").strip().lower()
        cast_target_size = getattr(cast_policy, "target_size", None)
        actors = config_items_as_dicts(config_actors)
        generated_actors = config_items_as_dicts(subject_locations.get("actors", []))
        actors = (
            _merge_configured_actors(actors, generated_actors, mode=cast_mode, target_size=cast_target_size)
            if actors else generated_actors
        )
        if not actors and cast_target_size is not None:
            raise FeverSlopValidationError(
                f"Resolved cast has 0 actors, expected {cast_target_size} from cast policy",
            )
        enforce_explicit_cast_attributes(actors, configured_actor_items)
        structured_locations = (
            config_items_as_dicts(config_structured_locations)
            or config_items_as_dicts(subject_locations.get("locations", []))
        )
        global_resolution = None
        if app_config is not None and any(
            get_config_value(config, field_name, ())
            for field_name in ("global_cast", "global_locations", "global_styles", "global_props")
        ):
            if self.global_library_factory is None:
                raise RuntimeError("global assets require a composition-provided library factory")
            global_resolution = materialize_global_assets(
                config,
                app_config,
                library_factory=self.global_library_factory,
            )
            actors = list(global_resolution.actors) + actors
            structured_locations = list(global_resolution.locations) + structured_locations
        actor_ids = [str(actor.get("id") or "").strip() for actor in actors]
        if "" in actor_ids or len(actor_ids) != len(set(actor_ids)):
            raise FeverSlopValidationError("Resolved cast contains missing or duplicate actor ids")
        return actors, structured_locations, global_resolution

    def _build_generation_notes(self, config: Any) -> dict:
        story_notes = join_notes(
            get_steering_value(config, "global_"),
            get_steering_value(config, "story_idea"),
        )
        style_notes = join_notes(
            get_steering_value(config, "global_"),
            get_steering_value(config, "style"),
        )
        configured_actor_items = config_items_as_dicts(get_config_value(config, "actors", []) or [])
        cast_anchor_notes = "\n".join(
            f"- id={actor.get('id')}; name={actor.get('name')}; "
            f"role={actor.get('role') or '(generate)'}; gender={actor.get('gender') or '(generate)'}"
            for actor in configured_actor_items
        )
        subject_location_notes = join_notes(
            get_steering_value(config, "global_"),
            get_steering_value(config, "subject"),
            get_steering_value(config, "locations"),
            "Configured cast anchors. Preserve these ids and any non-empty role/gender values. "
            "Generate missing role/gender and all missing visual_description/image_prompt fields "
            "from the story and character arc; do not invent a conflicting gender.\n" + cast_anchor_notes
            if cast_anchor_notes else "",
        )
        return {
            "story_notes": story_notes,
            "style_notes": style_notes,
            "subject_location_notes": subject_location_notes,
            "configured_actor_items": configured_actor_items,
        }

    def _extract_semantic_intent(
        self,
        *,
        llm: Any,
        story_idea: str,
        notes: dict,
        semantic_intent_json: Path | None,
        artifact_store: Any,
        log_file: Callable[[str, Path], None],
        reporter: Any,
    ) -> dict:
        """Run the typed intent extraction and persist the artifact.

        With no extractor factory configured (tests, or a non-LLM run) this
        is a no-op that returns an empty ledger -- it never invents entities.
        A failed extraction is bounded: it still persists a constructible,
        empty ledger with a visible status so subject/location generation
        always has a valid artifact.
        """
        factory = self.intent_extractor_factory
        if factory is None or llm is None:
            extractor = SemanticIntentExtractor()
        else:
            extractor = factory(llm)
        result = extractor.extract(story_idea=story_idea, notes=notes.get("story_notes", ""))
        payload = {
            "status": result.status,
            "warnings": result.warnings,
            "ledger": result.ledger.to_dict(),
        }
        if semantic_intent_json is not None and artifact_store is not None:
            artifact_store.write_json(semantic_intent_json, payload)
            if log_file is not None:
                log_file("Semantic Intent JSON", semantic_intent_json)
        if reporter is not None and result.warnings:
            reporter.message(
                f"[yellow]Semantic intent extraction: {result.status} "
                f"({len(result.warnings)} warnings)[/yellow]"
            )
            self._report_warning_details(
                reporter,
                result.warnings,
                title="Semantic intent extraction",
            )
        return payload

    @staticmethod
    def _report_warning_details(
        reporter: Any,
        warnings: list[str],
        *,
        title: str,
        limit: int = 20,
    ) -> None:
        report_warning = getattr(reporter, "warning", None)
        for warning in warnings[:limit]:
            if callable(report_warning):
                report_warning(warning, title=title)
            else:
                reporter.message(f"[yellow]{title}: {warning}[/yellow]")
        remaining = len(warnings) - limit
        if remaining > 0:
            summary = f"{remaining} additional warnings are stored in the JSON artifact"
            if callable(report_warning):
                report_warning(summary, title=title)
            else:
                reporter.message(f"[yellow]{title}: {summary}[/yellow]")

    def _resolve_story_idea(self, *, config_values: dict, prompt_pipeline: Any, all_lyrics: str, notes: dict, run_spinner: Callable, reporter: Any) -> str:
        return resolve_text_override(
            configured_value=config_values["story_idea"],
            reporter=reporter,
            message="[yellow]Using story_idea override from project config.[/yellow]",
            generated_value_factory=lambda: run_spinner(
                "Generating story idea...",
                lambda: prompt_pipeline.create_story_idea(
                    lyrics=all_lyrics,
                    notes=notes["story_notes"],
                ),
            ),
        )

    def _resolve_style_block(self, *, config_values: dict, prompt_pipeline: Any, all_lyrics: str, notes: dict, run_spinner: Callable, reporter: Any) -> str:
        return resolve_text_override(
            configured_value=config_values["style"],
            reporter=reporter,
            message="[yellow]Using style override from project config.[/yellow]",
            generated_value_factory=lambda: run_spinner(
                "Generating style block...",
                lambda: prompt_pipeline.create_style_block(
                    lyrics=all_lyrics,
                    notes=notes["style_notes"],
                ),
            ),
        )

    def _resolve_subject_and_locations(
        self,
        *,
        config: Any,
        prompt_pipeline: Any,
        config_values: dict,
        notes: dict,
        story_idea: str,
        run_spinner: Callable[[str, Callable[[], Any]], Any],
        reporter: Any,
    ) -> tuple:
        has_configured_subject_assets = bool(
            config_values["subject"]
            and config_items_as_dicts(config_values["actors"])
            and config_items_as_dicts(config_values["structured_locations"]),
        )
        subject_locations = (
            {}
            if has_configured_subject_assets
            and not any(_actor_needs_llm_enrichment(actor) for actor in notes["configured_actor_items"])
            else run_spinner(
                "Generating subject and locations fallback...",
                lambda: call_with_supported_kwargs(
                    prompt_pipeline.create_subject_and_locations,
                    story_idea=story_idea,
                    notes=notes["subject_location_notes"],
                    cast_idea=config_values["cast_idea"],
                ),
            )
        )
        subject = resolve_text_override(
            configured_value=config_values["subject"],
            reporter=reporter,
            message="[yellow]Using subject override from project config.[/yellow]",
            generated_value_factory=lambda: subject_locations["subject"],
        )
        locations = resolve_locations_override(
            configured_locations=config_values["locations"],
            generated_locations=subject_locations.get("locations", []),
            reporter=reporter,
        )
        return subject_locations, subject, locations

    def _derive_narrative_contract(
        self,
        *,
        config: Any,
        prompt_pipeline: Any,
        story_idea: str,
        actors: list[dict],
        structured_locations: list[dict],
        notes: dict,
        run_spinner: Callable[[str, Callable[[], Any]], Any],
        reporter: Any,
    ) -> tuple[dict, str]:
        """Resolve the narrative contract, preferring config over LLM derivation.

        A non-empty configured contract wins unchanged (source ``config``).
        Otherwise the contract is derived from the story idea and the resolved
        cast/locations (source ``llm``). Derivation is bounded: any failure or
        a contract that references unknown canonical ids falls back to an empty
        contract (source ``empty``) so the pipeline never blocks on it.
        """
        configured = get_config_value(config, "narrative_contract", {}) or {}
        if configured:
            if reporter is not None:
                reporter.message("[cyan]Narrative contract: using configured contract.[/cyan]")
            return dict(configured), "config"
        if reporter is not None:
            reporter.message("[cyan]Narrative contract: deriving from story idea...[/cyan]")
        try:
            derived = run_spinner(
                "Deriving narrative contract...",
                lambda: prompt_pipeline.create_narrative_contract(
                    story_idea=story_idea,
                    locations=structured_locations,
                    actors=actors,
                    notes=notes.get("story_notes", ""),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - bounded, never blocks
            if reporter is not None:
                reporter.message(f"[yellow]Narrative contract derivation failed: {exc}[/yellow]")
            return {}, "empty"
        derived = derived if isinstance(derived, dict) else {}
        if not derived:
            if reporter is not None:
                reporter.message("[yellow]Narrative contract: LLM returned empty; using empty contract.[/yellow]")
            return {}, "empty"
        warnings = self._validate_narrative_contract(derived, actors, structured_locations)
        if warnings:
            if reporter is not None:
                reporter.message(
                    f"[yellow]Narrative contract invalid ({'; '.join(warnings)}); "
                    f"using empty contract.[/yellow]"
                )
            return {}, "empty"
        if reporter is not None:
            reporter.message(
                "[green]Narrative contract derived "
                f"({len(derived.get('location_order') or [])} locations, "
                f"{len(derived.get('milestone_order') or [])} milestones).[/green]"
            )
        return derived, "llm"

    def _validate_narrative_contract(
        self,
        contract: dict,
        actors: list[dict],
        structured_locations: list[dict],
    ) -> list[str]:
        """Return warnings for canonical ids the contract invents.

        Only location and actor ids can be checked against the resolved cast
        and locations; milestone and chronology-exception names are narrative
        and are not validated.
        """
        if not isinstance(contract, dict) or not contract:
            return []
        location_ids = {
            str(location.get("id") or "").strip()
            for location in (structured_locations or [])
            if isinstance(location, dict) and str(location.get("id") or "").strip()
        }
        actor_ids = {
            str(actor.get("id") or "").strip()
            for actor in (actors or [])
            if isinstance(actor, dict) and str(actor.get("id") or "").strip()
        }
        warnings: list[str] = []
        for raw in contract.get("location_order") or ():
            item_id = _contract_entry_id(raw)
            if item_id and item_id not in location_ids:
                warnings.append(f"location_order references unknown location {item_id!r}")
        allowed = contract.get("actor_allowed_locations") or {}
        if isinstance(allowed, dict):
            for actor_id, allowed_locations in allowed.items():
                actor_id = str(actor_id or "").strip()
                if actor_id and actor_id not in actor_ids:
                    warnings.append(f"actor_allowed_locations references unknown actor {actor_id!r}")
                for raw in allowed_locations or ():
                    item_id = _contract_entry_id(raw)
                    if item_id and item_id not in location_ids:
                        warnings.append(
                            f"actor_allowed_locations[{actor_id!r}] references unknown location {item_id!r}"
                        )
        terminal = contract.get("terminal_states") or {}
        if isinstance(terminal, dict):
            for actor_id in terminal:
                actor_id = str(actor_id or "").strip()
                if actor_id and actor_id not in actor_ids:
                    warnings.append(f"terminal_states references unknown actor {actor_id!r}")
        return warnings

    def build_resolved_global_context(
        self,
        *,
        config: Any,
        app_config: Any = None,
        prompt_pipeline: Any,
        all_lyrics: str,
        run_spinner: Callable[[str, Callable[[], Any]], Any],
        reporter: Any = None,
        console: Any = None,
        llm: Any = None,
        semantic_intent_json: Path | None = None,
        artifact_store: Any = None,
        log_file: Callable[[str, Path], None] | None = None,
    ) -> dict:
        if reporter is None and console is not None:
            reporter = console
        notes = self._build_generation_notes(config)
        config_values = self._collect_config_values(config)
        story_idea = self._resolve_story_idea(
            config_values=config_values,
            prompt_pipeline=prompt_pipeline,
            all_lyrics=all_lyrics,
            notes=notes,
            run_spinner=run_spinner,
            reporter=reporter,
        )
        self._extract_semantic_intent(
            llm=llm,
            story_idea=story_idea,
            notes=notes,
            semantic_intent_json=semantic_intent_json,
            artifact_store=artifact_store,
            log_file=log_file,
            reporter=reporter,
        )
        style_block = self._resolve_style_block(
            config_values=config_values,
            prompt_pipeline=prompt_pipeline,
            all_lyrics=all_lyrics,
            notes=notes,
            run_spinner=run_spinner,
            reporter=reporter,
        )

        cast_idea = config_values["cast_idea"]
        cast_mode = config_values["cast_mode"]
        cast_target_size = config_values["cast_target_size"]
        subject_locations, subject, locations = self._resolve_subject_and_locations(
            config=config,
            prompt_pipeline=prompt_pipeline,
            config_values=config_values,
            notes=notes,
            story_idea=story_idea,
            run_spinner=run_spinner,
            reporter=reporter,
        )

        actors, structured_locations, global_resolution = self._resolve_actors_and_locations(
            config=config,
            app_config=app_config,
            subject_locations=subject_locations,
            configured_actor_items=notes["configured_actor_items"],
        )

        audio_refs = get_config_value(config, "minimax_h3_audio_refs", None)
        audio_subject_bindings = (
            getattr(audio_refs, "subject_bindings", {})
            if audio_refs is not None
            else {}
        )

        narrative_contract, narrative_contract_source = self._derive_narrative_contract(
            config=config,
            prompt_pipeline=prompt_pipeline,
            story_idea=story_idea,
            actors=actors,
            structured_locations=structured_locations,
            notes=notes,
            run_spinner=run_spinner,
            reporter=reporter,
        )

        return {
            "story_idea": story_idea,
            "style": style_block,
            "subject": subject,
            "locations": locations,
            "actors": actors,
            "cast_idea": cast_idea,
            "cast_policy": {"mode": cast_mode, "target_size": cast_target_size},
            "cast_contract": {
                "target_size": cast_target_size,
                "resolved_size": len(actors),
                "configured_ids": [str(actor.get("id")) for actor in notes["configured_actor_items"]],
                "source": "configured_and_cast_idea" if cast_idea else "configured_or_generated",
            },
            "structured_locations": structured_locations,
            "props": list(global_resolution.props) if global_resolution else [],
            "styles": list(global_resolution.styles) if global_resolution else [],
            "global_asset_snapshots": list(global_resolution.snapshots) if global_resolution else [],
            "subject_mode": config_values["subject_mode"],
            "max_scene_actors": config_values["max_scene_actors"],
            "narrative_contract": narrative_contract,
            "narrative_contract_source": narrative_contract_source,
            "audio_subject_bindings": audio_subject_bindings,
            "video_pipeline": str(get_config_value(config, "video_pipeline", "ltx_i2v") or "ltx_i2v").strip(),
            "language": config_values["language"],
            "silent_mode": config_values["silent_mode"],
            "location_constraint": build_location_constraint(locations),
            "steering": {
                "global": get_steering_value(config, "global_"),
                "story_idea": get_steering_value(config, "story_idea"),
                "style": get_steering_value(config, "style"),
                "subject": get_steering_value(config, "subject"),
                "locations": get_steering_value(config, "locations"),
                "concepts": get_steering_value(config, "concepts"),
                "zimage": get_steering_value(config, "zimage"),
                "ltx": get_steering_value(config, "ltx"),
                "final_prompts": get_steering_value(config, "final_prompts"),
            },
            "prompt_guidance": config.prompt_guidance.as_prompt_context(),
        }
