from __future__ import annotations

from collections.abc import Callable
from typing import Any

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.errors import FeverSlopDataError
from feverslop.ports.generate_pipeline import H3PromptBuilderFactory
from feverslop.prompting.dspy_h3_models import PromptMode
from feverslop.prompting.model_types import resolve_model_type
from feverslop.utils.sub_step_progress import SubStepProgress


def _attach_relay_segments(stage1_segments: list[dict], relay_scenes: list[dict]) -> list[dict]:
    """Join the frame relay artifact to the segment records consumed by H3."""
    relay_by_segment = {
        str(scene.get("metadata", {}).get("segment_id") or scene.get("segment_id")): scene
        for scene in relay_scenes
        if scene.get("metadata", {}).get("segment_id") or scene.get("segment_id")
    }
    relay_by_scene = {
        int(scene["scene"]): scene
        for scene in relay_scenes
        if scene.get("scene") is not None
    }
    enriched = []
    for segment in stage1_segments:
        result = dict(segment)
        relay_scene = relay_by_segment.get(str(segment.get("segment_id")))
        if relay_scene is None and segment.get("scene") is not None:
            relay_scene = relay_by_scene.get(int(segment["scene"]))
        if relay_scene:
            result.setdefault("fps", relay_scene.get("fps"))
            result.setdefault("duration_seconds", relay_scene.get("duration_seconds"))
            relay = relay_scene.get("prompt_relay") or (relay_scene.get("ltx") or {}).get("prompt_relay")
            if relay:
                ltx = dict(result.get("ltx") or {})
                ltx.setdefault("prompt_relay", relay)
                result["ltx"] = ltx
        enriched.append(result)
    return enriched


def _attach_beat_events(stage1_segments: list[dict], beat_data: dict[str, Any]) -> list[dict]:
    """Attach bounded scene-local beat events to H3 input segments."""
    beats = beat_data.get("beats") or []
    bpm = float(beat_data.get("bpm") or 0)
    enriched = []
    for segment in stage1_segments:
        start = float(segment.get("start") or segment.get("abs_start_seconds") or 0)
        end_value = segment.get("end") or segment.get("abs_end_seconds")
        if end_value is None:
            end_value = start + float(segment.get("duration") or segment.get("duration_seconds") or 0)
        end = float(end_value)
        local_beats = []
        for beat in beats:
            absolute = float(beat.get("time") or 0)
            if absolute < start or absolute >= end:
                continue
            local_beats.append({
                "time_seconds": round(absolute - start, 4),
                "downbeat": bool(beat.get("downbeat")),
                "impact": float(beat.get("impact") or 0),
            })
        result = dict(segment)
        if local_beats:
            result["performance_timing"] = {"bpm": bpm, "beats": local_beats}
        enriched.append(result)
    return enriched


def _attach_subject_directives(stage1_segments: list[dict], scene_prompts: list[dict]) -> list[dict]:
    directives_by_segment = {
        str(scene.get("segment_id")): scene.get("subject_directives")
        for scene in scene_prompts
        if scene.get("subject_directives") is not None
    }
    enriched = []
    for segment in stage1_segments:
        result = dict(segment)
        directives = directives_by_segment.get(str(segment.get("segment_id")))
        if directives is not None:
            result["subject_directives"] = directives
        enriched.append(result)
    return enriched


def _configured_audio_paths(
    config: Any,
    stem_files: dict[str, Any] | None,
    input_audio: Any | None = None,
) -> dict[str, Any] | None:
    """Return only the audio stems selected for the MiniMax reference workflow."""
    if not stem_files and input_audio is None:
        return None

    configured_stems = list(getattr(getattr(config, "minimax_h3_audio_refs", None), "stems", ()))
    if not configured_stems:
        return None

    available = dict(stem_files or {})
    if input_audio is not None:
        available.setdefault("full_mix", input_audio)
    selected = {
        stem_name: available[stem_name]
        for stem_name in configured_stems
        if stem_name in available
    }
    return selected or None


class H3PromptPipeline:
    """Application service for H3-structured prompt generation (stage 8.5)."""

    defer_until_references = True

    required_keys = {
        "scene_prompts_json",
        "stage1_segments",
        "concept_prompts",
        "scene_details",
        "global_context",
        "h3_prompts_json",
        "app_config",
        "config",
    }
    produced_keys = {"h3_prompts"}

    def __init__(
        self,
        *,
        llm_factory: Callable[[Any], Any],
        h3_prompt_builder_factory: H3PromptBuilderFactory,
        dspy_prompt_builder_factory: H3PromptBuilderFactory | None = None,
        checkpoint_store_factory: Callable[[GenerateRenderPlanContext], Any] | None = None,
    ):
        self.llm_factory = llm_factory
        self.h3_prompt_builder_factory = h3_prompt_builder_factory
        self.dspy_prompt_builder_factory = dspy_prompt_builder_factory
        self.checkpoint_store_factory = checkpoint_store_factory

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        app_config = context["app_config"]
        config = context["config"]
        stage1_segments = context["stage1_segments"]
        concept_prompts = context["concept_prompts"]
        scene_details = context["scene_details"]
        global_context = context["global_context"]
        h3_prompts_json = context["h3_prompts_json"]
        artifact_store = context["artifact_store"]
        log_step = context["log_step"]
        log_file = context["log_file"]

        relay_path = context.setdefault("ltx_prompt_relay_json", None)
        if relay_path is not None:
            stage1_segments = _attach_relay_segments(
                stage1_segments,
                artifact_store.read_json(relay_path),
            )
        scene_prompts_path = context.setdefault("scene_prompts_json", None)
        if scene_prompts_path is not None:
            try:
                stage1_segments = _attach_subject_directives(
                    stage1_segments,
                    artifact_store.read_json(scene_prompts_path),
                )
            except (FileNotFoundError, KeyError):
                pass
        beat_path = context.setdefault("beat_json", None)
        if beat_path is not None:
            try:
                beat_data = artifact_store.read_json(beat_path)
            except FileNotFoundError:
                beat_data = None
            if isinstance(beat_data, dict):
                stage1_segments = _attach_beat_events(stage1_segments, beat_data)

        log_step("8.5. H3 Structured Prompts")
        llm = self.llm_factory(app_config)
        builder_factory = self.h3_prompt_builder_factory
        reporter = context["reporter"] if "reporter" in context.keys() else None
        try:
            model_spec = resolve_model_type(config.video_pipeline)
        except ValueError as exc:
            model_spec = None
            if reporter is not None:
                reporter.message(
                    f"video_pipeline '{config.video_pipeline}' has no H3 model spec ({exc}); "
                    "falling back to the legacy H3 prompt builder with T2V mode",
                )
        if model_spec and model_spec.is_minimax_h3 and self.dspy_prompt_builder_factory:
            builder_factory = self.dspy_prompt_builder_factory
        builder = builder_factory(llm)
        checkpoint_store = (
            self.checkpoint_store_factory(context)
            if model_spec and model_spec.is_minimax_h3 and self.checkpoint_store_factory is not None
            else None
        )
        generator_revision = _generator_revision(app_config, builder)
        selected_scene_numbers = (
            context["selected_scene_numbers"]
            if "selected_scene_numbers" in context.keys()
            else None
        )
        selected_scene_selection_complete = bool(
            context["selected_scene_selection_complete"]
            if "selected_scene_selection_complete" in context.keys()
            else False
        )

        mode = model_spec.prompt_mode.value if model_spec else PromptMode.T2V.value
        stem_files = context["stem_files"] if "stem_files" in context.keys() else None
        audio_paths = stem_files
        if model_spec and model_spec.prompt_mode is PromptMode.R2V:
            configured = _configured_audio_paths(config, stem_files, getattr(config, "input_audio", None))
            if configured:
                audio_paths = dict(stem_files or {})
                if getattr(config, "input_audio", None) is not None:
                    audio_paths["full_mix"] = config.input_audio

        progress = SubStepProgress(reporter, "H3 prompts", len(stage1_segments))
        builder.build_all_h3_prompts(
            stage1_segments=stage1_segments,
            concept_prompts=concept_prompts,
            scene_details=scene_details,
            global_context=global_context,
            mode=mode,
            video_type="music_video",
            output_json_path=h3_prompts_json,
            artifact_store=artifact_store,
            audio_paths=audio_paths,
            reference_root=getattr(config, "project_dir", None),
            progress_callback=lambda current, total: progress.update(current),
            status_callback=lambda current, total, status: (
                reporter.message(
                    f"[cyan]H3 prompts: {current}/{total} scenes - "
                    f"{'start' if status == 'started' else 'reused' if status == 'reused' else 'completed'}[/cyan]",
                )
                if reporter is not None
                else None
            ),
            warning_callback=(
                getattr(reporter, "warning", None)
                if reporter is not None
                else None
            ),
            checkpoint_store=checkpoint_store,
            generator_revision=generator_revision,
            preserve_existing_aggregate=selected_scene_numbers is not None,
            reuse_checkpoints=selected_scene_numbers is None or selected_scene_selection_complete,
        )
        log_file("H3 Prompts JSON", h3_prompts_json)
        context["h3_prompts"] = artifact_store.read_json(h3_prompts_json)
        if model_spec and model_spec.is_minimax_h3:
            not_approved = [
                str(item.get("segment_id"))
                for item in context["h3_prompts"]
                if (item.get("prompt_judge") or {}).get("verdict") != "good"
            ]
            if not_approved:
                raise FeverSlopDataError(
                    "MiniMax H3 prompt validation blocked render preparation for scenes: "
                    + ", ".join(not_approved),
                )
        if reporter is not None:
            bad_judgements = []
            for item in context["h3_prompts"]:
                judge = item.get("prompt_judge") or {}
                if judge.get("verdict") == "bad":
                    bad_judgements.append(item)
                    reporter.message(
                        "[yellow]H3 prompt judge: BAD for "
                        f"{item.get('segment_id')}; prompt saved and pipeline continues: "
                        f"{'; '.join(str(issue) for issue in judge.get('issues') or [])}[/yellow]",
                    )
            if bad_judgements:
                reporter.message(
                    "[yellow]H3 prompt judge summary: "
                    f"{len(bad_judgements)} scene(s) marked BAD. "
                    "Prompts were saved; review and optionally correct them manually "
                    "before rendering: "
                    + ", ".join(str(item.get("segment_id")) for item in bad_judgements)
                    + "[/yellow]",
                )
            else:
                reporter.message("[green]H3 prompt judge summary: all generated prompts marked GOOD.[/green]")
        return context


def _generator_revision(app_config: Any, builder: Any) -> dict[str, Any]:
    revision = {"builder": f"{type(builder).__module__}.{type(builder).__qualname__}"}
    builder_revision = getattr(builder, "checkpoint_revision", None)
    if callable(builder_revision):
        revision.update(builder_revision())
    llm_config = getattr(app_config, "llm", None)
    if llm_config is not None:
        model_for = getattr(llm_config, "model_for", None)
        if callable(model_for):
            revision["model"] = str(model_for("structured"))
        revision["prompt_judge_attempts"] = int(
            getattr(llm_config, "prompt_judge_attempts", 3),
        )
    return revision
