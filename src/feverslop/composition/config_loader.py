from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import re

from feverslop.path_utils import coerce_local_path
from feverslop.scene_artifacts import SceneArtifactLayout


@dataclass(frozen=True)
class PipelineRunContext:
    artifact_layout: SceneArtifactLayout
    project_config_path: Path
    project_config_dir: Path
    input_audio: Path
    song_id: str
    project_file_stem: str
    project_output_dir: Path
    timeline_dir: Path
    prompts_dir: Path
    render_dir: Path
    stage1_segments: Path
    resolved_context: Path
    concept_prompts: Path
    scene_details: Path
    scene_prompts: Path
    render_plan: Path
    reference_plan: Path
    ingredients_plan: Path
    references_dir: Path
    compact_plan: Path
    anchored_plan: Path
    storyboard_dir: Path
    storyboard_page: Path
    ltx_dir: Path
    ltx_debug_dir: Path
    final_concat_video: Path
    final_concat: Path
    final_concat_scene_audio_debug: Path
    concat_list: Path


@dataclass(frozen=True)
class PipelineRunResult:
    render_plan_path: Path
    final_video_path: Path | None = None
    video_only_path: Path | None = None


@dataclass
class PipelineRunState:
    args: argparse.Namespace
    context: PipelineRunContext
    app_config_path: Path
    storyboard_workflow: Path
    reference_hero_workflow: Path
    reference_edit_workflow: Path
    msr_workflow: Path
    ingredients_workflow: Path
    relay_workflow: Path
    single_prompt_workflow: Path
    plan_for_next_step: Path
    video_only_path: Path | None = None
    final_video_path: Path | None = None


def convert_to_safe_file_stem(value, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    safe = safe.strip("._-")
    return safe or fallback


def rewrite_concat_list(rendered_files: list[Path], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    concat_file = output_dir / "concat_list.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with concat_file.open("w", encoding="utf-8") as f:
        for path in rendered_files:
            f.write(f"file '{Path(path).resolve().as_posix()}'\n")
    return concat_file


def collect_render_plan_scene_clips(
    render_plan_path: str | Path,
    output_dir: str | Path,
    *,
    layout: SceneArtifactLayout | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8-sig"))
    clips: list[Path] = []
    missing: list[Path] = []
    for scene in render_plan:
        scene_number = int(scene["scene"])
        candidates = ([layout.scene_final_video(scene_number)] if layout else []) + [
            output_dir / f"scene_{scene_number:04}.mp4",
            output_dir / "final" / f"scene_{scene_number:04}.mp4",
        ]
        clip = next((candidate for candidate in candidates if candidate.exists()), None)
        if clip is None:
            missing.append(layout.scene_final_video(scene_number) if layout else candidates[0])
            continue
        clips.append(clip)

    if missing:
        raise FileNotFoundError(
            "Cannot build final concat; missing rendered scene clips: "
            + ", ".join(str(path) for path in missing[:10])
        )
    return clips


def count_render_plan_items(render_plan_path: str | Path, scene_numbers: set[int] | None = None, limit: int | None = None) -> int:
    render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8-sig"))
    if scene_numbers is not None:
        render_plan = [scene for scene in render_plan if int(scene["scene"]) in scene_numbers]
    if limit is not None:
        render_plan = render_plan[:limit]
    return len(render_plan)


def runner_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_runner_path(value: str | Path) -> Path:
    return coerce_local_path(value, base_dir=runner_root())


def build_run_context(args: argparse.Namespace) -> PipelineRunContext:
    project_config = args.project_config
    project_root = args.project_root

    if not project_config:
        if not project_root:
            project_root = os.fspath(Path("projects") / "my_first_project")

        project_root_path = resolve_runner_path(project_root)
        if project_root_path.is_file():
            project_config_path = project_root_path
        else:
            project_config_path = project_root_path / "config.json"
    else:
        project_config_path = resolve_runner_path(project_config)

    project_config_path = project_config_path.resolve()
    project_config_dir = project_config_path.parent
    project_config_json = json.loads(project_config_path.read_text(encoding="utf-8-sig"))
    input_audio = coerce_local_path(str(project_config_json["input_audio"]), base_dir=project_config_dir)
    input_audio = input_audio.resolve()

    song_id = input_audio.stem
    project_file_stem = convert_to_safe_file_stem(project_config_json.get("project_name"), song_id)
    project_output_dir = project_config_dir / "output"
    timeline_dir = project_output_dir / "timeline"
    prompts_dir = project_output_dir / "prompts"
    render_dir = project_output_dir / "render"
    artifact_layout = SceneArtifactLayout(project_config_dir)
    storyboard_dir = artifact_layout.storyboard_dir
    vp = getattr(args, "video_pipeline", "ltx_i2v")
    if vp == "ltx_msr":
        ltx_dir = render_dir / "ltx_msr"
        ltx_debug_dir = render_dir / "ltx_msr_debug"
    elif vp == "ltx_ingredients":
        ltx_dir = render_dir / "ltx_ingredients"
        ltx_debug_dir = render_dir / "ltx_ingredients_debug"
    else:
        ltx_dir = render_dir / f"ltx_{args.render_mode}"
        ltx_debug_dir = render_dir / f"ltx_{args.render_mode}_debug"
    if args.smoke_only:
        ltx_dir = ltx_dir.with_name(ltx_dir.name + "_smoke")

    return PipelineRunContext(
        artifact_layout=artifact_layout,
        project_config_path=project_config_path,
        project_config_dir=project_config_dir,
        input_audio=input_audio,
        song_id=song_id,
        project_file_stem=project_file_stem,
        project_output_dir=project_output_dir,
        timeline_dir=timeline_dir,
        prompts_dir=prompts_dir,
        render_dir=render_dir,
        stage1_segments=timeline_dir / f"stage1_segments_{song_id}.json",
        resolved_context=prompts_dir / f"resolved_context_{song_id}.json",
        concept_prompts=prompts_dir / f"concept_prompts_{song_id}.json",
        scene_details=prompts_dir / f"scene_details_{song_id}.json",
        scene_prompts=prompts_dir / f"scene_prompts_{song_id}.json",
        render_plan=artifact_layout.base_plan,
        reference_plan=artifact_layout.references_plan,
        ingredients_plan=artifact_layout.ingredients_plan,
        references_dir=artifact_layout.references_dir,
        compact_plan=artifact_layout.compact_plan,
        anchored_plan=artifact_layout.anchored_plan,
        storyboard_dir=storyboard_dir,
        storyboard_page=storyboard_dir / "index.html",
        ltx_dir=ltx_dir,
        ltx_debug_dir=ltx_debug_dir,
        final_concat_video=artifact_layout.video_only,
        final_concat=artifact_layout.movie,
        final_concat_scene_audio_debug=artifact_layout.final_dir / "scene_audio_debug.mp4",
        concat_list=artifact_layout.final_dir / "concat_list.txt",
    )
