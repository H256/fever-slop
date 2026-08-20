from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from feverslop.application.render_plan_validation import (
    require_non_empty_render_plan,
    validate_render_plan_timeline,
)


def export_render_plan_to_mlt(
    *,
    render_plan_path: str | Path,
    clip_paths: Sequence[str | Path],
    output_path: str | Path,
    width: int,
    height: int,
    fps: int,
    audio_path: str | Path | None = None,
    project_name: str | None = None,
) -> Path:
    """Write an MLT XML timeline for Shotcut and Kdenlive."""
    plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8-sig"))
    if isinstance(plan, dict):
        plan = plan.get("shots") or plan.get("scenes") or []
    if not isinstance(plan, list):
        raise ValueError(f"Render plan must be a JSON list: {render_plan_path}")
    if len(plan) != len(clip_paths):
        raise ValueError(
            "MLT export requires one rendered clip per render-plan entry "
            f"(got {len(clip_paths)} clips for {len(plan)} entries)"
        )

    require_non_empty_render_plan(plan, render_plan_path=render_plan_path)
    validate_render_plan_timeline(plan, fps=fps, render_plan_path=render_plan_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mlt", {
        "LC_NUMERIC": "C",
        "version": "7.0.0",
        "title": "Shotcut timeline export",
        "producer": "main_bin",
    })
    ET.SubElement(root, "profile", {
        "description": f"Custom {int(width)}x{int(height)} {int(fps)} fps",
        "width": str(int(width)),
        "height": str(int(height)),
        "frame_rate_num": str(int(fps)),
        "frame_rate_den": "1",
        "progressive": "1",
        "sample_aspect_num": "1",
        "sample_aspect_den": "1",
        "display_aspect_num": str(int(width)),
        "display_aspect_den": str(int(height)),
        "colorspace": "709",
    })

    # MLT resolves producer references in document order. Keep the playlists
    # detached until all chains they reference have been appended to the root.
    video_playlist = ET.Element("playlist", {"id": "playlist0", "autoclose": "1"})
    ET.SubElement(video_playlist, "property", {"name": "shotcut:video"}).text = "1"
    ET.SubElement(video_playlist, "property", {"name": "shotcut:name"}).text = "V1"
    audio_playlist = ET.Element("playlist", {"id": "playlist1", "autoclose": "1"})
    ET.SubElement(audio_playlist, "property", {"name": "shotcut:audio"}).text = "1"
    ET.SubElement(audio_playlist, "property", {"name": "shotcut:name"}).text = "A1"
    total_frames = 0
    timeline_cursor = 0

    indexed_entries = list(enumerate(zip(plan, clip_paths, strict=True), start=1))
    indexed_entries.sort(
        key=lambda item: (
            item[1][0].get("abs_start_seconds") is None if isinstance(item[1][0], dict) else True,
            float(item[1][0].get("abs_start_seconds", 0.0)) if isinstance(item[1][0], dict) else 0.0,
            item[0],
        )
    )
    for index, (entry, clip_path) in indexed_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Render plan entry {index} must be an object")
        scene_number = int(entry.get("scene") or entry.get("scene_number") or index)
        duration = float(entry.get("duration_seconds", 0.0))
        if duration <= 0:
            raise ValueError(f"Render plan scene {scene_number} has no positive duration")
        path = Path(clip_path)
        if not path.is_file():
            raise FileNotFoundError(f"Rendered clip does not exist: {path}")
        start_seconds = entry.get("abs_start_seconds")
        if start_seconds is not None:
            start_frame = max(0, round(float(start_seconds) * int(fps)))
            end_frame = max(start_frame + 1, round((float(start_seconds) + duration) * int(fps)))
            frames = end_frame - start_frame
        else:
            start_frame = timeline_cursor
            frames = max(1, math.ceil(duration * int(fps)))
        if start_frame < timeline_cursor:
            overlap = timeline_cursor - start_frame
            if overlap <= 1:
                # Render-plan seconds are floating point values while MLT is
                # frame-based. Treat a one-frame boundary discrepancy as a
                # contiguous cut instead of a real overlap.
                start_frame = timeline_cursor
            else:
                raise ValueError(
                    "MLT export cannot represent overlapping render-plan entries: "
                    f"scene {scene_number} starts at frame {start_frame}, "
                    f"before frame {timeline_cursor}"
                )
        if start_frame > timeline_cursor:
            ET.SubElement(video_playlist, "blank", {"length": str(start_frame - timeline_cursor)})
        producer_id = f"video_{index:04}"
        _add_avformat_producer(
            root,
            producer_id,
            path,
            output,
            frames - 1,
            caption=f"Scene {scene_number:04d}",
            comment=_scene_comment(entry),
        )
        ET.SubElement(video_playlist, "entry", {
            "producer": producer_id,
            "in": "0",
            "out": str(frames - 1),
        })
        timeline_cursor = start_frame + frames
        total_frames = max(total_frames, timeline_cursor)

    if audio_path is not None:
        audio = Path(audio_path)
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio}")
        _add_avformat_producer(
            root,
            "audio_original",
            audio,
            output,
            max(0, total_frames - 1),
            caption="Original audio",
        )
        ET.SubElement(audio_playlist, "entry", {
            "producer": "audio_original",
            "in": "0",
            "out": str(max(0, total_frames - 1)),
        })

    root.append(video_playlist)
    root.append(audio_playlist)
    ET.SubElement(root, "property", {"name": "shotcut:projectNotes"}).text = _project_notes(
        project_name=project_name or output.stem,
        scene_count=len(plan),
        first_scene=int(plan[0].get("scene") or plan[0].get("scene_number") or 1),
        last_scene=int(plan[-1].get("scene") or plan[-1].get("scene_number") or len(plan)),
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        audio_path=audio_path,
        render_plan_path=render_plan_path,
    )

    main_bin = ET.SubElement(root, "playlist", {"id": "main_bin"})
    ET.SubElement(main_bin, "property", {"name": "xml_retain"}).text = "1"
    for producer_id in [f"video_{index:04}" for index in range(1, len(plan) + 1)]:
        ET.SubElement(main_bin, "entry", {"producer": producer_id})
    if audio_path is not None:
        ET.SubElement(main_bin, "entry", {"producer": "audio_original"})

    background_producer = ET.SubElement(root, "producer", {"id": "black", "in": "0", "out": str(max(0, total_frames - 1))})
    ET.SubElement(background_producer, "property", {"name": "mlt_service"}).text = "color"
    ET.SubElement(background_producer, "property", {"name": "resource"}).text = "black"
    background = ET.SubElement(root, "playlist", {"id": "background"})
    ET.SubElement(background, "property", {"name": "shotcut:video"}).text = "1"
    ET.SubElement(background, "property", {"name": "shotcut:name"}).text = "Background"
    ET.SubElement(background, "entry", {"producer": "black", "in": "0", "out": str(max(0, total_frames - 1))})

    tractor = ET.SubElement(root, "tractor", {
        "id": "main",
        "in": "0",
        "out": str(max(0, total_frames - 1)),
    })
    ET.SubElement(tractor, "property", {"name": "shotcut"}).text = "1"
    ET.SubElement(tractor, "track", {"producer": "background"})
    ET.SubElement(tractor, "track", {"producer": "playlist0"})
    if audio_path is not None:
        ET.SubElement(tractor, "track", {"producer": "playlist1", "hide": "video"})

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def _add_avformat_producer(
    root: ET.Element,
    producer_id: str,
    path: Path,
    project_file: Path,
    out_frame: int,
    caption: str,
    comment: str | None = None,
) -> None:
    producer = ET.SubElement(root, "chain", {
        "id": producer_id,
        "in": "0",
        "out": str(out_frame),
    })
    ET.SubElement(producer, "property", {"name": "resource"}).text = _relative_path(path, project_file)
    ET.SubElement(producer, "property", {"name": "mlt_service"}).text = "avformat"
    ET.SubElement(producer, "property", {"name": "shotcut:caption"}).text = caption
    if comment:
        ET.SubElement(producer, "property", {"name": "shotcut:comment"}).text = comment


def _relative_path(path: Path, project_file: Path) -> str:
    return Path(os.path.relpath(path.resolve(), project_file.parent.resolve())).as_posix()


def _scene_comment(entry: dict) -> str | None:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    fields = (
        ("Story", metadata.get("base_concept")),
        ("Camera", metadata.get("camera_motion")),
        ("Character", metadata.get("character_motion")),
    )
    lines = [f"{label}: {value}" for label, value in fields if isinstance(value, str) and value.strip()]
    if entry.get("seed") is not None:
        lines.append(f"Seed: {entry['seed']}")
    return "\n".join(lines) or None


def _project_notes(
    *,
    project_name: str,
    scene_count: int,
    first_scene: int,
    last_scene: int,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
    audio_path: str | Path | None,
    render_plan_path: str | Path,
) -> str:
    duration_seconds = max(0, round(total_frames / int(fps)))
    minutes, seconds = divmod(duration_seconds, 60)
    audio_name = Path(audio_path).name if audio_path is not None else "none"
    return "\n".join([
        f"Project: {project_name}",
        "Pipeline: FeverSlop",
        f"Scenes: {scene_count} (Scene {first_scene:04d} - Scene {last_scene:04d})",
        f"Profile: {int(width)}x{int(height)} @ {int(fps)} fps",
        f"Duration: {minutes:02d}:{seconds:02d}",
        "Video track: V1 - scene clips",
        f"Audio track: A1 - Original audio ({audio_name})",
        f"Render plan: {Path(render_plan_path).name}",
        "Per-scene story, motion, and seed details are stored in each clip comment.",
        "Regenerate: --stage export_timeline --format mlt",
    ])
