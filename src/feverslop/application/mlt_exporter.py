from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from feverslop.application.render_plan_validation import (
    compute_timeline_intervals,
    load_render_plan_entries,
    validate_render_plan_timeline,
)
from feverslop.utils.io import atomic_write_text


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
    entries = load_render_plan_entries(render_plan_path)
    plan = [entry[0] for entry in entries]
    if len(plan) != len(clip_paths):
        raise ValueError(
            "MLT export requires one rendered clip per render-plan entry "
            f"(got {len(clip_paths)} clips for {len(plan)} entries)",
        )

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
    clip_audio_playlist = ET.Element("playlist", {"id": "playlist1", "autoclose": "1"})
    ET.SubElement(clip_audio_playlist, "property", {"name": "shotcut:audio"}).text = "1"
    ET.SubElement(clip_audio_playlist, "property", {"name": "shotcut:name"}).text = "A1 - Clip audio"
    original_audio_playlist = ET.Element("playlist", {"id": "playlist2", "autoclose": "1"})
    ET.SubElement(original_audio_playlist, "property", {"name": "shotcut:audio"}).text = "1"
    ET.SubElement(original_audio_playlist, "property", {"name": "shotcut:name"}).text = "A2 - Original audio"
    total_frames, _cursor = _append_scene_clips(
        root,
        video_playlist,
        clip_audio_playlist,
        compute_timeline_intervals(entries, fps=fps),
        clip_paths,
        output,
    )

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
        ET.SubElement(original_audio_playlist, "entry", {
            "producer": "audio_original",
            "in": "0",
            "out": str(max(0, total_frames - 1)),
        })

    root.append(video_playlist)
    root.append(clip_audio_playlist)
    root.append(original_audio_playlist)
    _append_project_notes(root, plan, output, width, height, fps, total_frames, audio_path, render_plan_path, project_name)
    _append_bin_and_background(root, plan, total_frames, audio_path)
    _append_tractor(root, total_frames, audio_path)

    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    atomic_write_text(output, payload)
    return output


def _append_scene_clips(
    root: ET.Element,
    video_playlist: ET.Element,
    clip_audio_playlist: ET.Element,
    intervals,
    clip_paths: Sequence[str | Path],
    output: Path,
) -> tuple[int, int]:
    """Add one producer and playlist entry per rendered clip; return (total_frames, cursor)."""
    timeline_cursor = 0
    total_frames = 0
    # Intervals are sorted by timeline position; clip_paths arrive in render-plan
    # (input) order. Reorder the clips by each interval's original 1-based plan
    # index so every clip is attached to the entry it was rendered for, even
    # when the plan's entries are not already chronological.
    ordered_clips = [clip_paths[interval[0] - 1] for interval in intervals]
    for _position, (interval, clip_path) in enumerate(zip(intervals, ordered_clips, strict=True)):
        original_index, entry, scene_number, duration, _start_seconds, start_frame, end_frame = interval
        path = Path(clip_path)
        if not path.is_file():
            raise FileNotFoundError(f"Rendered clip does not exist: {path}")
        frames = end_frame - start_frame
        if start_frame < timeline_cursor:
            if end_frame <= timeline_cursor:
                raise ValueError(
                    "MLT export cannot represent overlapping render-plan entries: "
                    f"scene {scene_number} ends at frame {end_frame}, "
                    f"before frame {timeline_cursor}",
                )
            # Mirrors the shared validation's accumulated-drift correction:
            # trim the overlapping tail so the cut stays contiguous.
            start_frame = timeline_cursor
            frames = end_frame - start_frame
        if start_frame > timeline_cursor:
            ET.SubElement(video_playlist, "blank", {"length": str(start_frame - timeline_cursor)})
        producer_id = f"video_{original_index:04}"
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
        ET.SubElement(clip_audio_playlist, "entry", {
            "producer": producer_id,
            "in": "0",
            "out": str(frames - 1),
        })
        timeline_cursor = start_frame + frames
        total_frames = max(total_frames, timeline_cursor)
    return total_frames, timeline_cursor


def _append_project_notes(
    root: ET.Element,
    plan: list[dict],
    output: Path,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
    audio_path: str | Path | None,
    render_plan_path: str | Path,
    project_name: str | None,
) -> None:
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


def _append_bin_and_background(
    root: ET.Element,
    plan: list[dict],
    total_frames: int,
    audio_path: str | Path | None,
) -> None:
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


def _append_tractor(
    root: ET.Element,
    total_frames: int,
    audio_path: str | Path | None,
) -> None:
    tractor = ET.SubElement(root, "tractor", {
        "id": "main",
        "in": "0",
        "out": str(max(0, total_frames - 1)),
    })
    ET.SubElement(tractor, "property", {"name": "shotcut"}).text = "1"
    ET.SubElement(tractor, "track", {"producer": "background"})
    ET.SubElement(tractor, "track", {"producer": "playlist0", "hide": "audio"})
    ET.SubElement(tractor, "track", {"producer": "playlist1", "hide": "video"})
    if audio_path is not None:
        ET.SubElement(tractor, "track", {"producer": "playlist2", "hide": "video"})


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
    # Keep the sub-second fraction: derive the display from an exact
    # millisecond total so the duration is not rounded to whole seconds.
    fps_i = int(fps)
    duration_total_ms = round((total_frames * 1000.0 / fps_i) if fps_i else 0.0)
    minutes, remainder_ms = divmod(max(0, duration_total_ms), 60_000)
    seconds, millis = divmod(remainder_ms, 1_000)
    audio_name = Path(audio_path).name if audio_path is not None else "none"
    return "\n".join([
        f"Project: {project_name}",
        "Pipeline: FeverSlop",
        f"Scenes: {scene_count} (Scene {first_scene:04d} - Scene {last_scene:04d})",
        f"Profile: {int(width)}x{int(height)} @ {int(fps)} fps",
        f"Duration: {minutes:02d}:{seconds:02d}.{millis:03d}",
        "Video track: V1 - scene clips",
        "Audio track: A1 - Clip audio (embedded scene audio; removable)",
        f"Audio track: A2 - Original audio ({audio_name})",
        f"Render plan: {Path(render_plan_path).name}",
        "Per-scene story, motion, and seed details are stored in each clip comment.",
        "Regenerate: --stage export_timeline --format mlt",
    ])
