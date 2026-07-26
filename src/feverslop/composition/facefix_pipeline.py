from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_facefix_backend import ComfyUIFaceFixRenderBackend
from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.face_compositor import FaceCompositor
from feverslop.adapters.insightface_extractor import InsightFaceExtractor
from feverslop.adapters.insightface_tracker import InsightFaceTracker
from feverslop.application.facefix_pipeline import FaceFixPipelineStep, FaceFixRequest
from feverslop.config.app_config import AppConfig
from feverslop.domain.face_detection import FaceRepairData
from feverslop.domain.facefix_rendering import FaceFixConfig
from feverslop.path_utils import coerce_local_path
from feverslop.ports.reporting import ConsoleReporter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceFixCompositionOptions:
    app_config_path: str | Path = "./app_config.json"
    workflow_path: str | Path = ""
    crop_workflow_path: str | Path = ""
    scenes_dir: str | Path = ""
    project_dir: str | Path | None = None
    scene_numbers: list[int] | None = None
    reference_images: list[Path] | None = None
    skip_existing: bool = True
    postprocess: bool = True
    ffmpeg_path: str = "ffmpeg"
    postprocess_reencode: bool = True
    ffmpeg_debug: bool = False
    keyframe_indices: str = "0,16,32,48"
    guiding_strength: float = 0.2
    cond_image_strength: float = 0.5
    temporal_tile_size: int = 56
    temporal_overlap: int = 24
    temporal_overlap_cond_strength: float = 0.5
    crop_size: int = 768
    anchor_interval: int = 16
    crop_padding: float = 0.25
    feather_pixels: int = 48
    color_match_strength: float = 0.65
    recognition_threshold: float = 0.85
    use_crop_pipeline: bool = True


def build_facefix_step(
    options: FaceFixCompositionOptions,
    *,
    console: Console | None = None,
) -> FaceFixPipelineStep:
    app_config = AppConfig.load(options.app_config_path)
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )

    config = FaceFixConfig(
        keyframe_indices=options.keyframe_indices,
        guiding_strength=options.guiding_strength,
        cond_image_strength=options.cond_image_strength,
        temporal_tile_size=options.temporal_tile_size,
        temporal_overlap=options.temporal_overlap,
        temporal_overlap_cond_strength=options.temporal_overlap_cond_strength,
        postprocess=options.postprocess,
        ffmpeg_path=options.ffmpeg_path,
    )

    backend = ComfyUIFaceFixRenderBackend(
        client=client,
        workflow_path=coerce_local_path(options.workflow_path),
        project_dir=coerce_local_path(options.project_dir) if options.project_dir else None,
        config=config,
        postprocess=options.postprocess,
        ffmpeg_path=options.ffmpeg_path,
        postprocess_reencode=options.postprocess_reencode,
        ffmpeg_debug=options.ffmpeg_debug,
        model_resolver=model_resolver,
    )

    reporter = ConsoleReporter(console) if console is not None else None
    return FaceFixPipelineStep(
        backend=backend,
        config=config,
        reporter=reporter,
    )


def run_facefix(
    options: FaceFixCompositionOptions,
    *,
    console: Console | None = None,
) -> list[Path]:
    """Build and execute the FaceFix pipeline.

    When use_crop_pipeline is True, uses the new crop-and-composite approach.
    Otherwise falls back to the legacy full-res approach.
    """
    if options.use_crop_pipeline:
        return _run_crop_facefix(options, console=console)
    return _run_legacy_facefix(options, console=console)


def _run_legacy_facefix(
    options: FaceFixCompositionOptions,
    *,
    console: Console | None = None,
) -> list[Path]:
    step = build_facefix_step(options, console=console)
    scenes_dir = coerce_local_path(options.scenes_dir)

    request = FaceFixRequest(
        scenes_dir=scenes_dir,
        scene_numbers=options.scene_numbers,
        reference_images=options.reference_images or [],
        skip_existing=options.skip_existing,
    )

    return step.execute(request)


def _run_crop_facefix(
    options: FaceFixCompositionOptions,
    *,
    console: Console | None = None,
) -> list[Path]:
    app_config = AppConfig.load(options.app_config_path)
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )

    config = FaceFixConfig(
        keyframe_indices=options.keyframe_indices,
        guiding_strength=options.guiding_strength,
        cond_image_strength=options.cond_image_strength,
        temporal_tile_size=options.temporal_tile_size,
        temporal_overlap=options.temporal_overlap,
        temporal_overlap_cond_strength=options.temporal_overlap_cond_strength,
        postprocess=options.postprocess,
        ffmpeg_path=options.ffmpeg_path,
    )

    crop_workflow_path = coerce_local_path(options.crop_workflow_path)
    if not crop_workflow_path.exists():
        crop_workflow_path = Path("workflows/video_ltxv_facefix_crop.json")
    if not crop_workflow_path.exists():
        crop_workflow_path = coerce_local_path(options.workflow_path)

    crop_backend = ComfyUIFaceFixCropBackend(
        client=client,
        workflow_path=crop_workflow_path,
        project_dir=coerce_local_path(options.project_dir) if options.project_dir else None,
        config=config,
        postprocess=options.postprocess,
        ffmpeg_path=options.ffmpeg_path,
        postprocess_reencode=options.postprocess_reencode,
        ffmpeg_debug=options.ffmpeg_debug,
        model_resolver=model_resolver,
    )

    extractor = InsightFaceExtractor()
    tracker = InsightFaceTracker(extractor)
    compositor = FaceCompositor(
        feather_pixels=options.feather_pixels,
        color_match_strength=options.color_match_strength,
        diagnostic=True,
    )

    scenes_dir = coerce_local_path(options.scenes_dir)
    project_dir = coerce_local_path(options.project_dir) if options.project_dir else None
    scene_numbers = options.scene_numbers
    if scene_numbers is None:
        scene_numbers = sorted(
            int(d.name.split("_")[1])
            for d in scenes_dir.iterdir()
            if d.is_dir() and d.name.startswith("scene_")
        )

    reporter = ConsoleReporter(console) if console is not None else None
    results = []

    actor_sheets = options.reference_images or []
    if not actor_sheets and project_dir:
        actors_dir = project_dir / "output" / "references" / "actors"
        if actors_dir.is_dir():
            for actor_dir in sorted(actors_dir.iterdir()):
                if not actor_dir.is_dir():
                    continue
                views = actor_dir / "views"
                if views.is_dir():
                    for sheet in sorted(views.glob("*sheet.png")):
                        actor_sheets.append(sheet)

    actor_embeddings: dict[str, np.ndarray] = {}
    actor_id_map: dict[str, Path] = {}
    for sheet_path in actor_sheets:
        actor_id = sheet_path.parent.parent.name
        face_ref = extractor.extract_face_from_image(sheet_path)
        if face_ref is not None:
            emb = extractor.extract_embedding(face_ref)
            if emb is not None:
                actor_embeddings[actor_id] = emb
                actor_id_map[actor_id] = sheet_path

    for scene_number in scene_numbers:
        scene_dir = scenes_dir / f"scene_{scene_number:04d}"
        source = scene_dir / "final.mp4"
        if not source.exists():
            if reporter:
                reporter.message(f"[yellow]WARN[/yellow] FaceFix: final.mp4 missing for scene {scene_number}, skipping")
            continue

        final_facefix = scene_dir / "final_facefix.mp4"
        if options.skip_existing and final_facefix.exists():
            results.append(final_facefix)
            if reporter:
                reporter.message(f"[green]OK[/green] FaceFix scene {scene_number}: already exists")
            continue

        original_frames = _load_video_frames(source)
        if original_frames is None:
            if reporter:
                reporter.message(f"[yellow]WARN[/yellow] FaceFix: cannot load frames for scene {scene_number}")
            continue

        current_frames = original_frames.copy()
        face_repairs_for_composite: list[FaceRepairData] = []

        for actor_id in actor_embeddings:
            facefix_dir = scene_dir / "facefix" / actor_id
            repaired_dir = facefix_dir / "repaired"
            repaired_mp4 = facefix_dir / f"repaired_{actor_id}.mp4"

            if options.skip_existing and repaired_mp4.exists():
                if reporter:
                    reporter.message(f"[green]OK[/green] FaceFix scene {scene_number}/{actor_id}: already exists")
                track_result = tracker.track_video(
                    video_path=source,
                    actor_embeddings={actor_id: actor_embeddings[actor_id]},
                    crop_size=options.crop_size,
                    anchor_interval=options.anchor_interval,
                    crop_padding=options.crop_padding,
                    output_dir=facefix_dir,
                    actor_id=actor_id,
                )
                face_repairs_for_composite.append(FaceRepairData(
                    actor_id=actor_id,
                    repaired_frames_dir=repaired_dir,
                    track_entries=track_result.entries,
                    crop_size=options.crop_size,
                ))
                continue

            if reporter:
                reporter.message(f"FaceFix scene {scene_number}/{actor_id}: tracking...")

            track_result = tracker.track_video_with_encoder(
                video_path=source,
                actor_embeddings={actor_id: actor_embeddings[actor_id]},
                crop_size=options.crop_size,
                anchor_interval=options.anchor_interval,
                crop_padding=options.crop_padding,
                output_dir=facefix_dir,
                actor_id=actor_id,
                ffmpeg_path=options.ffmpeg_path,
            )

            if not track_result.entries or track_result.crop_mp4_path is None:
                if reporter:
                    reporter.message(f"[yellow]WARN[/yellow] FaceFix scene {scene_number}/{actor_id}: no tracks found, skipping")
                continue

            if reporter:
                reporter.message(f"FaceFix scene {scene_number}/{actor_id}: rendering...")

            crop_backend.render_scene(
                scene_number=scene_number,
                face_crop_mp4=track_result.crop_mp4_path,
                anchors_dir=facefix_dir / "anchors",
                output_dir=facefix_dir,
                actor_id=actor_id,
            )

            repaired_video = facefix_dir / f"repaired_{actor_id}.mp4"
            if not repaired_video.exists():
                repaired_video = facefix_dir / "raw_facefix_crop.mp4"

            if repaired_video.exists():
                repaired_frames = _load_video_frames(repaired_video)
                if repaired_frames is not None:
                    repaired_dir.mkdir(parents=True, exist_ok=True)
                    for i, frame in enumerate(repaired_frames):
                        cv2.imwrite(str(repaired_dir / f"repaired_{i:06d}.png"), frame)

                face_repairs_for_composite.append(FaceRepairData(
                    actor_id=actor_id,
                    repaired_frames_dir=repaired_dir,
                    track_entries=track_result.entries,
                    crop_size=options.crop_size,
                ))

        if face_repairs_for_composite:
            if reporter:
                reporter.message(f"FaceFix scene {scene_number}: compositing {len(face_repairs_for_composite)} actor(s)...")

            composite_result = compositor.composite(
                face_repairs=face_repairs_for_composite,
                original_frames=current_frames,
                output_dir=scene_dir / "facefix",
            )

            _save_video_frames(composite_result.composited_frames, final_facefix, source, options.ffmpeg_path)
            results.append(final_facefix)

            if reporter:
                reporter.message(f"[green]OK[/green] FaceFix scene {scene_number}: {final_facefix}")
        else:
            import shutil
            shutil.copy2(source, final_facefix)
            results.append(final_facefix)

    return results


def _load_video_frames(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    return np.array(frames)


def _save_video_frames(frames: np.ndarray, output_path: Path, reference_video: Path, ffmpeg_path: str = "ffmpeg") -> None:
    import subprocess
    temp_dir = output_path.parent / "temp_frames_export"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(temp_dir / f"frame_{i:06d}.png"), bgr)

    cap = cv2.VideoCapture(str(reference_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    subprocess.run(
        [
            ffmpeg_path,
            "-r", str(fps),
            "-i", str(temp_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "18",
            "-y",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
