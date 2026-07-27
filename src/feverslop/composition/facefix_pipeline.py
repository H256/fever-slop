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
from feverslop.adapters.face_debug import FaceDebugAdapter
from feverslop.adapters.face_detector_insightface import InsightFaceDetectorAdapter
from feverslop.adapters.face_identity import FaceIdentityAdapter
from feverslop.adapters.face_mask import FaceMaskAdapter
from feverslop.application.face_pipeline import FacePipeline
from feverslop.application.facefix_pipeline import FaceFixPipelineStep, FaceFixRequest
from feverslop.config.app_config import AppConfig
from feverslop.domain.face_detection import (
    FaceBox,
    FaceProcessingPolicy,
    FaceRepairData,
    FaceTrackEntry,
    FrameResult,
)
from feverslop.domain.facefix_rendering import FaceFixConfig
from feverslop.path_utils import coerce_local_path
from feverslop.ports.reporting import ConsoleReporter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceFixCompositionOptions:
    app_config_path: str | Path = "./app_config.json"
    workflow_path: str | Path = ""
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

    crop_backend = ComfyUIFaceFixCropBackend(
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

    # -- Hexagonal pipeline adapters --
    detector = InsightFaceDetectorAdapter()
    identity_adapter = FaceIdentityAdapter(min_similarity=options.recognition_threshold)
    mask_adapter = FaceMaskAdapter()

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

    # -- Actor sheet discovery (unchanged) --
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

    # -- Extract reference embeddings and register identities --
    from feverslop.adapters.insightface_extractor import InsightFaceExtractor

    extractor = InsightFaceExtractor()
    actor_embeddings: dict[str, np.ndarray] = {}
    for sheet_path in actor_sheets:
        actor_id = sheet_path.parent.parent.name
        face_ref = extractor.extract_face_from_image(sheet_path)
        if face_ref is not None:
            emb = extractor.extract_embedding(face_ref)
            if emb is not None:
                actor_embeddings[actor_id] = emb
                identity_adapter.register_reference(emb, actor_id)

    # -- Per-scene processing --
    for scene_number in scene_numbers:
        scene_dir = scenes_dir / f"scene_{scene_number:04d}"
        source = scene_dir / "final.mp4"
        if not source.exists():
            if reporter:
                reporter.message(
                    f"[yellow]WARN[/yellow] FaceFix: final.mp4 missing for scene "
                    f"{scene_number}, skipping"
                )
            continue

        final_facefix = scene_dir / "final_facefix.mp4"
        if options.skip_existing and final_facefix.exists():
            results.append(final_facefix)
            if reporter:
                reporter.message(
                    f"[green]OK[/green] FaceFix scene {scene_number}: already exists"
                )
            continue

        original_frames = _load_video_frames(source)
        if original_frames is None:
            if reporter:
                reporter.message(
                    f"[yellow]WARN[/yellow] FaceFix: cannot load frames for scene "
                    f"{scene_number}"
                )
            continue

        # -- Build FacePipeline for this scene --
        debug_dir = scene_dir / "facefix" / "debug"
        debug_adapter = FaceDebugAdapter(debug_dir)
        policy = FaceProcessingPolicy(
            min_detection_score=0.5,
            min_identity_score=0.0,
            # Identity check only makes sense with multiple actors to distinguish between.
            enable_identity_check=len(actor_embeddings) > 1,
            track_confirmation_frames=3,
            track_max_missing_frames=6,
            face_crop_expansion=1.0 + options.crop_padding,
            debug_output=True,
        )
        pipeline = FacePipeline(
            detector=detector,
            identity_port=identity_adapter,
            mask_port=mask_adapter,
            debug_port=debug_adapter,
            policy=policy,
        )
        pipeline.reset()

        # -- Process every frame through FacePipeline --
        actor_frames: dict[str, list[tuple[int, FrameResult]]] = {}
        total_frames = len(original_frames)

        for frame_idx in range(total_frames):
            result = pipeline.process_frame(original_frames[frame_idx], frame_idx)
            if result.processed:
                # FrameResult doesn't carry actor_id directly yet (the pipeline's
                # identity_port returns (actor_id, score) but only the score is
                # stored on FrameResult). For single-actor scenes this works
                # because every processed frame belongs to the same actor.
                # Multi-actor support requires adding actor_id to FrameResult.
                matched_actor: str | None = None
                if len(actor_embeddings) == 1:
                    matched_actor = next(iter(actor_embeddings))
                elif result.identity_score is not None:
                    # Identity check passed — pick first registered actor.
                    # TODO: add actor_id to FrameResult for multi-actor scenes.
                    matched_actor = next(iter(actor_embeddings))
                else:
                    matched_actor = "unknown"

                actor_frames.setdefault(matched_actor, []).append(
                    (frame_idx, result)
                )

                if reporter and frame_idx % max(1, total_frames // 40) == 0:
                    reporter.message(
                        f"FaceFix scene {scene_number}: frame "
                        f"{frame_idx}/{total_frames}"
                    )

        current_frames = original_frames.copy()
        face_repairs_for_composite: list[FaceRepairData] = []
        compositor = FaceCompositor(
            feather_pixels=options.feather_pixels,
            color_match_strength=options.color_match_strength,
            diagnostic=True,
        )

        # -- Per-actor crop / render / repair --
        for actor_id, frames_list in actor_frames.items():
            facefix_dir = scene_dir / "facefix" / actor_id
            repaired_dir = facefix_dir / "repaired"
            repaired_mp4 = facefix_dir / f"repaired_{actor_id}.mp4"
            crop_frames_dir = facefix_dir / "crops"
            anchor_dir = facefix_dir / "anchors"

            if options.skip_existing and repaired_mp4.exists():
                if reporter:
                    reporter.message(
                        f"[green]OK[/green] FaceFix scene {scene_number}/{actor_id}: "
                        f"already exists"
                    )
                # Rebuild track entries from pipeline results.
                track_entries = [
                    FaceTrackEntry(
                        frame_index=fi,
                        box=FaceBox(
                            x1=int(result.box.x1),
                            y1=int(result.box.y1),
                            x2=int(result.box.x2),
                            y2=int(result.box.y2),
                            confidence=result.detection_score or 0.0,
                            actor_id=actor_id,
                        ),
                    )
                    for fi, result in frames_list
                    if result.box is not None
                ]
                face_repairs_for_composite.append(
                    FaceRepairData(
                        actor_id=actor_id,
                        repaired_frames_dir=repaired_dir,
                        track_entries=track_entries,
                        crop_size=options.crop_size,
                    )
                )
                continue

            if not frames_list:
                continue

            if reporter:
                reporter.message(
                    f"FaceFix scene {scene_number}/{actor_id}: "
                    f"extracting {len(frames_list)} crops..."
                )

            # --- Extract and save face crops from original frames ---
            crop_frames_dir.mkdir(parents=True, exist_ok=True)
            anchor_dir.mkdir(parents=True, exist_ok=True)
            anchor_paths: list[Path] = []

            track_entries: list[FaceTrackEntry] = []

            for fi, result in frames_list:
                if result.box is None:
                    continue
                box = result.box
                crop = _extract_face_crop(
                    original_frames[fi],
                    box.x1, box.y1, box.x2, box.y2,
                    options.crop_size, options.crop_padding,
                )
                crop_path = crop_frames_dir / f"crop_{fi:06d}.png"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                track_entries.append(
                    FaceTrackEntry(
                        frame_index=fi,
                        box=FaceBox(
                            x1=int(box.x1),
                            y1=int(box.y1),
                            x2=int(box.x2),
                            y2=int(box.y2),
                            confidence=result.detection_score or 0.0,
                            actor_id=actor_id,
                        ),
                        crop_path=crop_path,
                    )
                )

                # Anchor frames at regular intervals
                if fi % options.anchor_interval == 0:
                    anchor_path = anchor_dir / f"anchor_{fi:06d}.png"
                    cv2.imwrite(str(anchor_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    anchor_paths.append(anchor_path)

            if not track_entries:
                if reporter:
                    reporter.message(
                        f"[yellow]WARN[/yellow] FaceFix scene {scene_number}/"
                        f"{actor_id}: no tracks found, skipping"
                    )
                continue

            # --- Encode crop MP4 ---
            crop_mp4 = facefix_dir / "face_crop.mp4"
            if not crop_mp4.exists():
                _encode_crop_mp4(crop_frames_dir, crop_mp4, source, options.ffmpeg_path)

            # --- Render through ComfyUI ---
            if reporter:
                reporter.message(
                    f"FaceFix scene {scene_number}/{actor_id}: rendering..."
                )

            crop_backend.render_scene(
                scene_number=scene_number,
                face_crop_mp4=crop_mp4,
                anchors_dir=anchor_dir,
                output_dir=facefix_dir,
                actor_id=actor_id,
            )

            # --- Load repaired frames ---
            repaired_video = facefix_dir / f"repaired_{actor_id}.mp4"
            if not repaired_video.exists():
                repaired_video = facefix_dir / "raw_facefix_crop.mp4"

            if repaired_video.exists():
                repaired_frames = _load_video_frames(repaired_video)
                if repaired_frames is not None:
                    repaired_dir.mkdir(parents=True, exist_ok=True)
                    for i, frame in enumerate(repaired_frames):
                        cv2.imwrite(
                            str(repaired_dir / f"repaired_{i:06d}.png"), frame
                        )

                face_repairs_for_composite.append(
                    FaceRepairData(
                        actor_id=actor_id,
                        repaired_frames_dir=repaired_dir,
                        track_entries=track_entries,
                        crop_size=options.crop_size,
                    )
                )

        # -- Composite and save (unchanged) --
        if face_repairs_for_composite:
            if reporter:
                reporter.message(
                    f"FaceFix scene {scene_number}: compositing "
                    f"{len(face_repairs_for_composite)} actor(s)..."
                )

            composite_result = compositor.composite(
                face_repairs=face_repairs_for_composite,
                original_frames=current_frames,
                output_dir=scene_dir / "facefix",
            )

            _save_video_frames(
                composite_result.composited_frames,
                final_facefix,
                source,
                options.ffmpeg_path,
            )
            results.append(final_facefix)

            if reporter:
                reporter.message(
                    f"[green]OK[/green] FaceFix scene {scene_number}: {final_facefix}"
                )
        else:
            import shutil

            shutil.copy2(source, final_facefix)
            results.append(final_facefix)

    return results


def _extract_face_crop(
    frame: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    crop_size: int, padding: float,
) -> np.ndarray:
    """Extract a padded, square face crop from an RGB frame."""
    h, w = frame.shape[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1)
    size += int(size * padding * 2)

    nx1 = int(cx - size / 2)
    ny1 = int(cy - size / 2)
    nx2 = nx1 + size
    ny2 = ny1 + size

    ox = max(0, -nx1)
    oy = max(0, -ny1)
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    crop = frame[ny1:ny2, nx1:nx2]
    ch, cw = crop.shape[:2]

    if ch < size or cw < size:
        pad_h = max(0, size - ch)
        pad_w = max(0, size - cw)
        crop = np.pad(
            crop,
            ((oy, pad_h - oy), (ox, pad_w - ox), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return cv2.resize(crop, (crop_size, crop_size))


def _encode_crop_mp4(
    frames_folder: Path,
    output_path: Path,
    reference_video: Path,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Encode a directory of crop PNGs into an MP4 using ffmpeg."""
    import subprocess

    cap = cv2.VideoCapture(str(reference_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    subprocess.run(
        [
            ffmpeg_path,
            "-r", str(fps),
            "-i", str(frames_folder / "crop_%06d.png"),
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
