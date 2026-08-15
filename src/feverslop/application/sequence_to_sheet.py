"""Deterministic sequence frame extraction, selection, and sheet composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from feverslop.application.orbitsheets_logic import select_orbitsheet_frames


def recommended_view_count(kind: Any) -> int:
    """Return the OrbitSheets-compatible default for an asset kind."""
    value = getattr(kind, "value", kind)
    return 6 if str(value) == "character" else 5 if str(value) == "location" else 4


def recommended_sheet_layout(kind: Any) -> tuple[int, tuple[int, int]]:
    """Return a compact layout while preserving the H3 frame aspect ratio."""
    value = getattr(kind, "value", kind)
    if str(value) == "character":
        return 2, (512, 288)
    if str(value) == "location":
        return 3, (512, 288)
    return 2, (512, 288)


@dataclass(frozen=True, slots=True)
class FrameSelectionConfig:
    view_count: int = 4
    sharpness_weight: float = 0.60
    diversity_weight: float = 0.25
    coverage_weight: float = 0.15


def generate_sequence_to_sheet(
    library: Any,
    *,
    kind: Any,
    asset_id: str,
    look_id: str,
    sequence_video: str | Path,
    anchor_image: str | Path | None = None,
    view_count: int | None = None,
    backend: str = "offline",
    profile: str = "sequence_to_sheet_v1",
    subject: str = "the reference subject",
    vision_endpoint: str | None = None,
) -> dict[str, Any]:
    """Extract, select, compose, and publish a neutral sequence-to-sheet result."""
    view_count = recommended_view_count(kind) if view_count is None else view_count
    if type(view_count) is not int or view_count < 1:
        raise ValueError("view_count must be a positive integer")
    sequence_path = Path(sequence_video)
    if not sequence_path.is_file():
        raise FileNotFoundError(f"sequence video not found: {sequence_path}")
    current = library.get(kind, asset_id)
    with tempfile.TemporaryDirectory(prefix="sequence-to-sheet-") as temporary:
        run_dir = Path(temporary)
        extracted_dir = run_dir / "frames"
        candidates = extract_video_frames(
            sequence_path,
            extracted_dir,
            sample_count=max(view_count * 4, view_count),
        )
        selected = select_orbitsheet_frames(
            candidates,
            count=view_count,
            subject=subject,
            vision_endpoint=vision_endpoint,
        )
        sheet = run_dir / "sheet.png"
        columns, panel_size = recommended_sheet_layout(kind)
        compose_contact_sheet(selected, sheet, columns=columns, panel_size=panel_size)
        updated = library.update_look_artifacts(
            kind,
            asset_id,
            look_id,
            anchor_image=anchor_image,
            sequence_video=sequence_path,
            selected_frames=selected,
            sheet_image=sheet,
            provenance={"backend": backend, "profile": profile},
            expected_revision=current.revision,
        )
    return {
        "asset_id": updated.id,
        "kind": updated.kind.value,
        "look_id": look_id,
        "revision": updated.revision,
        "selected_frames": len(selected),
        "sheet_image": f"looks/{look_id}/sheet.png",
        "backend": backend,
        "profile": profile,
    }

    def __post_init__(self) -> None:
        if type(self.view_count) is not int or self.view_count < 1:
            raise ValueError("view_count must be a positive integer")
        weights = (self.sharpness_weight, self.diversity_weight, self.coverage_weight)
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("selection weights must be non-negative and not all zero")


@dataclass(frozen=True, slots=True)
class _FrameFeatures:
    path: Path
    position: int
    sharpness: float
    descriptor: np.ndarray


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    sample_count: int,
) -> tuple[Path, ...]:
    """Extract evenly spaced PNG frames from a video in deterministic order."""
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise ValueError(f"video contains no readable frames: {source}")
        indices = np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int)
        outputs: list[Path] = []
        for output_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                raise ValueError(f"could not read video frame {frame_index} from {source}")
            output_path = target_dir / f"frame_{output_index:04}.png"
            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"could not write extracted frame: {output_path}")
            outputs.append(output_path)
        return tuple(outputs)
    finally:
        capture.release()


def select_frames(
    frame_paths: Iterable[str | Path],
    *,
    config: FrameSelectionConfig,
) -> tuple[Path, ...]:
    """Select sharp, visually diverse, temporally distributed frames."""
    paths = tuple(Path(path) for path in frame_paths)
    if not paths:
        raise ValueError("at least one frame is required")
    if len(set(paths)) != len(paths):
        raise ValueError("frame paths must be unique")

    features = tuple(_read_features(path, position) for position, path in enumerate(paths))
    if len(features) <= config.view_count:
        return tuple(feature.path for feature in features)

    sharpness = np.asarray([feature.sharpness for feature in features], dtype=np.float64)
    sharp_min = float(sharpness.min())
    sharp_span = float(sharpness.max() - sharp_min)
    sharp_normalized = (
        np.ones(len(features), dtype=np.float64)
        if sharp_span <= 1e-12
        else (sharpness - sharp_min) / sharp_span
    )
    selected: list[_FrameFeatures] = []
    for slot in range(config.view_count):
        target_position = slot * (len(features) - 1) / max(config.view_count - 1, 1)
        segment_half_width = max(
            1.0,
            (len(features) - 1) / max(config.view_count - 1, 1) / 2.0,
        )
        ranked: list[tuple[float, int, _FrameFeatures]] = []
        for feature_index, feature in enumerate(features):
            if feature in selected:
                continue
            in_temporal_segment = abs(feature.position - target_position) <= segment_half_width
            if not in_temporal_segment:
                continue
            coverage = 1.0 - abs(feature.position - target_position) / max(len(features) - 1, 1)
            if not selected:
                diversity = 1.0
            else:
                diversity = min(
                    float(np.linalg.norm(feature.descriptor - other.descriptor))
                    for other in selected
                )
                diversity = min(1.0, diversity / 2.0)
            score = (
                config.sharpness_weight * float(sharp_normalized[feature_index])
                + config.diversity_weight * diversity
                + config.coverage_weight * coverage
            )
            ranked.append((score, -feature.position, feature))
        if not ranked:
            # A prior slot may have consumed the only frame in a narrow
            # segment; retain deterministic behavior without losing a view.
            for feature_index, feature in enumerate(features):
                if feature in selected:
                    continue
                coverage = 1.0 - abs(feature.position - target_position) / max(len(features) - 1, 1)
                diversity = 1.0 if not selected else min(
                    1.0,
                    min(float(np.linalg.norm(feature.descriptor - other.descriptor)) for other in selected) / 2.0,
                )
                score = (
                    config.sharpness_weight * float(sharp_normalized[feature_index])
                    + config.diversity_weight * diversity
                    + config.coverage_weight * coverage
                )
                ranked.append((score, -feature.position, feature))
        selected.append(max(ranked, key=lambda item: (item[0], item[1]))[2])

    return tuple(feature.path for feature in sorted(selected, key=lambda item: item.position))


def compose_contact_sheet(
    frame_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    columns: int = 2,
    panel_size: tuple[int, int] = (512, 512),
    include_labels: bool = True,
) -> Path:
    """Compose selected frames into a deterministic PNG contact sheet."""
    paths = tuple(Path(path) for path in frame_paths)
    if not paths:
        raise ValueError("at least one frame is required")
    if type(columns) is not int or columns < 1:
        raise ValueError("columns must be a positive integer")
    panel_width, panel_height = panel_size
    if panel_width < 1 or panel_height < 1:
        raise ValueError("panel dimensions must be positive")

    label_height = 22 if include_labels else 0
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + label_height)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            panel = source.convert("RGB")
            panel.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (panel_width, panel_height), (0, 0, 0))
            cell.paste(panel, ((panel_width - panel.width) // 2, (panel_height - panel.height) // 2))
        x = (index % columns) * panel_width
        y = (index // columns) * (panel_height + label_height)
        sheet.paste(cell, (x, y))
        if include_labels:
            draw.text((x + 6, y + panel_height + 3), f"view {index + 1}", fill=(235, 235, 235))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG")
    return target


def _read_features(path: Path, position: int) -> _FrameFeatures:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read frame image: {path}")
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    descriptor = resized.reshape(-1)
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return _FrameFeatures(path=path, position=position, sharpness=sharpness, descriptor=descriptor)
