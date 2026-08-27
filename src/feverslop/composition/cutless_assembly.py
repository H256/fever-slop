from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from feverslop.adapters.cutless_assembly import CutlessAssemblyService
from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.cutless_boundaries import (
    CutlessBoundary,
    CutlessBoundaryDiagnostic,
    build_cutless_assembly_plan,
)


def assemble_declared_cutless_group(
    *,
    group: dict,
    clips_by_segment: dict[str, Path],
    frame_count: Callable[[Path], int],
    extract_first_frame: Callable[[Path, Path], Path],
    extract_last_frame: Callable[[Path, Path], Path],
    assembly_service: CutlessAssemblyService,
    output_file: Path,
    diagnostics_file: Path,
    fps: int,
    duplicate_policy: str = "reject",
) -> Path:
    segments = list(group.get("segments") or [])
    segment_ids = [str(segment.get("segment_id") or "").strip() for segment in segments]
    if not segment_ids or any(not segment_id for segment_id in segment_ids):
        raise ValueError("cutless group segments require stable segment IDs")
    missing = [segment_id for segment_id in segment_ids if segment_id not in clips_by_segment]
    if missing:
        raise FileNotFoundError(
            "cutless group is missing rendered segment clips: " + ", ".join(missing),
        )

    boundaries: list[CutlessBoundary] = []
    diagnostics: list[CutlessBoundaryDiagnostic] = []
    frame_counts: dict[str, int] = {}
    first_hashes: dict[str, str] = {}
    last_hashes: dict[str, str] = {}
    frame_dir = diagnostics_file.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for segment_id in segment_ids:
        clip = clips_by_segment[segment_id]
        frame_counts[segment_id] = int(frame_count(clip))
        first = extract_first_frame(clip, frame_dir / f"{segment_id}.first.png")
        last = extract_last_frame(clip, frame_dir / f"{segment_id}.last.png")
        first_hashes[segment_id] = sha256_file(first)
        last_hashes[segment_id] = sha256_file(last)

    for predecessor, successor in zip(segment_ids, segment_ids[1:]):
        boundaries.append(CutlessBoundary(
            predecessor_segment_id=predecessor,
            successor_segment_id=successor,
            boundary_frame_sha256=last_hashes[predecessor],
        ))
        successor_segment = next(
            segment for segment in segments
            if str(segment.get("segment_id") or "").strip() == successor
        )
        expected_successor_frames = round(float(successor_segment.get("duration_seconds", 0.0)) * fps)
        diagnostics.append(CutlessBoundaryDiagnostic(
            predecessor_segment_id=predecessor,
            successor_segment_id=successor,
            predecessor_last_frame_sha256=last_hashes[predecessor],
            successor_first_frame_sha256=first_hashes[successor],
            similarity=1.0 if last_hashes[predecessor] == first_hashes[successor] else 0.0,
            timing_delta_frames=abs(frame_counts[successor] - expected_successor_frames),
        ))

    plan = build_cutless_assembly_plan(
        segment_ids,
        boundaries,
        diagnostics,
        duplicate_policy=duplicate_policy,
    )
    diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_file.write_text(
        json.dumps({**group, "assembly": {
            "segment_ids": list(plan.segment_ids),
            "trim_first_frame_segments": list(plan.trim_first_frame_segments),
            "outcome": plan.outcome,
            "crossfade": plan.crossfade,
            "diagnostics": [diagnostic.__dict__ for diagnostic in plan.diagnostics],
        }}, indent=2),
        encoding="utf-8",
    )
    return assembly_service.assemble(
        clips_by_segment,
        plan,
        output_file,
        segment_frame_counts=frame_counts,
        fps=fps,
    )
