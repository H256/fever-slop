"""Shared construction of verified last-frame continuity manifests."""

from __future__ import annotations

from pathlib import Path

from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.continuity import BoundaryFrameManifest

LAST_FRAME_EXTRACTOR_REVISION = "last-frame-v1"


def stored_continuity_path(path: Path, project_dir: Path | None) -> str:
    """Return an artifact path in the representation used by manifests."""
    resolved = Path(path).resolve()
    if project_dir is not None:
        project = Path(project_dir).resolve()
        if resolved.is_relative_to(project):
            return resolved.relative_to(project).as_posix()
    return resolved.as_posix()


def build_boundary_frame_manifest(
    source_clip: Path,
    frame_path: Path,
    frame_index: int,
    *,
    project_dir: Path | None,
) -> BoundaryFrameManifest:
    """Build the common verified manifest for an extracted predecessor frame."""
    return BoundaryFrameManifest.create(
        source_clip_path=stored_continuity_path(source_clip, project_dir),
        source_clip_sha256=sha256_file(source_clip),
        frame_index=int(frame_index or 0),
        extractor_revision=LAST_FRAME_EXTRACTOR_REVISION,
        frame_path=stored_continuity_path(frame_path, project_dir),
        frame_sha256=sha256_file(frame_path),
    )
