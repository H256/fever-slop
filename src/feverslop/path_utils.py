from __future__ import annotations

import os
from os import PathLike
from pathlib import Path, PureWindowsPath
import warnings


WORKFLOW_PATH_ALIASES = {
    "workflows/audio_song_v2.json": "workflows/audio/audio-model/audio_song_v2.json",
    "workflows/image_detail_easyuse_startframe_v1.json": "workflows/image/image-model/image_detail_easyuse_startframe_v1.json",
    "workflows/image_edit_flux2_klein_1ref_v1.json": "workflows/image/image-model/image_edit_flux2_klein_1ref_v1.json",
    "workflows/image_edit_flux2_klein_2ref_v1.json": "workflows/image/image-model/image_edit_flux2_klein_2ref_v1.json",
    "workflows/image_mask_sam3_actor_regions_v1.json": "workflows/image/image-model/image_mask_sam3_actor_regions_v1.json",
    "workflows/image_repair_sdxl_ipadapter_identity_v1.json": "workflows/image/image-model/image_repair_sdxl_ipadapter_identity_v1.json",
    "workflows/image_t2i_startframe_ideogram_director_v1.json": "workflows/image/image-model/image_t2i_startframe_ideogram_director_v1.json",
    "workflows/image_t2i_startframe_ideogram_v1.json": "workflows/image/image-model/image_t2i_startframe_ideogram_v1.json",
    "workflows/image_t2i_startframe_krea_v1.json": "workflows/image/image-model/image_t2i_startframe_krea_v1.json",
    "workflows/image_t2i_startframe_v1.json": "workflows/image/image-model/image_t2i_startframe_v1.json",
    "workflows/sequence_to_sheet_minimax_h3_i2va_v1.json": "workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json",
}


def resolve_workflow_reference(value: str | PathLike[str]) -> str:
    """Normalize a workflow reference and resolve only explicitly known aliases."""
    raw = os.fspath(value)
    if _is_native_absolute(Path(raw)) or _looks_like_windows_absolute(raw):
        return raw
    normalized = raw.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    replacement = WORKFLOW_PATH_ALIASES.get(normalized.casefold())
    if replacement is None:
        return normalized
    warnings.warn(
        f"Legacy workflow path '{normalized}' is deprecated; use '{replacement}'.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement


def coerce_local_path(
    value: str | PathLike[str],
    *,
    base_dir: str | PathLike[str] | None = None,
    containment_root: str | PathLike[str] | None = None,
) -> Path:
    """Coerce local path strings that may use Windows or POSIX separators.

    When *containment_root* is provided the resolved path must stay under
    that root directory.  Raises :class:`ValueError` on escape attempts
    (e.g. ``../`` traversal).
    """
    raw = os.fspath(value)
    path = Path(raw)
    if _is_native_absolute(path) or _looks_like_windows_absolute(raw):
        coerced = path
    else:
        coerced = Path(raw.replace("\\", "/"))

    if base_dir is not None and not coerced.is_absolute():
        coerced = Path(base_dir) / coerced

    if containment_root is not None:
        resolved = coerced.resolve()
        root = Path(containment_root).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Path escapes containment root: {resolved} is not under {root}",
            )
    return coerced


def _is_native_absolute(path: Path) -> bool:
    return path.is_absolute()


def _looks_like_windows_absolute(value: str) -> bool:
    windows_path = PureWindowsPath(value)
    return bool(windows_path.drive and windows_path.root)
