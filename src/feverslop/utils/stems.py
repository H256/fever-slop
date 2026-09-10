from __future__ import annotations

from pathlib import Path

SUPPORTED_STEM_SUFFIXES = frozenset({".wav", ".mp3", ".flac"})


def discover_stem_files(
    stems_dir: str | Path,
    input_audio: str | Path | None = None,
) -> dict[str, Path] | None:
    """Discover every named stem matching the input audio basename."""
    stems_dir = Path(stems_dir)
    if not stems_dir.is_dir():
        return None

    input_stem = Path(input_audio).stem if input_audio is not None else None
    discovered: dict[str, Path] = {}
    for path in sorted(stems_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_STEM_SUFFIXES:
            continue
        stem_name, separator, source_stem = path.stem.partition("_")
        if not separator or (input_stem is not None and source_stem != input_stem):
            continue
        current = discovered.get(stem_name)
        if current is None or path.suffix.lower() == ".wav":
            discovered[stem_name] = path
    return discovered or None
