"""Guards for text corruption introduced while decoding model responses."""

from __future__ import annotations


def replacement_character_positions(text: str) -> tuple[int, ...]:
    """Return zero-based positions of UTF-8 replacement characters in *text*."""
    return tuple(index for index, character in enumerate(text) if character == "\ufffd")


def ensure_no_replacement_character(text: str) -> None:
    """Raise an actionable error when a decoded response contains U+FFFD."""
    positions = replacement_character_positions(text)
    if positions:
        position = positions[0]
        suffix = "" if len(positions) == 1 else f" ({len(positions)} occurrences)"
        raise ValueError(
            f"prompt contains UTF-8 replacement character (U+FFFD) at position {position}{suffix}; "
            "check the response decoding",
        )
