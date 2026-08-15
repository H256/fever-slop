"""Backend-neutral prompts for multi-view reference sheets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H3SheetPrompt:
    prompt: str
    shots: int
    frames: int
    rotation_degrees: int = 0


def _fit_rotation(rotation: str, take_seconds: float) -> tuple[int, str]:
    ceiling = max(1.0, float(take_seconds)) * 40.0
    requested = {"quarter": 90, "half": 180, "full": 360}.get(rotation, None)
    degrees = min(requested or max((value for value in (90, 180, 360) if value <= ceiling), default=90), ceiling)
    degrees = int(degrees // 10 * 10)
    return degrees, f"a {degrees}-degree turn" if degrees < 360 else "a complete 360-degree turn"


def build_h3_character_prompt(
    description: str,
    *,
    shots: int = 5,
    frames: int = 124,
    framing: str = "full body, generous margin",
) -> H3SheetPrompt:
    duration = float(frames) / 24.0
    per_shot = duration / max(1, shots)
    shot_list = " → ".join(("full-body front", "left profile", "right profile", "rear", "full-body three-quarter")[:shots])
    prompt = (
        f"The same character throughout: {description.strip()}. "
        f"Create a {duration:.1f}-second I2VA reference take with {shots} distinct locked-off shots joined by hard cuts. "
        f"Shot order: {shot_list}. Each shot lasts approximately {per_shot:.2f} seconds. "
        f"Use {framing}; keep the complete figure inside the frame with margin. "
        "Preserve face, hair, clothing, accessories, proportions, colors and silhouette exactly. "
        "Keep the horizon level and the camera static within each shot. No extra characters, text, collage or watermark."
    )
    return H3SheetPrompt(prompt, shots, frames)


def build_h3_location_prompt(
    description: str,
    *,
    shots: int = 5,
    frames: int = 124,
    coverage: str = "cut views",
    rotation: str = "auto",
) -> H3SheetPrompt:
    if coverage == "continuous move":
        degrees, turn = _fit_rotation(rotation, float(frames) / 24.0)
        prompt = (
            f"The same location throughout: {description.strip()}. Continue one continuous, level camera move around the subject, "
            f"making {turn} at a slow constant rate. Stay at the same height and distance; keep vertical lines vertical and the horizon level. "
            "Preserve architecture, materials, props, lighting and layout. No people, text, cuts or new structures."
        )
        return H3SheetPrompt(prompt, 0, frames, degrees)
    prompt = (
        f"The same location throughout: {description.strip()}. Create a {float(frames) / 24.0:.1f}-second reference take with "
        f"{shots} locked-off tripod views joined by hard cuts: front, right side, rear, left side and a wide establishing view. "
        "Keep the camera level and static in every shot. Preserve architecture, materials, props, lighting and layout. "
        "No people, text, watermark or unrelated structures."
    )
    return H3SheetPrompt(prompt, shots, frames)
