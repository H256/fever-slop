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
    shots: int = 6,
    frames: int = 124,
    framing: str = "full body, generous margin",
    shot_seconds: float = 0.75,
    scared_shot: bool = True,
    backdrop: str = "plain seamless neutral grey studio backdrop",
    visual_style: str = "Cinematic, live-action",
) -> H3SheetPrompt:
    duration = float(frames) / 24.0
    total_shots = 6 if scared_shot else min(5, max(1, shots))
    step = max(0.25, min(float(shot_seconds), duration / total_shots))
    at = [f"{int(i * step // 60):02d}:{i * step % 60:06.3f}" for i in range(total_shots)]
    subject = description.strip().rstrip(".") or "the character"
    style = visual_style.strip().rstrip(".") or "Cinematic, live-action"
    set_dressing = backdrop.strip().rstrip(".") or "plain seamless neutral grey studio backdrop"
    set_dressing = set_dressing[0].upper() + set_dressing[1:]
    margin = (
        "Every full-body shot is framed with generous empty margin on every side; "
        "the whole figure and anything extending beyond the body stays fully inside the frame."
        if framing == "full body, generous margin"
        else "Every full-body shot is tightly framed but never crops the figure."
    )
    scared = (
        f" [Shot 6] At {at[5]}, the shot cuts to a medium close-up of the face and shoulders, "
        "still facing the camera, the expression now frightened: eyes wide and brows raised and "
        "drawn together, mouth slightly open. Only the expression changes; the figure does not "
        "flinch, recoil, turn away or move."
        if scared_shot and total_shots == 6 else ""
    )
    prompt = (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        "integrated_multimodal_description: "
        "Six distinct locked-off views joined by hard cuts. "
        f"[Shot 1] {style}, a full-body shot frames {subject}, the entire figure visible from head to toe, "
        f"facing the camera. {margin} The figure stands still in a neutral upright pose with arms relaxed "
        "at the sides and does not move, walk, gesture or turn at any point, and holds a neutral expression "
        "until the final shot. [Shot 2] At "
        f"{at[1] if len(at) > 1 else '00:00.750'}, the shot cuts to a tight close-up of the face, framed from "
        "the top of the hair to the chin, still facing the camera, the mouth closed and a still neutral expression. "
        "[Shot 3] At "
        f"{at[2] if len(at) > 2 else '00:01.500'}, the shot cuts to a left side profile of the full figure, the "
        "whole body visible from head to toe with the same framing margin, the head turned to show the left profile, "
        "the mouth closed and a still neutral expression. [Shot 4] At "
        f"{at[3] if len(at) > 3 else '00:02.250'}, the shot cuts to a right side profile of the full figure, the "
        "whole body visible from head to toe with the same framing margin, the head turned to show the right profile, "
        "the mouth closed and a still neutral expression. [Shot 5] At "
        f"{at[4] if len(at) > 4 else '00:03.000'}, the shot cuts to a rear view of the full figure, the back "
        "of the body visible from head to toe with the same framing margin, the mouth closed and a still neutral expression."
        f"{scared} {set_dressing}, even neutral lighting from every side, no props and no cast shadows. "
        "Clothing, hair, colours, proportions and every visible detail remain identical across every shot.\n\n"
        "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
    )
    return H3SheetPrompt(prompt, total_shots, frames)


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
