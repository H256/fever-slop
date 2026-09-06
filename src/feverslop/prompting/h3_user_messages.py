from __future__ import annotations

from typing import Mapping


def checkpoint_unjudged_message(generated: Mapping[str, object]) -> str:
    """Explain an absent H3 judge verdict without exposing only internal codes."""
    detail = str(generated.get("dspy_error") or "").strip()
    if "h3.audio.missing" in detail:
        return (
            "H3 quality check did not run because prompt preparation could not place every "
            "required audio reference. No action is needed for a generated prompt: resume "
            "the project and FeverSlop will rebuild it. If this scene uses a manual prompt "
            "override, add the required <Audio N> reference to a sound or shot section, or "
            f"clear the override. Technical details: {detail}"
        )
    if "h3.shot.timestamp" in detail:
        return (
            "H3 quality check did not run because the prompt shot cuts no longer match the "
            "scene timeline. No action is needed for a generated prompt: resume the project "
            "and FeverSlop will rebuild it. If this scene uses a manual prompt override, "
            f"update its shot timestamps or clear the override. Technical details: {detail}"
        )
    if detail:
        return (
            "H3 quality check did not run because prompt preparation needs recovery. No action "
            "is needed for a generated prompt: resume the project and FeverSlop will rebuild it. "
            "If this scene uses a manual prompt override, correct it or clear the override. "
            f"Technical details: {detail}"
        )
    return (
        "H3 quality check did not return a verdict. The prompt was saved; resume the project "
        "to run the check again."
    )


def renderer_recovery_message(technical_details: str) -> str:
    """Describe automatic reference binding recovery in user-facing terms."""
    return (
        "The H3 planner returned inconsistent reference labels; FeverSlop is automatically "
        "repairing the generated prompt. No action is needed unless the same warning remains "
        f"after a resume. Technical details: {technical_details}"
    )


def render_reference_contract_message(scene_number: object, technical_details: str) -> str:
    """Describe a render-blocking H3 reference mismatch and the available recovery."""
    return (
        f"Scene {scene_number} cannot be sent to MiniMax because its H3 prompt refers to "
        "subjects or media references that are not defined in subject_definitions or are not "
        "bound to this scene. If you edited a prompt override, use only the defined labels and "
        "bind every supplied Picture, Video, and Audio reference; otherwise clear the override "
        "and resume so FeverSlop can regenerate the prompt. "
        f"Technical details: {technical_details}"
    )
