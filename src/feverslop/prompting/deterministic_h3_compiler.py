from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload


class DeterministicH3Compiler:
    """Compile structured facts and creative fields without model or I/O side effects."""

    def __init__(self, *, max_words: int | None = None) -> None:
        if max_words is not None and (isinstance(max_words, bool) or max_words <= 0):
            raise ValueError("max_words must be positive or None")
        self.max_words = max_words

    def compile(
        self,
        *,
        mode: str,
        facts: LockedSceneFacts,
        shots: Sequence[CreativeShotPayload],
        shot_windows: Mapping[str, tuple[float, float]],
        references: Mapping[str, Sequence[str]] | None = None,
    ) -> str:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"base", "reference"}:
            raise ValueError("mode must be base or reference")
        if not isinstance(facts, LockedSceneFacts):
            raise TypeError("facts must be LockedSceneFacts")
        by_id: dict[str, CreativeShotPayload] = {}
        for shot in shots:
            if not isinstance(shot, CreativeShotPayload):
                raise TypeError("shots must contain CreativeShotPayload values")
            if shot.shot_id in by_id:
                raise ValueError(f"duplicate shot ID: {shot.shot_id}")
            if shot.shot_id not in shot_windows:
                raise ValueError(f"missing timing window for shot: {shot.shot_id}")
            by_id[shot.shot_id] = shot

        lines = ["BASE PROMPT" if normalized_mode == "base" else "FULL REFERENCE PROMPT", f"Scene: {facts.scene_id}"]
        if facts.facts:
            lines.append("Locked facts:")
            lines.extend(f"- {fact.category}/{fact.key}: {fact.value}" for fact in facts.facts)
        for index, shot_id in enumerate(sorted(by_id), start=1):
            start, end = shot_windows[shot_id]
            if float(start) < 0 or float(end) <= float(start):
                raise ValueError(f"invalid timing window for shot: {shot_id}")
            shot = by_id[shot_id]
            lines.append(f"[Shot {index} | {_time(start)}-{_time(end)}]")
            lines.append(f"Action: {shot.visible_action.strip()}")
            lines.append(f"Performance: {shot.performance.strip()}")
            if shot.camera_behavior:
                lines.append(f"Camera: {shot.camera_behavior.strip()}")
            if shot.environmental_motion:
                lines.append(f"Environment motion: {shot.environmental_motion.strip()}")
            if shot.transition_intent:
                lines.append(f"Transition: {shot.transition_intent.strip()}")
            labels = sorted({str(label).strip() for label in (references or {}).get(shot_id, ()) if str(label).strip()})
            if labels:
                lines.append("References: " + ", ".join(labels))
        result = "\n".join(lines)
        if self.max_words is not None and len(result.split()) > self.max_words:
            raise ValueError(f"compiled prompt exceeds word budget ({self.max_words})")
        return result


def _time(value: Any) -> str:
    seconds = float(value)
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"
