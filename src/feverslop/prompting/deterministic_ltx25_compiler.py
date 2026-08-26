from __future__ import annotations

from collections.abc import Mapping, Sequence

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload


class DeterministicLTX25Compiler:
    """Render typed creative fields into stable LTX 2.5 prompt text."""

    def __init__(self, *, max_words: int | None = None) -> None:
        if max_words is not None and (isinstance(max_words, bool) or max_words <= 0):
            raise ValueError("max_words must be positive or None")
        self.max_words = max_words

    def compile(
        self,
        *,
        facts: LockedSceneFacts,
        shots: Sequence[CreativeShotPayload],
        shot_windows: Mapping[str, tuple[float, float]],
        mode: str = "i2v",
    ) -> str:
        if not isinstance(facts, LockedSceneFacts):
            raise TypeError("facts must be LockedSceneFacts")
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"t2v", "i2v", "r2v", "msr", "ingredients"}:
            raise ValueError("mode must be a supported LTX 2.5 mode")
        by_id = {shot.shot_id: shot for shot in shots}
        if len(by_id) != len(tuple(shots)):
            raise ValueError("duplicate shot ID")
        lines = [f"LTX 2.5 {normalized_mode.upper()} PROMPT", f"Scene: {facts.scene_id}"]
        lines.extend(f"Locked fact {fact.category}/{fact.key}: {fact.value}" for fact in facts.facts)
        for index, shot_id in enumerate(sorted(by_id), start=1):
            if shot_id not in shot_windows:
                raise ValueError(f"missing timing window for shot: {shot_id}")
            start, end = shot_windows[shot_id]
            if float(start) < 0 or float(end) <= float(start):
                raise ValueError(f"invalid timing window for shot: {shot_id}")
            shot = by_id[shot_id]
            lines.append(f"[Shot {index} | {float(start):.3f}-{float(end):.3f}s]")
            lines.append(f"Action: {shot.visible_action.strip()}")
            lines.append(f"Performance: {shot.performance.strip()}")
            if shot.camera_behavior:
                lines.append(f"Camera: {shot.camera_behavior.strip()}")
            if shot.environmental_motion:
                lines.append(f"Environment: {shot.environmental_motion.strip()}")
            if shot.transition_intent:
                lines.append(f"Transition: {shot.transition_intent.strip()}")
        result = "\n".join(lines)
        if self.max_words is not None and len(result.split()) > self.max_words:
            raise ValueError(f"compiled prompt exceeds word budget ({self.max_words})")
        return result
