from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.dspy_h3_models import PromptMode
from feverslop.prompting.dspy_h3_models import ResolvedPromptPlan
from feverslop.prompting.prompt_contract_validation import PromptContractError, validate_prompt_contract


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
        plan: ResolvedPromptPlan | None = None,
        facts: LockedSceneFacts,
        shots: Sequence[CreativeShotPayload],
        shot_windows: Mapping[str, tuple[float, float]],
        references: Mapping[str, Sequence[str]] | None = None,
        prepared_reference_labels: Sequence[str] | None = None,
        duration_seconds: float | None = None,
    ) -> str:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"base", "reference", "ref", *(item.value for item in PromptMode)}:
            raise ValueError("mode must be base, reference, or a PromptMode value")
        if plan is not None:
            return self._compile_guide_prompt(
                mode=(
                    PromptMode.R2V
                    if normalized_mode in {"reference", "ref", "r2v"}
                    else PromptMode.T2V
                    if normalized_mode == "base"
                    else PromptMode(normalized_mode)
                ),
                plan=plan,
                facts=facts,
                prepared_reference_labels=prepared_reference_labels,
                duration_seconds=duration_seconds,
            )
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
        issues = validate_prompt_contract(
            result,
            facts=facts,
            shots=tuple(by_id[key] for key in sorted(by_id)),
            shot_windows=shot_windows,
            references=references,
            prepared_reference_labels=prepared_reference_labels,
            duration_seconds=duration_seconds,
        )
        if issues:
            raise PromptContractError(issues)
        return result

    def _compile_guide_prompt(
        self,
        *,
        mode: PromptMode,
        plan: ResolvedPromptPlan,
        facts: LockedSceneFacts,
        prepared_reference_labels: Sequence[str] | None,
        duration_seconds: float | None,
    ) -> str:
        """Serialize LLM-authored fields into the MiniMax guide grammar."""
        if mode is PromptMode.R2V:
            subject_lines = [subject.render() for subject in plan.subjects]
            represented = {
                *[label for subject in plan.subjects for label in subject.source_references],
                *[usage.reference_label for usage in plan.reference_usage],
            }
            for label in prepared_reference_labels or ():
                if label in represented:
                    continue
                subject_lines.append(
                    f"{label} is a prepared reference input used by the target video."
                )
            retention_lines = [
                f"{usage.reference_label}: reference - {usage.details}"
                for usage in plan.reference_usage
            ]
            for label in prepared_reference_labels or ():
                if label in represented:
                    continue
                marker = "reference" if "Audio" in label else "fully_preserved"
                retention_lines.append(f"{label}: {marker} - reference is applied in the target video.")
            detailed = "\n".join(
                f"[Shot {index}] {shot.description}"
                + (" " + " ".join(shot.reference_labels) if shot.reference_labels else "")
                for index, shot in enumerate(plan.shots, start=1)
            )
            sections = [
                "subject_definitions:\n" + "\n".join(subject_lines),
                "summary: [reference generation] " + plan.creative_intent,
                "retention_analysis:\n" + "\n".join(retention_lines),
                "detailed_description: " + detailed,
                "overall_soundscape: " + plan.overall_soundscape,
                "non_diegetic_music: " + (plan.non_diegetic_music or "N/A"),
            ]
        else:
            detailed = "\n".join(
                f"[Shot {index}] {shot.description}"
                + (" " + " ".join(shot.reference_labels) if shot.reference_labels else "")
                for index, shot in enumerate(plan.shots, start=1)
            )
            sections = [
                "integrated_multimodal_description: " + detailed,
                "overall_soundscape: " + plan.overall_soundscape,
                "non_diegetic_music: " + (plan.non_diegetic_music or "N/A"),
            ]
            instruction = plan.alignment_instruction
            duration = float(duration_seconds or 0.0)
            if not instruction and mode is PromptMode.I2V:
                instruction = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
            elif not instruction and mode is PromptMode.FL2V:
                instruction = (
                    "How the reference pictures align with the target video — "
                    "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                    f"Picture 2 (from Shot 1) aligns with the {duration:.2f}-second mark of the target video."
                )
            elif not instruction and mode is PromptMode.L2V:
                final_shot = max((shot.shot_number for shot in plan.shots), default=1)
                instruction = (
                    "How the reference pictures align with the target video — "
                    f"<Picture 1> (from [Shot {final_shot}]) aligns with the {duration:.2f}-second mark of the target video."
                )
            if instruction:
                sections.insert(0, instruction)
        result = "\n\n".join(section.strip() for section in sections)
        if self.max_words is not None and len(result.split()) > self.max_words:
            raise ValueError(f"compiled prompt exceeds word budget ({self.max_words})")
        return result


def creative_shots_from_plan(plan: ResolvedPromptPlan) -> tuple[CreativeShotPayload, ...]:
    """Project DSPy plan shots into backend-neutral creative payloads."""
    if not isinstance(plan, ResolvedPromptPlan):
        raise TypeError("plan must be a ResolvedPromptPlan")
    result: list[CreativeShotPayload] = []
    seen: set[int] = set()
    for shot in plan.shots:
        number = int(shot.shot_number)
        if number in seen:
            raise ValueError(f"duplicate planned shot number: {number}")
        seen.add(number)
        result.append(CreativeShotPayload(
            shot_id=f"shot-{number:04d}",
            visible_action=shot.visible_action or shot.description,
            performance=shot.performance or plan.creative_intent,
            camera_behavior=shot.camera_behavior,
            environmental_motion=shot.environmental_motion,
            transition_intent=shot.transition_intent,
        ))
    return validate_creative_shots_against_plan(plan, result)


def validate_creative_shots_against_plan(
    plan: ResolvedPromptPlan,
    shots: Sequence[CreativeShotPayload],
) -> tuple[CreativeShotPayload, ...]:
    """Validate and order creative payloads against their enclosing plan.

    Shot IDs are derived from the plan's stable shot numbers.  Keeping this
    check at the plan boundary prevents a structurally valid payload from
    smuggling an unrelated shot into deterministic prompt compilation.
    """
    if not isinstance(plan, ResolvedPromptPlan):
        raise TypeError("plan must be a ResolvedPromptPlan")
    expected: list[str] = []
    for planned in plan.shots:
        shot_id = f"shot-{int(planned.shot_number):04d}"
        if shot_id in expected:
            raise ValueError(f"duplicate planned shot ID: {shot_id}")
        expected.append(shot_id)

    by_id: dict[str, CreativeShotPayload] = {}
    for shot in shots:
        if not isinstance(shot, CreativeShotPayload):
            raise TypeError("shots must contain CreativeShotPayload values")
        if shot.shot_id not in expected:
            raise ValueError(f"unknown shot ID: {shot.shot_id}")
        if shot.shot_id in by_id:
            raise ValueError(f"duplicate creative shot ID: {shot.shot_id}")
        by_id[shot.shot_id] = shot

    for shot_id in expected:
        if shot_id not in by_id:
            raise ValueError(f"missing creative shot payload: {shot_id}")
    return tuple(by_id[shot_id] for shot_id in expected)


def _time(value: Any) -> str:
    seconds = float(value)
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"
