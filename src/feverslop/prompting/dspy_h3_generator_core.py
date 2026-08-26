from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt,
    GeneratedVideoPrompt,
    ImageAnalysisMode,
    MusicIntent,
    PlannedSubject,
    PromptJudgeResult,
    PromptMode,
    ReferenceAsset,
    ReferenceKind,
    ReferenceLimits,
    ReferenceRole,
    ReferenceVideoPrompt,
    ResolvedPromptPlan,
    ResolvedReference,
    SubjectDefinition,
    VideoPromptRequest,
)
from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.guide_loader import load_markdown_guide

logger = logging.getLogger(__name__)

_ACTIVE_VOCAL_PATTERN = re.compile(
    r"\b(?:sing|sings|singing|lip[- ]sync(?:s|ing)?|mouth\s+(?:moves|moving|opens?))\b",
    re.IGNORECASE,
)
_VOCAL_NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"no|not|never|without|"
    r"does\s+not|doesn't|do\s+not|don't|"
    r"is\s+not|isn't|are\s+not|aren't|"
    r"avoid|avoids|avoiding|prohibit|prohibits|forbid|forbids|"
    r"absent|inactive|closed|still"
    r")\b",
    re.IGNORECASE,
)


_REFERENCE_LABEL_PATTERN = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>")


def _split_reference_labels(value: str) -> list[str]:
    """Return individual H3 reference labels from an LM-serialized value.

    Small instruction models sometimes serialize a list element as one string,
    e.g. "<Picture 2>, <Picture 3>, <Picture 4>".  Treat that as three labels
    rather than as one unknown reference.
    """
    text = str(value or "").strip()
    if not text:
        return []
    labels = _REFERENCE_LABEL_PATTERN.findall(text)
    return labels or [text]


def _normalize_plan_reference_labels(plan: Any) -> None:
    """Normalize LM reference-label serialization before contract validation."""
    for subject in plan.subjects:
        normalized: list[str] = []
        for value in subject.source_references:
            normalized.extend(_split_reference_labels(value))
        subject.source_references = list(dict.fromkeys(normalized))

    normalized_usages = []
    for usage in plan.reference_usage:
        labels = _split_reference_labels(usage.reference_label)
        for label in labels:
            normalized_usages.append(usage.model_copy(update={"reference_label": label}))
    plan.reference_usage = normalized_usages

    for shot in plan.shots:
        normalized: list[str] = []
        for value in shot.reference_labels:
            normalized.extend(_split_reference_labels(value))
        shot.reference_labels = list(dict.fromkeys(normalized))


def _contains_active_vocal_language(text: str) -> bool:
    """Detect an affirmative visible vocal-performance instruction.

    This check is deliberately clause-local. Small instruction models often use
    phrases such as "is not singing", "no lip-sync", or "singing is absent".
    Those are valid instrumental constraints and must not trigger a retry.
    """
    for match in _ACTIVE_VOCAL_PATTERN.finditer(text):
        # Restrict the negation check to the current clause/sentence so an
        # unrelated "not" elsewhere cannot mask a real vocal instruction.
        left_boundary = max(
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        right_candidates = [
            pos for pos in (
                text.find(".", match.end()),
                text.find("!", match.end()),
                text.find("?", match.end()),
                text.find(";", match.end()),
                text.find("\n", match.end()),
            )
            if pos >= 0
        ]
        right_boundary = min(right_candidates) if right_candidates else len(text)
        clause = text[left_boundary + 1:right_boundary]

        if _VOCAL_NEGATION_PATTERN.search(clause):
            continue
        return True
    return False


def _deterministic_reference_definitions(refs: list[ResolvedReference]) -> list[str]:
    """Render structural non-picture reference declarations deterministically.

    The LM may use an <Audio N> anchor correctly in summary/shots/retention while
    omitting the boilerplate declaration from the header.  These declarations are
    pure serialization from already-resolved metadata and must not depend on the LM.
    """
    definitions: list[str] = []
    for ref in refs:
        if ref.kind is ReferenceKind.AUDIO:
            description = (ref.description or "synchronized audio reference").strip()
            definitions.append(f"{ref.label} is the {description} and is reused for the scene.")
    return definitions


class VideoPromptGenerator:
    """Integrated DSPy planner, analyzer, and renderer."""

    def __init__(self, *, base_guide_path: str | Path, reference_guide_path: str | Path,
                 llm: Any, image_analysis_mode: ImageAnalysisMode = ImageAnalysisMode.MISSING_ONLY,
                 limits: ReferenceLimits | None = None,
                 dspy_runtime: DspyRuntime | None = None,
                 warning_callback: Callable[..., None] | None = None):
        self.base_guide_path = Path(str(base_guide_path)).name
        self.reference_guide_path = Path(str(reference_guide_path)).name
        self.limits = limits or ReferenceLimits()
        self.dspy_runtime = dspy_runtime or DspyRuntime.create()
        signatures = self.dspy_runtime.signatures
        self.image_analyzer = LocalImageAnalyzer(
            self.dspy_runtime.predict(signatures.analyze_image),
            image_analysis_mode,
        )
        self.planner = self.dspy_runtime.predict(signatures.build_prompt_plan)
        self.base_renderer = self.dspy_runtime.predict(signatures.render_base_prompt)
        self.reference_renderer = self.dspy_runtime.predict(signatures.render_reference_prompt)
        self.judge = (
            self.dspy_runtime.predict(signatures.judge_final_prompt)
            if getattr(signatures, "judge_final_prompt", None) is not None
            else None
        )
        self.judge_attempts = max(1, int(getattr(llm, "prompt_judge_attempts", 3)))
        self.warning_callback = warning_callback
        self.lm = self.dspy_runtime.make_lm(llm)

    def set_warning_callback(self, callback: Callable[..., None] | None) -> None:
        self.warning_callback = callback

    def _warning(self, message: str, *, title: str = "H3 warning") -> None:
        logger.warning(message)
        callback = getattr(self, "warning_callback", None)
        if callback is not None:
            callback(message, title=title)

    def _judge_final_prompt(
        self,
        request: VideoPromptRequest,
        plan: ResolvedPromptPlan,
        references: list[ResolvedReference],
        prompt: BaseVideoPrompt | ReferenceVideoPrompt,
    ) -> PromptJudgeResult | None:
        if self.judge is None:
            return None
        guide_path = self.reference_guide_path if request.mode is PromptMode.R2V else self.base_guide_path
        try:
            output = self.judge(
                guide=self._read(guide_path),
                final_prompt=prompt.render(),
                authoritative_plan=plan.model_dump_json(indent=2),
                references=references,
            )
            value = getattr(output, "judge", output)
            return PromptJudgeResult.model_validate(value)
        except Exception as exc:
            self._warning(f"H3 final prompt judge unavailable: {exc}")
            return PromptJudgeResult(
                verdict="bad",
                issues=[f"judge unavailable: {type(exc).__name__}: {exc}"],
            )

    @staticmethod
    def _read(path: str | Path) -> str:
        return load_markdown_guide(path)

    @staticmethod
    def _label(kind: ReferenceKind, number: int) -> str:
        return f"<{kind.value.title()} {number}>"

    def _resolve_references(self, refs: list[ReferenceAsset]) -> list[ResolvedReference]:
        counts = defaultdict(int)
        result = []
        for ref in refs:
            counts[ref.kind] += 1
            limit_name = {
                ReferenceKind.PICTURE: "max_pictures",
                ReferenceKind.AUDIO: "max_audio",
                ReferenceKind.VIDEO: "max_videos",
            }[ref.kind]
            if counts[ref.kind] > getattr(self.limits, limit_name):
                raise ValueError(f"Too many {ref.kind.value} references")
            description = ref.description
            if self.image_analyzer.should_analyze(ref):
                description = self.image_analyzer.analyze(ref)
            if not description:
                raise ValueError(f"{self._label(ref.kind, counts[ref.kind])} has no usable description")
            result.append(ResolvedReference(
                label=self._label(ref.kind, counts[ref.kind]), kind=ref.kind, source=ref.source,
                role=ref.role, description=description, name=ref.name, use_audio=ref.use_audio,
            ))
        return result

    def _plan(self, request: VideoPromptRequest, refs: list[ResolvedReference]) -> ResolvedPromptPlan:
        allowed = {ref.label for ref in refs}
        # Only references that semantically define reusable visible content must
        # be mapped to a PlannedSubject. Frame anchors, composition/style refs,
        # camera/motion refs, etc. are valid without becoming subjects.
        subject_reference_labels = {
            ref.label for ref in refs
            if ref.kind is ReferenceKind.PICTURE
            and ref.role in {ReferenceRole.SUBJECT, ReferenceRole.ENVIRONMENT}
        }
        notes = request.notes or ""
        for attempt in range(1, 4):
            prediction = self.planner(
                mode=request.mode.value, user_prompt=request.user_prompt,
                duration_seconds=request.duration_seconds, references=refs, notes=notes,
                strict_fidelity=request.strict_fidelity,
                requested_music_intent=request.music_intent.value if request.music_intent else "",
                relay_segments=request.relay_segments,
            )
            plan = prediction.plan
            _normalize_plan_reference_labels(plan)
            if request.music_intent is not None:
                plan.music_intent = request.music_intent
            if plan.music_intent == MusicIntent.NONE:
                plan.non_diegetic_music = None
            missing_music_description = bool(
                plan.music_intent != MusicIntent.NONE and not plan.non_diegetic_music,
            )
            unknown = {
                label
                for subject in plan.subjects
                for label in subject.source_references
                if label not in allowed
            }
            unknown.update(
                usage.reference_label
                for usage in plan.reference_usage
                if usage.reference_label not in allowed
            )
            unknown.update(
                label
                for shot in plan.shots
                for label in shot.reference_labels
                if label not in allowed
            )
            mapped_subject_references = {
                label
                for subject in plan.subjects
                for label in subject.source_references
                if label in subject_reference_labels
            }
            unmapped_subject_references = subject_reference_labels - mapped_subject_references

            # Subject/environment reference membership is deterministic metadata,
            # not a creative LM decision. If the planner describes a reference in
            # reference_usage/shots but omits the corresponding PlannedSubject,
            # repair that serialization immediately instead of spending another
            # LM call asking it to reproduce information we already know.
            if unmapped_subject_references:
                missing_refs = [
                    ref for ref in refs if ref.label in unmapped_subject_references
                ]
                existing_names = {subject.name.strip().casefold() for subject in plan.subjects}
                for ref in missing_refs:
                    base_name = ref.name or ref.label.strip("<>")
                    name = base_name
                    suffix = 2
                    while name.strip().casefold() in existing_names:
                        name = f"{base_name} {suffix}"
                        suffix += 1
                    existing_names.add(name.strip().casefold())
                    plan.subjects.append(PlannedSubject(
                        name=name,
                        description=ref.description,
                        source_references=[ref.label],
                    ))
                logger.info(
                    "H3 planner normalized required subject mappings from reference metadata: %s",
                    ", ".join(f"{ref.label} -> {ref.name or ref.label}" for ref in missing_refs),
                )
                mapped_subject_references.update(unmapped_subject_references)
                unmapped_subject_references.clear()

            if not unknown and not unmapped_subject_references and not missing_music_description:
                break
            error = "Planner contract mismatch: " + "; ".join((
                f"unknown={sorted(unknown)!r}",
                f"unmapped_subject_references={sorted(unmapped_subject_references)!r}",
                f"missing_music_description={missing_music_description!r}",
                f"music_intent={plan.music_intent.value!r}",
                f"requested_music_intent={(request.music_intent.value if request.music_intent else '')!r}",
                f"allowed={sorted(allowed)!r}",
            ))
            if attempt == 3:
                self._warning(
                    "H3 planner contract warning after final attempt; continuing with result: "
                    f"{error}",
                )
                break
            self._warning(f"H3 planner retry {attempt + 1}/3: {error}", title="H3 planner retry")
            mapping_contract = "; ".join(
                f'{ref.label} -> reusable subject "{ref.name or ref.label.strip("<>")}" '
                f'with role "{ref.role.value}" and description "{ref.description}"'
                for ref in refs
                if ref.label in subject_reference_labels
            )
            non_subject_contract = "; ".join(
                f'{ref.label} -> role "{ref.role.value}" (use according to its role; do not force it into a subject)'
                for ref in refs
                if ref.label not in subject_reference_labels
            )
            notes = (
                f"{request.notes or ''}\n\n"
                f"The previous plan was invalid ({error}). Retry using only the exact "
                "reference labels listed in allowed. Do not rename, omit, or invent labels. "
                "Map each subject/environment picture reference to at least one reusable subject. "
                "Do not force frame, style, composition, camera, motion, temporal, video, or audio references "
                "into subjects; represent them through reference_usage and/or the relevant shots instead. "
                f"Required subject mappings: {mapping_contract or 'none'}. "
                f"Role-only references: {non_subject_contract or 'none'}. "
                + (
                    "The previous plan enabled non-diegetic music but omitted its description. "
                    "If requested_music_intent is 'none', set music_intent='none' and omit non_diegetic_music. "
                    "Otherwise keep the enabled music intent and provide a concrete non_diegetic_music description."
                    if missing_music_description else ""
                )
            ).strip()
        subjects = []
        seen = set()
        for index, subject in enumerate(plan.subjects, 1):
            key = subject.name.strip().casefold()
            if key in seen:
                raise ValueError(f"Planner created duplicate subject name: {subject.name!r}")
            seen.add(key)
            subjects.append(SubjectDefinition(
                label=f"<Subject {index}>", name=subject.name,
                description=subject.description, source_references=subject.source_references,
            ))
        return ResolvedPromptPlan(
            creative_intent=plan.creative_intent, subjects=subjects,
            reference_usage=plan.reference_usage, shots=plan.shots,
            overall_soundscape=plan.overall_soundscape, music_intent=plan.music_intent,
            non_diegetic_music=plan.non_diegetic_music,
            alignment_instruction=plan.alignment_instruction,
        )

    def _render_reference(
        self,
        request: VideoPromptRequest,
        plan: ResolvedPromptPlan,
        refs: list[ResolvedReference],
    ) -> Any:
        """Render within the resolved slot contract, retrying structural drift."""
        allowed_subjects = {subject.label for subject in plan.subjects}
        allowed_references = {reference.label for reference in refs}
        notes = request.notes or ""
        for attempt in range(1, 4):
            output = self.reference_renderer(
                guide=self._read(self.reference_guide_path),
                user_prompt=request.user_prompt,
                plan=plan,
                references=refs,
                notes=notes,
                strict_fidelity=request.strict_fidelity,
                music_intent=plan.music_intent.value,
                relay_segments=request.relay_segments,
            )
            rendered_fields = "\n".join(str(getattr(output, field, "") or "") for field in (
                "summary",
                "detailed_description",
                "overall_soundscape",
                "non_diegetic_music",
            ))
            detailed_description = str(getattr(output, "detailed_description", "") or "")
            used_subjects = set(re.findall(r"<Subject\s+\d+>", rendered_fields))
            used_references = set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", rendered_fields))
            retention = list(getattr(output, "retention_analysis", ()) or ())
            retention_targets = [str(item.target_label) for item in retention]
            undefined_subjects = used_subjects - allowed_subjects
            unknown_references = used_references - allowed_references
            duplicate_retention = {
                label for label in retention_targets if retention_targets.count(label) > 1
            }
            unknown_retention = set(retention_targets) - allowed_subjects - allowed_references
            missing_subject_retention = allowed_subjects - set(retention_targets)
            fully_instrumental = bool(request.relay_segments) and all(
                str(segment.get("state") or "").strip().lower() == "instrumental"
                for segment in request.relay_segments
            )
            # Only visible performance instructions belong in this contract check.
            # Summary/soundscape/music may legitimately mention vocals or singing as
            # source-audio context without instructing the on-screen subject to sing.
            active_vocal_language = bool(
                fully_instrumental
                and _contains_active_vocal_language(detailed_description),
            )
            if not any((
                undefined_subjects,
                unknown_references,
                duplicate_retention,
                unknown_retention,
                missing_subject_retention,
                active_vocal_language,
            )):
                return output
            error = "Renderer reference contract mismatch: " + "; ".join((
                f"undefined_subjects={sorted(undefined_subjects)!r}",
                f"unknown_references={sorted(unknown_references)!r}",
                f"duplicate_retention={sorted(duplicate_retention)!r}",
                f"unknown_retention={sorted(unknown_retention)!r}",
                f"missing_subject_retention={sorted(missing_subject_retention)!r}",
                f"active_vocal_language={active_vocal_language!r}",
            ))
            if attempt == 3:
                self._warning(
                    "H3 renderer contract warning after final attempt; continuing with result: "
                    f"{error}",
                )
                return output
            self._warning(f"H3 renderer retry {attempt + 1}/3: {error}", title="H3 renderer retry")
            subject_map = ", ".join(
                f"{subject.label}={subject.name} from {subject.source_references}"
                for subject in plan.subjects
            )
            notes = (
                f"{request.notes or ''}\n\nThe previous rendered prompt was invalid ({error}). "
                f"Use only these exact subject mappings: {subject_map}. Use only reference "
                f"labels {sorted(allowed_references)!r}. Emit exactly one retention entry per subject."
            ).strip()
        raise AssertionError("unreachable")

    def __call__(self, request_data: dict[str, Any]) -> GeneratedVideoPrompt:
        section_only = bool(request_data.get("_section_only"))
        if request_data.get("mode") == "ref":
            request_data = {**request_data, "mode": PromptMode.R2V.value}
        request = VideoPromptRequest.model_validate(request_data)
        # Planning and optional image analysis are DSPy calls as well.  They
        # must use the same configured LM as the final renderer; otherwise a
        # planner failure is caught by the adapter and looks like a valid
        # legacy/concept prompt.
        with self.dspy_runtime.context(lm=self.lm):
            refs = self._resolve_references(request.references)
            plan = self._plan(request, refs)
            if section_only:
                # The planner is the only generative step in this mode.  The
                # backend prompt is compiled by DspyH3PromptBuilder from the
                # typed plan, so no model-generated prose can alter anchors or
                # locked structure.
                placeholder = BaseVideoPrompt(
                    integrated_multimodal_description=plan.creative_intent,
                    overall_soundscape=plan.overall_soundscape,
                    non_diegetic_music=plan.non_diegetic_music,
                    alignment_instruction=plan.alignment_instruction,
                )
                return GeneratedVideoPrompt(
                    mode=request.mode,
                    prompt=placeholder,
                    plan=plan,
                    references=refs,
                )
            effective_request = request
            prompt = None
            judge = None
            judge_attempts = []
            for attempt in range(1, self.judge_attempts + 1):
                if request.mode == PromptMode.R2V:
                    output = self._render_reference(effective_request, plan, refs)
                    prompt = ReferenceVideoPrompt(
                        subject_definitions=plan.subjects,
                        reference_definitions=_deterministic_reference_definitions(refs),
                        summary=output.summary,
                        retention_analysis=output.retention_analysis,
                        detailed_description=output.detailed_description,
                        overall_soundscape=output.overall_soundscape,
                        non_diegetic_music=output.non_diegetic_music,
                    )
                else:
                    output = self.base_renderer(
                        guide=self._read(self.base_guide_path), mode=request.mode.value,
                        user_prompt=request.user_prompt, plan=plan,
                        references=refs, notes=effective_request.notes or "",
                        strict_fidelity=request.strict_fidelity,
                        music_intent=plan.music_intent.value,
                        relay_segments=request.relay_segments,
                    )
                    prompt = output.result
                if plan.music_intent == MusicIntent.NONE:
                    prompt.non_diegetic_music = None
                judge = self._judge_final_prompt(effective_request, plan, refs, prompt)
                if judge is not None:
                    judge_attempts.append(judge)
                if judge is None or judge.verdict == "good" or attempt == self.judge_attempts:
                    break
                feedback = "; ".join(judge.issues) or "the prompt did not satisfy the supplied guide and plan"
                if judge.repair_instruction:
                    feedback += f" Repair instruction: {judge.repair_instruction}"
                self._warning(
                    f"H3 final prompt judge retry {attempt + 1}/{self.judge_attempts}: {feedback}",
                    title="H3 final prompt judge retry",
                )
                effective_request = effective_request.model_copy(update={
                    "notes": (
                        f"{effective_request.notes or ''}\n\n"
                        "Try again. Here are the judge errors you must fix: "
                        f"{feedback}"
                    ).strip(),
                })
        if judge is not None and judge.verdict == "bad":
            self._warning(
                "H3 final prompt judge rejected prompt: " + "; ".join(judge.issues),
                title="H3 final prompt judge",
            )
        return GeneratedVideoPrompt(
            mode=request.mode,
            prompt=prompt,
            plan=plan,
            references=refs,
            judge=judge,
            judge_attempts=judge_attempts,
        )
