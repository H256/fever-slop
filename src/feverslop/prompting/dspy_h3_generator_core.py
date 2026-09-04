from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Callable
from pathlib import Path
from typing import Any

from feverslop.domain.continuation_intent import ContinuationIntent
from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer
from feverslop.prompting.creative_field_repair import repair_creative_payloads
from feverslop.prompting.deterministic_h3_compiler import creative_shots_from_plan
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt,
    CreativeFieldIssue,
    GeneratedVideoPrompt,
    ImageAnalysisMode,
    MusicIntent,
    PlannedShot,
    PromptJudgeResult,
    PromptMode,
    ReferenceAsset,
    ReferenceKind,
    ReferenceLimits,
    ReferenceRole,
    ReferenceUsage,
    ReferenceVideoPrompt,
    ResolvedPromptPlan,
    ResolvedReference,
    SubjectDefinition,
    VideoPromptRequest,
)
from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.guide_loader import load_markdown_guide

logger = logging.getLogger(__name__)

_H3_JUDGE_MAX_TOKENS = 8192


def _subjects_from_references(refs: list[ResolvedReference]) -> list[SubjectDefinition]:
    subjects = []
    used_names: set[str] = set()
    for ref in refs:
        if ref.kind is not ReferenceKind.PICTURE or ref.role not in {
            ReferenceRole.SUBJECT,
            ReferenceRole.ENVIRONMENT,
        }:
            continue
        base_name = str(ref.name or ref.label.strip("<>")).strip()
        name = base_name
        suffix = 2
        while name.casefold() in used_names:
            name = f"{base_name} {suffix}"
            suffix += 1
        used_names.add(name.casefold())
        subjects.append(SubjectDefinition(
            label=f"<Subject {len(subjects) + 1}>",
            name=name,
            description=ref.description,
            source_references=[ref.label],
        ))
    return subjects


def _authoritative_shot_windows(
    request: VideoPromptRequest,
    shot_count: int,
) -> list[tuple[float | None, float | None, bool]]:
    if shot_count == 0:
        return []
    relay = list(request.relay_segments)
    if relay:
        if len(relay) != shot_count:
            raise ValueError(
                "creative plan must contain exactly one shot per authoritative relay segment",
            )
        windows = []
        for item in relay:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
            if start < 0 or end <= start:
                raise ValueError("authoritative relay segment has an invalid time window")
            windows.append((start, end, bool(item.get("hard_cut_after", False))))
        return windows
    duration = float(request.duration_seconds or 0.0)
    if duration <= 0:
        return [(None, None, False) for _ in range(shot_count)]
    step = duration / shot_count
    return [
        (index * step, (index + 1) * step, False)
        for index in range(shot_count)
    ]


def _authoritative_continuation_intents(
    request: VideoPromptRequest,
) -> list[ContinuationIntent]:
    intents = []
    for item in request.relay_segments:
        action_id = str(item.get("action_id") or "").strip()
        if not action_id:
            continue
        intents.append(ContinuationIntent(
            action_id=action_id,
            requires_continuation=bool(item.get("requires_continuation", False)),
            rationale=str(item.get("continuation_rationale") or "").strip(),
            desired_duration_seconds=item.get("desired_duration_seconds"),
        ))
    return intents

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


def _sanitize_creative_repair(value: Any) -> str:
    """Remove compiler-owned syntax from one LLM repair candidate field."""
    text = str(value or "")
    text = re.sub(
        r"<(Subject|Picture|Video|Audio)\s+\d+>",
        lambda match: {
            "subject": "the referenced subject",
            "picture": "the reference image",
            "video": "the reference video",
            "audio": "the reference audio",
        }[match.group(1).casefold()],
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{2}:\d{2}(?:[:.]\d{2,3})\b", "", text)
    return re.sub(r"\s+", " ", text).strip(" ,;:.\n\t")


def _normalize_judge_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    raw_verdict = str(payload.get("verdict") or "").strip().lower()
    if raw_verdict in {"good", "pass", "passed", "accept", "accepted"}:
        payload["verdict"] = "good"
    elif raw_verdict in {
        "bad", "fail", "failed", "reject", "rejected", "pass_with_minor_issues",
    }:
        payload["verdict"] = "bad"
    elif not raw_verdict:
        # Some local models return only observations and an empty issue list.
        # Treat that as a pass; any reported issue remains fail-closed.
        payload["verdict"] = "bad" if payload.get("issues") or payload.get("field_issues") else "good"
    else:
        payload["verdict"] = "bad"
    return payload


def _judge_guide(guide: str, mode: PromptMode) -> str:
    """Keep judge context to normative rules, not the guide's long example."""
    marker = "\n## 7. Complete Example" if mode is PromptMode.R2V else "\n## 5. Cases"
    return guide.split(marker, 1)[0].rstrip() if marker in guide else guide


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
        # A judge is an advisory review, never a generation control loop.
        self.judge_attempts = 1
        self.prompt_judge_blocking = False
        self.warning_callback = warning_callback
        self.lm = self.dspy_runtime.make_lm(llm)
        self.judge_lm = self.dspy_runtime.make_lm(
            llm,
            max_tokens=int(getattr(llm, "prompt_judge_max_tokens", _H3_JUDGE_MAX_TOKENS)),
        )

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
        prompt: BaseVideoPrompt | ReferenceVideoPrompt | str,
    ) -> PromptJudgeResult | None:
        if self.judge is None:
            return None
        guide_path = self.reference_guide_path if request.mode is PromptMode.R2V else self.base_guide_path
        try:
            guide = _judge_guide(self._read(guide_path), request.mode)
            output = self.judge(
                guide=guide,
                final_prompt=prompt if isinstance(prompt, str) else prompt.render(),
                authoritative_plan=json.dumps(
                    plan.model_dump(mode="json", exclude_none=True),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                references=references,
            )
            value = _normalize_judge_payload(getattr(output, "judge", output))
            try:
                return PromptJudgeResult.model_validate(value)
            except Exception:
                if not isinstance(value, Mapping):
                    raise
                repairable = []
                section_feedback = []
                for issue in value.get("field_issues", ()) or ():
                    try:
                        repairable.append(CreativeFieldIssue.model_validate(issue).model_dump())
                    except Exception:
                        if isinstance(issue, Mapping):
                            section_feedback.append(
                                f"{issue.get('field', 'section')}: "
                                f"{issue.get('repair_instruction', 'review this section')}"
                            )
                repaired = dict(value)
                repaired["field_issues"] = repairable
                repaired["issues"] = [
                    *[str(item) for item in (value.get("issues", ()) or ())],
                    *section_feedback,
                ]
                return PromptJudgeResult.model_validate(repaired)
        except Exception as exc:
            self._warning(f"H3 final prompt judge unavailable: {exc}")
            return PromptJudgeResult(
                verdict="bad",
                issues=[f"judge unavailable: {type(exc).__name__}: {exc}"],
            )

    def judge_compiled_prompt(
        self,
        *,
        request: dict[str, Any],
        plan: ResolvedPromptPlan,
        references: list[ResolvedReference],
        final_prompt: str,
    ) -> PromptJudgeResult | None:
        """Judge the exact deterministic prompt sent to the video backend."""
        if self.judge is None:
            return None
        resolved_references = [
            reference
            if isinstance(reference, ResolvedReference)
            else ResolvedReference.model_validate(reference)
            for reference in references
        ]
        request_model = VideoPromptRequest.model_validate({
            "mode": request["mode"],
            "user_prompt": request["user_prompt"],
            "duration_seconds": request.get("duration_seconds"),
            "references": [
                reference.model_dump() for reference in resolved_references
            ],
            "strict_fidelity": bool(request.get("strict_fidelity", True)),
        })
        with self.dspy_runtime.context(lm=getattr(self, "judge_lm", self.lm)):
            return self._judge_final_prompt(
                request_model,
                plan,
                resolved_references,
                final_prompt,
            )

    def repair_compiled_plan(
        self,
        *,
        request: dict[str, Any],
        plan: ResolvedPromptPlan,
        references: list[ResolvedReference],
        field_issues: list[CreativeFieldIssue],
    ) -> ResolvedPromptPlan:
        """Repair only judge-addressed creative fields in a compiled-plan retry."""
        resolved_references = [
            reference
            if isinstance(reference, ResolvedReference)
            else ResolvedReference.model_validate(reference)
            for reference in references
        ]
        request_model = VideoPromptRequest.model_validate({
            **request,
            "references": [
                reference.model_dump() for reference in resolved_references
            ],
        })
        with self.dspy_runtime.context(lm=self.lm):
            return self._repair_creative_plan(
                request_model,
                plan,
                resolved_references,
                field_issues,
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
        prediction = self.planner(
            mode=request.mode.value,
            user_prompt=request.user_prompt,
            duration_seconds=request.duration_seconds,
            references=refs,
            notes=request.notes or "",
            strict_fidelity=request.strict_fidelity,
            requested_music_intent=request.music_intent.value if request.music_intent else "",
            relay_segments=request.relay_segments,
        )
        creative = prediction.plan
        music_intent = request.music_intent or creative.music_intent
        subjects = _subjects_from_references(refs)
        reference_usage = [
            ReferenceUsage(
                reference_label=ref.label,
                purpose=ref.role.value.replace("_", " "),
                details=ref.description,
            )
            for ref in refs
            if not (
                ref.kind is ReferenceKind.PICTURE
                and ref.role in {ReferenceRole.SUBJECT, ReferenceRole.ENVIRONMENT}
            )
        ]
        subject_names = [subject.name for subject in subjects]
        authored_shots = list(creative.shots)
        authoritative_count = len(request.relay_segments) or 1
        if len(authored_shots) != authoritative_count:
            raise ValueError(
                "creative plan shot count does not match authoritative scene structure",
            )
        windows = _authoritative_shot_windows(request, len(authored_shots))
        shots = []
        for index, authored in enumerate(authored_shots):
            labels = [
                ref.label
                for ref in refs
                if (
                    ref.role is ReferenceRole.FIRST_FRAME and index == 0
                    or ref.role is ReferenceRole.LAST_FRAME and index == len(authored_shots) - 1
                    or ref.role not in {ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME}
                )
            ]
            start, end, hard_cut_after = windows[index]
            shots.append(PlannedShot.model_validate({
                "shot_number": index + 1,
                "start_seconds": start,
                "end_seconds": end,
                "description": authored.description,
                "prose_owner": authored.prose_owner,
                "visible_action": authored.visible_action,
                "performance": authored.performance,
                "camera_behavior": authored.camera_behavior,
                "environmental_motion": authored.environmental_motion,
                "transition_intent": authored.transition_intent,
                "involved_subjects": subject_names,
                "reference_labels": labels,
                "hard_cut_after": hard_cut_after,
            }))
        return ResolvedPromptPlan(
            creative_intent=creative.creative_intent,
            style_opening=creative.style_opening,
            subjects=subjects,
            reference_usage=reference_usage,
            shots=shots,
            overall_soundscape=creative.overall_soundscape,
            music_intent=music_intent,
            non_diegetic_music=(
                None if music_intent is MusicIntent.NONE else creative.non_diegetic_music
            ),
            alignment_instruction=None,
            continuation_intents=_authoritative_continuation_intents(request),
        )

    def _render_reference(
        self,
        request: VideoPromptRequest,
        plan: ResolvedPromptPlan,
        refs: list[ResolvedReference],
    ) -> Any:
        """Render once within the resolved slot contract; compiler owns recovery."""
        allowed_subjects = {subject.label for subject in plan.subjects}
        allowed_references = {reference.label for reference in refs}
        notes = request.notes or ""
        for attempt in range(1, 2):
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
            if attempt == 1:
                self._warning(
                    "H3 renderer contract warning; continuing with deterministic recovery: "
                    f"{error}",
                )
                return output
        raise AssertionError("unreachable")

    def _repair_creative_plan(
        self,
        request: VideoPromptRequest,
        plan: ResolvedPromptPlan,
        refs: list[ResolvedReference],
        field_issues: list,
    ) -> ResolvedPromptPlan:
        """Ask the planner for candidates, then apply only addressed fields."""
        if not field_issues:
            return plan
        issue_notes = "; ".join(
            f"{issue.shot_id}.{issue.field} ({issue.issue_code}): {issue.repair_instruction}"
            for issue in field_issues
        )
        try:
            prediction = self.planner(
                mode=request.mode.value,
                user_prompt=request.user_prompt,
                duration_seconds=request.duration_seconds,
                references=refs,
                notes=(
                    f"{request.notes or ''}\n\nRepair only these creative fields: {issue_notes}. "
                    "Return the same plan structure and preserve all other fields."
                ).strip(),
                strict_fidelity=request.strict_fidelity,
                requested_music_intent=request.music_intent.value if request.music_intent else "",
                relay_segments=request.relay_segments,
            )
            candidate_plan = prediction.plan
            current_payloads = creative_shots_from_plan(plan)
            candidates = {
                f"shot-{index:04d}": shot
                for index, shot in enumerate(candidate_plan.shots, start=1)
            }
            replacements = {}
            for issue in field_issues:
                payload = candidates.get(issue.shot_id)
                value = None if payload is None else _sanitize_creative_repair(
                    getattr(payload, issue.field, None),
                )
                if value is not None and str(value).strip():
                    replacements[(issue.shot_id, issue.field)] = str(value).strip()
            repaired_payloads = repair_creative_payloads(
                list(current_payloads), field_issues, replacements,
            )
            repaired_by_id = {payload.shot_id: payload for payload in repaired_payloads}
            shots = []
            for shot in plan.shots:
                shot_id = f"shot-{int(shot.shot_number):04d}"
                payload = repaired_by_id[shot_id]
                updates = {
                    issue.field: getattr(payload, issue.field)
                    for issue in field_issues
                    if issue.shot_id == shot_id
                }
                shots.append(shot.model_copy(update=updates))
            return plan.model_copy(update={
                "shots": shots,
            })
        except Exception as exc:
            self._warning(f"H3 creative field repair unavailable: {exc}")
            return plan

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
            for _attempt in range(1, self.judge_attempts + 1):
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
                        audio_subject_bindings=request.audio_subject_bindings,
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
                # The judgement is returned with the prompt as a user-facing
                # suggestion.  It must not cause retries or rewrite creative work.
                break
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
