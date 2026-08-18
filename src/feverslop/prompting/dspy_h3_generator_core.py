from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import logging
import re
from typing import Any

from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer
from feverslop.prompting.dspy_h3_models import (
    GeneratedVideoPrompt,
    ImageAnalysisMode,
    MusicIntent,
    PromptMode,
    ReferenceAsset,
    ReferenceKind,
    ReferenceLimits,
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
    r"\b(?:sings|singing|lip[- ]sync(?:s|ing)?|mouth\s+(?:moves|moving|opens?))\b",
    re.IGNORECASE,
)
_VOCAL_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|avoid|avoids|avoiding|prohibit|prohibits|forbid|forbids)\b[^.!?;]{0,80}$",
    re.IGNORECASE,
)


def _contains_active_vocal_language(text: str) -> bool:
    for match in _ACTIVE_VOCAL_PATTERN.finditer(text):
        prefix = text[max(0, match.start() - 100):match.start()]
        if _VOCAL_NEGATION_PATTERN.search(prefix):
            continue
        return True
    return False


class VideoPromptGenerator:
    """Integrated DSPy planner, analyzer, and renderer."""

    def __init__(self, *, base_guide_path: str | Path, reference_guide_path: str | Path,
                 llm: Any, image_analysis_mode: ImageAnalysisMode = ImageAnalysisMode.MISSING_ONLY,
                 limits: ReferenceLimits | None = None,
                 dspy_runtime: DspyRuntime | None = None):
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
        self.lm = self.dspy_runtime.make_lm(llm)

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
        visual_labels = {
            ref.label for ref in refs
            if ref.kind is ReferenceKind.PICTURE
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
            if request.music_intent is not None:
                plan.music_intent = request.music_intent
            if plan.music_intent == MusicIntent.NONE:
                plan.non_diegetic_music = None
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
            mapped_visuals = [
                label
                for subject in plan.subjects
                for label in subject.source_references
                if label in visual_labels
            ]
            unmapped_visual = visual_labels - set(mapped_visuals)
            duplicate_visual = {
                label for label in mapped_visuals if mapped_visuals.count(label) > 1
            }
            if not unknown and not unmapped_visual and not duplicate_visual:
                break
            error = "Planner reference contract mismatch: " + "; ".join((
                f"unknown={sorted(unknown)!r}",
                f"unmapped_visual={sorted(unmapped_visual)!r}",
                f"multiply_mapped_visual={sorted(duplicate_visual)!r}",
                f"allowed={sorted(allowed)!r}",
            ))
            if attempt == 3:
                raise ValueError(error)
            logger.warning("H3 planner retry %d/3: %s", attempt + 1, error)
            notes = (
                f"{request.notes or ''}\n\n"
                f"The previous plan was invalid ({error}). Retry using only the exact "
                "reference labels listed in allowed. Map every Picture and Video to exactly "
                "one subject; do not combine, rename, omit, or invent labels."
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
            active_vocal_language = bool(
                fully_instrumental
                and _contains_active_vocal_language(rendered_fields)
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
                raise ValueError(error)
            logger.warning("H3 renderer retry %d/3: %s", attempt + 1, error)
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
            if request.mode == PromptMode.R2V:
                output = self._render_reference(request, plan, refs)
                prompt = ReferenceVideoPrompt(
                    subject_definitions=plan.subjects, summary=output.summary,
                    retention_analysis=output.retention_analysis,
                    detailed_description=output.detailed_description,
                    overall_soundscape=output.overall_soundscape,
                    non_diegetic_music=output.non_diegetic_music,
                )
            else:
                output = self.base_renderer(
                    guide=self._read(self.base_guide_path), mode=request.mode.value,
                    user_prompt=request.user_prompt, plan=plan,
                    references=refs, notes=request.notes or "",
                    strict_fidelity=request.strict_fidelity,
                    music_intent=plan.music_intent.value,
                    relay_segments=request.relay_segments,
                )
                prompt = output.result
        if plan.music_intent == MusicIntent.NONE:
            prompt.non_diegetic_music = None
        return GeneratedVideoPrompt(mode=request.mode, prompt=prompt, plan=plan, references=refs)
