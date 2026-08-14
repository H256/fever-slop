from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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

    @staticmethod
    def _json(value: BaseModel | list[BaseModel]) -> str:
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else [
            item.model_dump(mode="json") for item in value
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _plan(self, request: VideoPromptRequest, refs: list[ResolvedReference]) -> ResolvedPromptPlan:
        prediction = self.planner(
            mode=request.mode.value, user_prompt=request.user_prompt, duration_seconds=request.duration_seconds,
            references_json=self._json(refs), notes=request.notes or "", strict_fidelity=request.strict_fidelity,
            requested_music_intent=request.music_intent.value if request.music_intent else "",
            relay_segments_json=json.dumps(request.relay_segments, ensure_ascii=False),
        )
        plan = prediction.plan
        if request.music_intent is not None:
            plan.music_intent = request.music_intent
        if plan.music_intent == MusicIntent.NONE:
            plan.non_diegetic_music = None
        allowed = {ref.label for ref in refs}
        for subject in plan.subjects:
            if any(label not in allowed for label in subject.source_references):
                raise ValueError("Planner invented an unknown reference")
        for usage in plan.reference_usage:
            if usage.reference_label not in allowed:
                raise ValueError("Planner invented an unknown reference")
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
            plan_json = self._json(plan)
            references_json = self._json(refs)
            if request.mode == PromptMode.R2V:
                output = self.reference_renderer(
                    guide=self._read(self.reference_guide_path), user_prompt=request.user_prompt,
                    plan_json=plan_json, references_json=references_json,
                    notes=request.notes or "",
                    strict_fidelity=request.strict_fidelity, music_intent=plan.music_intent.value,
                    relay_segments_json=json.dumps(request.relay_segments, ensure_ascii=False),
                )
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
                    user_prompt=request.user_prompt, plan_json=plan_json,
                    references_json=references_json, notes=request.notes or "",
                    strict_fidelity=request.strict_fidelity,
                    music_intent=plan.music_intent.value,
                    relay_segments_json=json.dumps(request.relay_segments, ensure_ascii=False),
                )
                prompt = output.result
        if plan.music_intent == MusicIntent.NONE:
            prompt.non_diegetic_music = None
        return GeneratedVideoPrompt(mode=request.mode, prompt=prompt, plan=plan, references=refs)
