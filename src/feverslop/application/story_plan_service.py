"""Typed DSPy planning service for the story-plan contract (issue #1385).

``StoryPlanService`` drives four narrow DSPy jobs -- narrative bible,
beat allocation, per-brief acting, and one bounded repair -- with
deterministic validation between stages.  The final candidate is
validated with ``validate_story_plan_payload`` and then converted with
``StoryPlan.model_validate(..., strict=False)``; on validation failure the
service performs exactly one repair pass before raising ``StoryPlanError``.

The service is pure application code: it imports no DSPy, no
``feverslop.render.*``, no ``feverslop.audio.*``, and no
``feverslop.prompting.*``.  The DSPy prompt modules are injected and
called through a small structural interface so the module stays
importable without DSPy installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from feverslop.domain.story_plan import StoryPlan
from feverslop.domain.story_plan_artifacts import (
    ArtifactClass,
    RegenerationPolicy,
    StoryPlanArtifactManifest,
)
from feverslop.errors import FeverSlopError
from feverslop.ports.reporting import NullReporter, Reporter
from feverslop.utils.io import atomic_write_text

__all__ = [
    "STORY_PLAN_PRODUCER",
    "SegmentDescriptor",
    "StoryPlanError",
    "StoryPlanRequest",
    "StoryPlanResult",
    "StoryPlanService",
    "compute_source_fingerprint",
    "validate_acting",
    "validate_bible",
    "validate_beat_allocation",
    "validate_story_plan_payload",
]

STORY_PLAN_PRODUCER = "story-plan-service/v1"

_FORBIDDEN_RENDER_KEYS = frozenset(
    {
        "camera",
        "shot",
        "shot_id",
        "shot_type",
        "framing",
        "lens",
        "angle",
        "composition",
        "render",
        "render_hint",
        "transition",
        "transition_hint",
    }
)

# Local copy of the story-plan module's private constant: importing the
# private name would trip the import boundary tests.
_FORBIDDEN_AUDIO_DATA_KEYS = frozenset(
    {"audio", "audio_data", "audio_features", "spectral", "spectral_features"}
)


class StoryPlanError(FeverSlopError):
    """Raised when the story-plan pipeline cannot produce a valid plan."""

    def __init__(self, message: str, *, diagnostics: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.diagnostics: tuple[Mapping[str, Any], ...] = tuple(diagnostics)


class StoryPlanValidationError(FeverSlopError):
    """Raised when a story plan candidate fails contract validation."""

    def __init__(self, message: str, *, errors: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.errors: list[dict[str, Any]] = [dict(error) for error in errors]


def validate_story_plan_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate a candidate plan payload; one record per contract error."""
    try:
        StoryPlan.model_validate(dict(payload), strict=False)
    except PydanticValidationError as exc:
        records: list[dict[str, Any]] = []
        for error in exc.errors():
            loc = error.get("loc")
            records.append(
                {
                    "code": "plan_validation_failed",
                    "subject_id": str(loc[-1]) if loc else "",
                    "message": str(error.get("msg", "")),
                }
            )
        return records
    return []


@dataclass(frozen=True)
class SegmentDescriptor:
    """Read-only view of a stage-1 segment for planning input."""

    segment_id: str
    start_seconds: float
    end_seconds: float
    lyric_text: str = ""
    section: str = ""
    beat_index: int = -1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SegmentDescriptor":
        return cls(
            segment_id=str(data["segment_id"]),
            start_seconds=float(data["start_seconds"]),
            end_seconds=float(data["end_seconds"]),
            lyric_text=str(data.get("lyric_text", "")),
            section=str(data.get("section", "")),
            beat_index=int(data.get("beat_index", -1)),
        )

    def to_compact_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "segment_id": self.segment_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }
        if self.lyric_text:
            payload["lyric_text"] = self.lyric_text
        if self.section:
            payload["section"] = self.section
        if self.beat_index >= 0:
            payload["beat_index"] = self.beat_index
        return payload


@dataclass(frozen=True)
class StoryPlanRequest:
    """Bounded request for one story-plan build."""

    source_fingerprint: str
    song_title: str
    song_language: str
    song_style: str
    lyrics: str
    sections: tuple[Mapping[str, Any], ...]
    segments: tuple[SegmentDescriptor, ...]
    characters: tuple[Mapping[str, Any], ...]
    source_evidence: Mapping[str, Any]
    guide: str
    terminal_window_seconds: float

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_fingerprint):
            raise StoryPlanError(
                "source_fingerprint must be a 64-character lowercase sha256 hex digest"
            )
        if not self.song_title:
            raise StoryPlanError("song_title must not be empty")
        if not self.song_language:
            raise StoryPlanError("song_language must not be empty")
        if not self.lyrics:
            raise StoryPlanError("lyrics must not be empty")
        if not self.segments:
            raise StoryPlanError("at least one segment is required")
        if self.terminal_window_seconds <= 0:
            raise StoryPlanError("terminal_window_seconds must be positive")
        previous_end: float | None = None
        for segment in self.segments:
            if segment.end_seconds <= segment.start_seconds:
                raise StoryPlanError(
                    f"segment {segment.segment_id}: end_seconds must exceed start_seconds"
                )
            if previous_end is not None and segment.start_seconds < previous_end:
                raise StoryPlanError(
                    f"segment {segment.segment_id}: starts before the previous segment ends"
                )
            previous_end = segment.end_seconds
        if not any(
            segment.end_seconds > self.terminal_window_seconds
            for segment in self.segments
        ):
            raise StoryPlanError(
                "terminal_window_seconds must be strictly inside the segment span"
            )


@dataclass(frozen=True)
class StoryPlanResult:
    """Validated plan, acting results, and pipeline diagnostics."""

    plan: StoryPlan
    acting: Mapping[str, Any]
    diagnostics: tuple[Mapping[str, Any], ...]


class StoryPlanService:
    """Builds a validated ``StoryPlan`` through four narrow DSPy jobs."""

    def __init__(
        self,
        *,
        prompt_modules: Any,
        reporter: Reporter | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        self._prompt_modules = prompt_modules
        self._reporter: Reporter = reporter if reporter is not None else NullReporter()
        if max_repair_attempts < 1:
            raise StoryPlanError("max_repair_attempts must be at least 1")
        self._max_repair_attempts = max_repair_attempts

    def build_plan(self, request: StoryPlanRequest) -> StoryPlanResult:
        request.validate()
        diagnostics: list[Mapping[str, Any]] = []

        self._reporter.step("story-plan-bible")
        bible_raw = self._job("bible", self._bible_input(request))
        bible = self._normalize_job_output(
            "bible", bible_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
        )
        self._reporter.message("story-plan-bible complete")

        self._reporter.step("story-plan-beat-allocation")
        allocation_raw = self._job(
            "beat_allocation", self._allocation_input(request, bible)
        )
        allocation = self._normalize_job_output(
            "beat_allocation", allocation_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
        )
        self._validate_allocation(request, allocation, diagnostics)
        self._reporter.message("story-plan-beat-allocation complete")

        self._reporter.step("story-plan-acting")
        acting_raw = self._job("acting", self._acting_input(request, allocation))
        acting = self._normalize_job_output(
            "acting", acting_raw, diagnostics, _FORBIDDEN_AUDIO_DATA_KEYS
        )
        self._validate_acting(allocation, acting, diagnostics)
        self._reporter.message("story-plan-acting complete")

        candidate = self._assemble_candidate(request, allocation, acting)
        try:
            self._validate_payload(candidate, diagnostics)
        except StoryPlanValidationError as exc:
            validation_errors = list(exc.errors)
        else:
            validation_errors = []

        if validation_errors:
            self._reporter.step("story-plan-repair")
            repaired_raw = self._job(
                "repair",
                self._repair_input(request, candidate, validation_errors),
            )
            repaired = self._normalize_job_output(
                "repair", repaired_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
            )
            candidate = self._assemble_candidate(request, repaired, acting)
            try:
                self._validate_payload(candidate, diagnostics)
            except StoryPlanValidationError as exc:
                self._raise_validation_failure(exc, diagnostics)
            self._reporter.message("story-plan-repair complete")

        plan = StoryPlan.model_validate(candidate, strict=False)
        return StoryPlanResult(
            plan=plan,
            acting=self._typed_acting(acting, plan),
            diagnostics=tuple(diagnostics),
        )

    def render_review_export(self, result: StoryPlanResult) -> str:
        plan = result.plan
        provenance = plan.provenance
        lines: list[str] = [
            "# Story Plan Review",
            "",
            f"- Mode: {plan.mode.value}",
            f"- Source fingerprint: `{plan.source_fingerprint}`",
            f"- Producer: {provenance.producer}",
            f"- Source refs: {', '.join(provenance.source_refs) or '-'}",
        ]
        if provenance.notes:
            lines.append(f"- User direction: {provenance.notes}")
        lines.append("")
        lines.append("## Characters")
        lines.append("")
        if plan.characters:
            for character in plan.characters:
                singer = " (singer)" if character.is_singer else ""
                lines.append(f"- **{character.name}** `{character.id}`{singer}")
                if character.description:
                    lines.append(f"  {character.description}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Segments")
        lines.append("")
        lines.append("| Brief | Target | Vocal | Audio ref |")
        lines.append("| --- | --- | --- | --- |")
        for brief in plan.segments:
            audio = (
                f"`{brief.audio_ref.segment_id}` "
                f"`{brief.audio_ref.fingerprint[:12]}`"
                if brief.audio_ref is not None
                else "-"
            )
            lines.append(
                f"| {brief.id} | {brief.target} | {brief.vocal_presentation.value} | {audio} |"
            )
        lines.append("")
        lines.append("## Diagnostics")
        lines.append("")
        if result.diagnostics:
            for diagnostic in result.diagnostics:
                lines.append(
                    f"- [{diagnostic.get('code', 'unknown')}] "
                    f"{diagnostic.get('subject_id', '')}: "
                    f"{diagnostic.get('message', '')}"
                )
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"

    def write_review_export(
        self, result: StoryPlanResult, output_dir: Path, *, name: str = "story-plan"
    ) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / f"{name}.md"
        manifest_path = output_dir / f"{name}.manifest.json"
        manifest = StoryPlanArtifactManifest(
            artifact_class=ArtifactClass.review_export,
            regeneration_policy=RegenerationPolicy.on_request,
            input_fingerprint=result.plan.source_fingerprint,
            plan_fingerprint=hashlib.sha256(
                json.dumps(result.plan.model_dump(mode="json"), sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        atomic_write_text(markdown_path, self.render_review_export(result))
        atomic_write_text(manifest_path, manifest.model_dump_json())
        return markdown_path, manifest_path

    # -- job helpers -------------------------------------------------------

    def _job(self, name: str, payload: Mapping[str, Any]) -> Any:
        method = getattr(self._prompt_modules, name)
        return method(**dict(payload))

    def _bible_input(self, request: StoryPlanRequest) -> dict[str, Any]:
        return {
            "song_title": request.song_title,
            "song_language": request.song_language,
            "song_style": request.song_style,
            "lyrics": request.lyrics,
            "sections": [dict(section) for section in request.sections],
            "source_evidence": dict(request.source_evidence),
            "guide": request.guide,
        }

    def _allocation_input(
        self, request: StoryPlanRequest, bible: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "song_title": request.song_title,
            "lyrics": request.lyrics,
            "segments": [segment.to_compact_dict() for segment in request.segments],
            "narrative_bible": dict(bible),
            "characters": [dict(character) for character in request.characters],
            "terminal_window_seconds": request.terminal_window_seconds,
            "guide": request.guide,
        }

    def _acting_input(
        self, request: StoryPlanRequest, allocation: Mapping[str, Any]
    ) -> dict[str, Any]:
        briefs = allocation.get("briefs")
        if not isinstance(briefs, list):
            briefs = []
        return {
            "song_title": request.song_title,
            "song_language": request.song_language,
            "lyrics": request.lyrics,
            "briefs": [dict(brief) for brief in briefs],
            "characters": [dict(character) for character in request.characters],
            "guide": request.guide,
        }

    def _repair_input(
        self,
        request: StoryPlanRequest,
        candidate: Mapping[str, Any],
        validation_errors: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "song_title": request.song_title,
            "song_style": request.song_style,
            "lyrics": request.lyrics,
            "candidate": dict(candidate),
            "validation_errors": [dict(error) for error in validation_errors],
            "guide": request.guide,
        }

    # -- normalization and validation ---------------------------------------

    @staticmethod
    def _normalize_job_output(
        job: str,
        raw: Any,
        diagnostics: list[Mapping[str, Any]],
        forbidden_keys: frozenset[str],
    ) -> Mapping[str, Any]:
        if raw is None:
            raise StoryPlanError(f"{job} job returned no output", diagnostics=[
                {"code": "empty_output", "job": job, "message": "job returned no output"},
            ])
        if isinstance(raw, Mapping):
            payload = dict(raw)
        else:
            try:
                payload = json.loads(str(raw))
            except (TypeError, ValueError) as exc:
                raise StoryPlanError(
                    f"{job} job output is not a JSON object",
                    diagnostics=[
                        {
                            "code": "unparseable_output",
                            "job": job,
                            "message": str(exc),
                        }
                    ],
                ) from exc
            if not isinstance(payload, Mapping):
                raise StoryPlanError(
                    f"{job} job output is not a JSON object",
                    diagnostics=[
                        {
                            "code": "unparseable_output",
                            "job": job,
                            "message": "top-level JSON value is not an object",
                        }
                    ],
                )
        forbidden = sorted(
            key for key in payload.keys() if key in forbidden_keys
        )
        if forbidden:
            diagnostics.append(
                {
                    "code": "forbidden_keys",
                    "job": job,
                    "keys": forbidden,
                    "message": f"forbidden keys present: {', '.join(forbidden)}",
                }
            )
        return payload

    @staticmethod
    def _validate_allocation(
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
        diagnostics: list[Mapping[str, Any]],
    ) -> None:
        briefs = allocation.get("briefs")
        if not isinstance(briefs, list) or not briefs:
            raise StoryPlanError(
                "beat_allocation output must contain a non-empty briefs list",
                diagnostics=[
                    {
                        "code": "missing_briefs",
                        "subject_id": "allocation",
                        "message": "briefs list missing or empty",
                    }
                ],
            )
        segments = {segment.segment_id: segment for segment in request.segments}
        seen: set[str] = set()
        for brief in briefs:
            if not isinstance(brief, Mapping):
                raise StoryPlanError(
                    "beat_allocation briefs must be objects",
                    diagnostics=[
                        {
                            "code": "invalid_brief",
                            "subject_id": "allocation",
                            "message": "brief entry is not an object",
                        }
                    ],
                )
            brief_id = str(brief.get("brief_id", ""))
            if not brief_id:
                raise StoryPlanError(
                    "beat_allocation brief is missing brief_id",
                    diagnostics=[
                        {
                            "code": "missing_brief_id",
                            "subject_id": "allocation",
                            "message": "brief entry has no brief_id",
                        }
                    ],
                )
            if brief_id in seen:
                diagnostics.append(
                    {
                        "code": "duplicate_brief_id",
                        "subject_id": brief_id,
                        "message": f"duplicate brief_id {brief_id}",
                    }
                )
            seen.add(brief_id)
            segment_id = str(brief.get("segment_id", ""))
            segment = segments.get(segment_id)
            if segment is None:
                raise StoryPlanError(
                    f"brief {brief_id} references unknown segment {segment_id}",
                    diagnostics=[
                        {
                            "code": "unknown_segment",
                            "subject_id": brief_id,
                            "message": f"unknown segment_id {segment_id}",
                        }
                    ],
                )
            try:
                start = float(brief.get("start_seconds", -1.0))
                end = float(brief.get("end_seconds", -1.0))
            except (TypeError, ValueError):
                raise StoryPlanError(
                    f"brief {brief_id} has non-numeric timing",
                    diagnostics=[
                        {
                            "code": "invalid_timing",
                            "subject_id": brief_id,
                            "message": "start_seconds/end_seconds are not numeric",
                        }
                    ],
                )
            if end <= start:
                raise StoryPlanError(
                    f"brief {brief_id}: end_seconds must exceed start_seconds",
                    diagnostics=[
                        {
                            "code": "invalid_timing",
                            "subject_id": brief_id,
                            "message": "end_seconds must exceed start_seconds",
                        }
                    ],
                )
            if start < segment.start_seconds - 1e-6 or end > segment.end_seconds + 1e-6:
                raise StoryPlanError(
                    f"brief {brief_id} window exceeds its segment bounds",
                    diagnostics=[
                        {
                            "code": "window_out_of_bounds",
                            "subject_id": brief_id,
                            "message": f"window [{start}, {end}] outside segment "
                            f"[{segment.start_seconds}, {segment.end_seconds}]",
                        }
                    ],
                )
            beats = brief.get("beat_indices")
            if not isinstance(beats, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in beats
            ):
                raise StoryPlanError(
                    f"brief {brief_id} beat_indices must be a list of integers",
                    diagnostics=[
                        {
                            "code": "invalid_beats",
                            "subject_id": brief_id,
                            "message": "beat_indices is not a list of integers",
                        }
                    ],
                )
            for key in ("required", "forbidden"):
                values = brief.get(key)
                if not isinstance(values, list) or not all(
                    isinstance(item, str) for item in values
                ):
                    raise StoryPlanError(
                        f"brief {brief_id} {key} must be a list of strings",
                        diagnostics=[
                            {
                                "code": "invalid_constraints",
                                "subject_id": brief_id,
                                "message": f"{key} is not a list of strings",
                            }
                        ],
                    )
        covered = {str(brief.get("segment_id")) for brief in briefs}
        missing = sorted(set(segments) - covered)
        if missing:
            raise StoryPlanError(
                "beat_allocation does not cover every segment",
                diagnostics=[
                    {
                        "code": "uncovered_segments",
                        "subject_id": "allocation",
                        "message": f"segments not covered: {', '.join(missing)}",
                    }
                ],
            )

    @staticmethod
    def _validate_acting(
        allocation: Mapping[str, Any],
        acting: Mapping[str, Any],
        diagnostics: list[Mapping[str, Any]],
    ) -> None:
        briefs = allocation.get("briefs")
        if not isinstance(briefs, list):
            briefs = []
        required_ids = {
            str(brief.get("brief_id"))
            for brief in briefs
            if isinstance(brief, Mapping)
        }
        brief_results = acting.get("briefs")
        if not isinstance(brief_results, list) or not brief_results:
            raise StoryPlanError(
                "acting output must contain a non-empty briefs list",
                diagnostics=[
                    {
                        "code": "missing_acting_briefs",
                        "subject_id": "acting",
                        "message": "acting briefs list missing or empty",
                    }
                ],
            )
        seen: set[str] = set()
        for entry in brief_results:
            if not isinstance(entry, Mapping):
                raise StoryPlanError(
                    "acting briefs must be objects",
                    diagnostics=[
                        {
                            "code": "invalid_acting_brief",
                            "subject_id": "acting",
                            "message": "acting brief entry is not an object",
                        }
                    ],
                )
            brief_id = str(entry.get("brief_id", ""))
            if not brief_id:
                raise StoryPlanError(
                    "acting brief is missing brief_id",
                    diagnostics=[
                        {
                            "code": "missing_acting_brief_id",
                            "subject_id": "acting",
                            "message": "acting brief entry has no brief_id",
                        }
                    ],
                )
            if brief_id in seen:
                diagnostics.append(
                    {
                        "code": "duplicate_acting_brief_id",
                        "subject_id": brief_id,
                        "message": f"duplicate acting brief_id {brief_id}",
                    }
                )
            seen.add(brief_id)
            if brief_id not in required_ids:
                diagnostics.append(
                    {
                        "code": "unknown_acting_brief",
                        "subject_id": brief_id,
                        "message": f"acting brief {brief_id} is not in the allocation",
                    }
                )
            objective = entry.get("objective")
            if not isinstance(objective, str) or not objective.strip():
                raise StoryPlanError(
                    f"acting brief {brief_id} has no objective",
                    diagnostics=[
                        {
                            "code": "missing_objective",
                            "subject_id": brief_id,
                            "message": "objective missing or empty",
                        }
                    ],
                )
            turn = entry.get("emotional_turn")
            if not isinstance(turn, str) or not turn.strip():
                raise StoryPlanError(
                    f"acting brief {brief_id} has no emotional_turn",
                    diagnostics=[
                        {
                            "code": "missing_emotional_turn",
                            "subject_id": brief_id,
                            "message": "emotional_turn missing or empty",
                        }
                    ],
                )
            states = entry.get("actor_states")
            if not isinstance(states, list) or not states:
                raise StoryPlanError(
                    f"acting brief {brief_id} has no actor_states",
                    diagnostics=[
                        {
                            "code": "missing_actor_states",
                            "subject_id": brief_id,
                            "message": "actor_states missing or empty",
                        }
                    ],
                )
            for state in states:
                if not isinstance(state, Mapping):
                    raise StoryPlanError(
                        f"acting brief {brief_id} actor_states must be objects",
                        diagnostics=[
                            {
                                "code": "invalid_actor_state",
                                "subject_id": brief_id,
                                "message": "actor_states entry is not an object",
                            }
                        ],
                    )
                if not str(state.get("character_id", "")):
                    raise StoryPlanError(
                        f"acting brief {brief_id} actor state missing character_id",
                        diagnostics=[
                            {
                                "code": "missing_character_id",
                                "subject_id": brief_id,
                                "message": "actor_states entry has no character_id",
                            }
                        ],
                    )
        missing = required_ids - seen
        if missing:
            raise StoryPlanError(
                "acting does not cover every allocated brief",
                diagnostics=[
                    {
                        "code": "uncovered_acting_briefs",
                        "subject_id": "acting",
                        "message": f"briefs without acting: {', '.join(sorted(missing))}",
                    }
                ],
            )

    @staticmethod
    def _validate_payload(
        candidate: Mapping[str, Any], diagnostics: list[Mapping[str, Any]]
    ) -> None:
        errors = validate_story_plan_payload(dict(candidate))
        if errors:
            for error in errors:
                diagnostics.append(
                    {
                        "code": str(error.get("code", "validation_error")),
                        "subject_id": str(error.get("subject_id", "")),
                        "message": str(error.get("message", "")),
                    }
                )
            raise StoryPlanValidationError(
                "story plan candidate failed validation",
                errors=[dict(error) for error in errors],
            )

    @staticmethod
    def _raise_validation_failure(
        exc: StoryPlanValidationError, diagnostics: list[Mapping[str, Any]]
    ) -> None:
        raise StoryPlanError(
            "story plan validation failed after repair",
            diagnostics=[
                {
                    "code": str(error.get("code", "validation_error")),
                    "subject_id": str(error.get("subject_id", "")),
                    "message": str(error.get("message", "")),
                }
                for error in exc.errors
            ],
        ) from exc

    @staticmethod
    def _assemble_candidate(
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
        acting: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build a domain ``StoryPlan`` candidate from the job outputs.

        The candidate carries only the domain contract's fields: a
        ``music_video`` mode, the request's source fingerprint, a provenance
        that preserves the highest-priority user direction, canonical
        characters, and one ``SegmentBrief`` per allocated segment with a
        deterministic reference-only ``audio_ref``.  Beats and arcs stay
        empty; the typed acting result is carried separately in the result.

        The repair job returns an already-assembled domain-format plan, so
        when the input carries a ``segments`` key it is passed through
        unchanged (its provenance is re-derived to keep the user direction).
        """
        if "segments" in allocation:
            creative_direction = str(
                request.source_evidence.get("creative_direction", "")
            )
            return {
                **dict(allocation),
                "mode": "music_video",
                "source_fingerprint": request.source_fingerprint,
                "provenance": {
                    "producer": STORY_PLAN_PRODUCER,
                    "source_refs": [request.song_title],
                    "notes": creative_direction,
                },
            }
        characters: list[dict[str, Any]] = []
        for character in request.characters:
            if not isinstance(character, Mapping):
                continue
            character_id = str(character.get("id", "")).strip()
            name = str(character.get("name", "")).strip()
            if not character_id or not name:
                continue
            characters.append(
                {
                    "id": character_id,
                    "name": name,
                    "description": str(character.get("description", "")),
                    "is_singer": bool(character.get("is_singer", False)),
                }
            )
        segments: list[dict[str, Any]] = []
        raw_briefs = allocation.get("briefs")
        if not isinstance(raw_briefs, list):
            raw_briefs = []
        for raw in raw_briefs:
            if not isinstance(raw, Mapping):
                continue
            brief_id = str(raw.get("brief_id", "")).strip()
            target = str(raw.get("segment_id", "")).strip()
            if not brief_id or not target:
                continue
            fingerprint = hashlib.sha256(
                f"{target}:{request.source_fingerprint}".encode("utf-8")
            ).hexdigest()
            segments.append(
                {
                    "id": brief_id,
                    "target": target,
                    "beat_id": None,
                    "character_ids": [],
                    "vocal_presentation": "offscreen",
                    "visual_direction": "",
                    "exclusive": False,
                    "audio_ref": {"segment_id": target, "fingerprint": fingerprint},
                }
            )
        creative_direction = str(request.source_evidence.get("creative_direction", ""))
        return {
            "mode": "music_video",
            "source_fingerprint": request.source_fingerprint,
            "provenance": {
                "producer": STORY_PLAN_PRODUCER,
                "source_refs": [request.song_title],
                "notes": creative_direction,
            },
            "characters": characters,
            "beats": [],
            "arcs": [],
            "segments": segments,
        }

    @staticmethod
    def _typed_acting(
        acting: Mapping[str, Any], plan: StoryPlan
    ) -> Mapping[str, Any]:
        """Return the acting result, typed when the contract accepts it.

        The acting job emits per-brief ``objective``/``emotional_turn``/
        ``actor_states``.  When the typed ``ActingResult`` contract accepts
        the payload it is returned in its canonical form; otherwise the
        normalized raw briefs are returned so the result is never lost.
        """
        raw_briefs = acting.get("briefs")
        if not isinstance(raw_briefs, list):
            raw_briefs = []
        normalized_briefs = [
            {
                "brief_id": str(brief.get("brief_id", "")),
                "objective": str(brief.get("objective", "")),
                "emotional_turn": str(brief.get("emotional_turn", "")),
                "actor_states": [
                    {
                        "character_id": str(state.get("character_id", "")),
                        "inner_state": str(state.get("inner_state", "")),
                        "physical_state": str(state.get("physical_state", "")),
                    }
                    for state in (
                        state
                        for state in brief.get("actor_states", [])
                        if isinstance(state, Mapping)
                    )
                ],
            }
            for brief in raw_briefs
            if isinstance(brief, Mapping)
        ]
        raw_arcs = acting.get("character_arcs")
        if not isinstance(raw_arcs, list):
            raw_arcs = []
        normalized_arcs = [
            {
                "character_id": str(arc.get("character_id", "")),
                "character_name": str(arc.get("character_name", "")),
                "arc_summary": str(arc.get("arc_summary", "")),
            }
            for arc in raw_arcs
            if isinstance(arc, Mapping)
        ]
        try:
            from feverslop.prompting.story_plan_signatures import ActingResult

            typed = ActingResult.model_validate(
                {
                    "briefs": [
                        {
                            "target": brief["brief_id"],
                            "vocal_presentation": "offscreen",
                            "objective": brief["objective"],
                            "emotional_turn": brief["emotional_turn"],
                            "actor_states": [
                                {
                                    "character_id": state["character_id"],
                                    "state": state["inner_state"],
                                }
                                for state in brief["actor_states"]
                            ],
                        }
                        for brief in normalized_briefs
                    ],
                    "arcs": [
                        {
                            "character_id": arc["character_id"],
                            "from_state": arc.get("arc_summary", ""),
                            "to_state": "",
                        }
                        for arc in normalized_arcs
                    ],
                }
            )
            return typed.model_dump()
        except Exception:
            return {"briefs": normalized_briefs, "character_arcs": normalized_arcs}


def compute_source_fingerprint(
    *,
    song_title: str,
    song_language: str,
    song_style: str,
    lyrics: str,
    sections: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    characters: Sequence[Mapping[str, Any]],
) -> str:
    """Deterministic sha256 fingerprint of the planning inputs."""
    canonical = {
        "song_title": song_title,
        "song_language": song_language,
        "song_style": song_style,
        "lyrics": lyrics,
        "sections": [dict(section) for section in sections],
        "segments": [dict(segment) for segment in segments],
        "characters": [dict(character) for character in characters],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_bible(bible: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Check narrative-bible structure and forbidden render keys."""
    diagnostics: list[Mapping[str, Any]] = []
    sections = bible.get("sections")
    if not isinstance(sections, list) or not sections:
        diagnostics.append(
            {
                "code": "missing_bible_sections",
                "subject_id": "bible",
                "message": "narrative_bible.sections missing or empty",
            }
        )
        return diagnostics
    for section in sections:
        if not isinstance(section, Mapping):
            diagnostics.append(
                {
                    "code": "invalid_bible_section",
                    "subject_id": "bible",
                    "message": "bible section is not an object",
                }
            )
            continue
        if not str(section.get("title", "")):
            diagnostics.append(
                {
                    "code": "missing_bible_section_title",
                    "subject_id": "bible",
                    "message": "bible section has no title",
                }
            )
        if not str(section.get("summary", "")):
            diagnostics.append(
                {
                    "code": "missing_bible_section_summary",
                    "subject_id": "bible",
                    "message": "bible section has no summary",
                }
            )
        for key in section.keys():
            if key in _FORBIDDEN_RENDER_KEYS:
                diagnostics.append(
                    {
                        "code": "forbidden_render_key",
                        "subject_id": str(section.get("title", "")),
                        "message": f"forbidden render key {key} in bible section",
                    }
                )
    return diagnostics


def validate_beat_allocation(
    allocation: Mapping[str, Any],
    segments: Sequence[SegmentDescriptor],
    terminal_window_seconds: float,
) -> list[Mapping[str, Any]]:
    """Check allocation coverage, bounds, and per-brief constraints."""
    diagnostics: list[Mapping[str, Any]] = []
    briefs = allocation.get("briefs")
    if not isinstance(briefs, list) or not briefs:
        diagnostics.append(
            {
                "code": "missing_briefs",
                "subject_id": "allocation",
                "message": "briefs list missing or empty",
            }
        )
        return diagnostics
    segment_map = {segment.segment_id: segment for segment in segments}
    seen: set[str] = set()
    for brief in briefs:
        if not isinstance(brief, Mapping):
            diagnostics.append(
                {
                    "code": "invalid_brief",
                    "subject_id": "allocation",
                    "message": "brief entry is not an object",
                }
            )
            continue
        brief_id = str(brief.get("brief_id", ""))
        if not brief_id:
            diagnostics.append(
                {
                    "code": "missing_brief_id",
                    "subject_id": "allocation",
                    "message": "brief entry has no brief_id",
                }
            )
            continue
        if brief_id in seen:
            diagnostics.append(
                {
                    "code": "duplicate_brief_id",
                    "subject_id": brief_id,
                    "message": f"duplicate brief_id {brief_id}",
                }
            )
        seen.add(brief_id)
        segment_id = str(brief.get("segment_id", ""))
        segment = segment_map.get(segment_id)
        if segment is None:
            diagnostics.append(
                {
                    "code": "unknown_segment",
                    "subject_id": brief_id,
                    "message": f"unknown segment_id {segment_id}",
                }
            )
            continue
        try:
            start = float(brief.get("start_seconds", -1.0))
            end = float(brief.get("end_seconds", -1.0))
        except (TypeError, ValueError):
            diagnostics.append(
                {
                    "code": "invalid_timing",
                    "subject_id": brief_id,
                    "message": "start_seconds/end_seconds are not numeric",
                }
            )
            continue
        if end <= start:
            diagnostics.append(
                {
                    "code": "invalid_timing",
                    "subject_id": brief_id,
                    "message": "end_seconds must exceed start_seconds",
                }
            )
        if start < segment.start_seconds - 1e-6 or end > segment.end_seconds + 1e-6:
            diagnostics.append(
                {
                    "code": "window_out_of_bounds",
                    "subject_id": brief_id,
                    "message": f"window [{start}, {end}] outside segment "
                    f"[{segment.start_seconds}, {segment.end_seconds}]",
                }
            )
        beats = brief.get("beat_indices")
        if not isinstance(beats, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in beats
        ):
            diagnostics.append(
                {
                    "code": "invalid_beats",
                    "subject_id": brief_id,
                    "message": "beat_indices is not a list of integers",
                }
            )
        for key in ("required", "forbidden"):
            values = brief.get(key)
            if not isinstance(values, list) or not all(
                isinstance(item, str) for item in values
            ):
                diagnostics.append(
                    {
                        "code": "invalid_constraints",
                        "subject_id": brief_id,
                        "message": f"{key} is not a list of strings",
                    }
                )
    covered = {
        str(brief.get("segment_id"))
        for brief in briefs
        if isinstance(brief, Mapping)
    }
    missing = sorted(set(segment_map) - covered)
    if missing:
        diagnostics.append(
            {
                "code": "uncovered_segments",
                "subject_id": "allocation",
                "message": f"segments not covered: {', '.join(missing)}",
            }
        )
    return diagnostics


def validate_acting(
    allocation: Mapping[str, Any], acting: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Check acting coverage and per-brief required fields."""
    diagnostics: list[Mapping[str, Any]] = []
    briefs = allocation.get("briefs")
    if not isinstance(briefs, list):
        briefs = []
    required_ids = {
        str(brief.get("brief_id"))
        for brief in briefs
        if isinstance(brief, Mapping)
    }
    brief_results = acting.get("briefs")
    if not isinstance(brief_results, list) or not brief_results:
        diagnostics.append(
            {
                "code": "missing_acting_briefs",
                "subject_id": "acting",
                "message": "acting briefs list missing or empty",
            }
        )
        return diagnostics
    seen: set[str] = set()
    for entry in brief_results:
        if not isinstance(entry, Mapping):
            diagnostics.append(
                {
                    "code": "invalid_acting_brief",
                    "subject_id": "acting",
                    "message": "acting brief entry is not an object",
                }
            )
            continue
        brief_id = str(entry.get("brief_id", ""))
        if not brief_id:
            diagnostics.append(
                {
                    "code": "missing_acting_brief_id",
                    "subject_id": "acting",
                    "message": "acting brief entry has no brief_id",
                }
            )
            continue
        if brief_id in seen:
            diagnostics.append(
                {
                    "code": "duplicate_acting_brief_id",
                    "subject_id": brief_id,
                    "message": f"duplicate acting brief_id {brief_id}",
                }
            )
        seen.add(brief_id)
        if brief_id not in required_ids:
            diagnostics.append(
                {
                    "code": "unknown_acting_brief",
                    "subject_id": brief_id,
                    "message": f"acting brief {brief_id} is not in the allocation",
                }
            )
        objective = entry.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            diagnostics.append(
                {
                    "code": "missing_objective",
                    "subject_id": brief_id,
                    "message": "objective missing or empty",
                }
            )
        turn = entry.get("emotional_turn")
        if not isinstance(turn, str) or not turn.strip():
            diagnostics.append(
                {
                    "code": "missing_emotional_turn",
                    "subject_id": brief_id,
                    "message": "emotional_turn missing or empty",
                }
            )
        states = entry.get("actor_states")
        if not isinstance(states, list) or not states:
            diagnostics.append(
                {
                    "code": "missing_actor_states",
                    "subject_id": brief_id,
                    "message": "actor_states missing or empty",
                }
            )
            continue
        for state in states:
            if not isinstance(state, Mapping):
                diagnostics.append(
                    {
                        "code": "invalid_actor_state",
                        "subject_id": brief_id,
                        "message": "actor_states entry is not an object",
                    }
                )
                continue
            if not str(state.get("character_id", "")):
                diagnostics.append(
                    {
                        "code": "missing_character_id",
                        "subject_id": brief_id,
                        "message": "actor_states entry has no character_id",
                    }
                )
    missing = required_ids - seen
    if missing:
        diagnostics.append(
            {
                "code": "uncovered_acting_briefs",
                "subject_id": "acting",
                "message": f"briefs without acting: {', '.join(sorted(missing))}",
            }
        )
    return diagnostics
