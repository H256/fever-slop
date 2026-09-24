"""Typed DSPy planning service for the story-plan contract (issue #1385).

``StoryPlanService`` drives three narrow DSPy jobs -- narrative bible, a
small beat sheet, and per-brief acting -- with deterministic validation and
scene binding between stages.  The LLM never allocates every scene and never
receives a complete generated plan for a retry: Python owns that large,
mechanical mapping.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
from feverslop.utils.sub_step_progress import SubStepProgress

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
        """Adapt either supported Stage 1 timing representation.

        Persisted Stage 1 artifacts use ``start`` / ``end``.  The planning
        boundary exposes the explicit ``*_seconds`` names, so newer callers
        may already provide those without changing the authoritative artifact.
        """
        start = data["start_seconds"] if "start_seconds" in data else data["start"]
        end = data["end_seconds"] if "end_seconds" in data else data["end"]
        return cls(
            segment_id=str(data["segment_id"]),
            start_seconds=float(start),
            end_seconds=float(end),
            lyric_text=str(data.get("lyric_text", data.get("lyrics", ""))),
            section=str(data.get("section", "")),
            beat_index=int(data.get("beat_index", -1)),
        )

    @staticmethod
    def has_stage1_timing(data: Mapping[str, Any]) -> bool:
        """Whether a Stage 1 record has one complete supported timing pair."""
        return (
            "start_seconds" in data and "end_seconds" in data
        ) or ("start" in data and "end" in data)

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
    story_idea: str = ""
    locations: tuple[Mapping[str, Any], ...] = ()
    props: tuple[Mapping[str, Any], ...] = ()

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
    live_prompts: Mapping[str, Any] = field(default_factory=dict)


class StoryPlanService:
    """Build a validated ``StoryPlan`` from small creative DSPy jobs."""

    def __init__(
        self,
        *,
        prompt_modules: Any,
        reporter: Reporter | None = None,
        acting_batch_size: int = 4,
    ) -> None:
        self._prompt_modules = prompt_modules
        self._reporter: Reporter = reporter if reporter is not None else NullReporter()
        if acting_batch_size < 1:
            raise StoryPlanError("acting_batch_size must be at least 1")
        self._acting_batch_size = acting_batch_size

    def build_plan(self, request: StoryPlanRequest) -> StoryPlanResult:
        request.validate()
        diagnostics: list[Mapping[str, Any]] = []

        self._reporter.step("Story plan - narrative bible")
        self._reporter.message("[cyan]Extracting the premise, stakes, and story facts…[/cyan]")
        bible_raw = self._reporter.run_progress(
            "Story plan - reading the story", lambda: self._job("bible", self._bible_input(request))
        )
        bible = self._normalize_job_output(
            "bible", bible_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
        )
        self._reporter.message("story-plan-bible complete")

        self._reporter.step("story-plan-arc-skeleton")
        arc_raw = self._job("arc_skeleton", self._arc_skeleton_input(request, bible))
        arc = self._normalize_job_output(
            "arc_skeleton", arc_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
        )
        beats = self._validate_arc(arc, diagnostics)
        self._reporter.message(
            "story-plan-arc-skeleton complete: "
            f"{self._describe_arc({'typed_beats': beats})}"
        )

        self._reporter.step("story-plan-beat-allocation")
        allocation_raw = self._job(
            "beat_allocation", self._allocation_input(request, bible, beats)
        )
        allocation_raw = self._normalize_job_output(
            "beat_allocation", allocation_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
        )
        brief_allocations = (
            allocation_raw.get("brief_allocations", [])
            if isinstance(allocation_raw, Mapping)
            else allocation_raw
        )
        allocation = {
            "beats": beats,
            "brief_allocations": (
                brief_allocations if isinstance(brief_allocations, list) else []
            ),
        }
        allocation = self._coerce_typed_allocation(request, allocation)
        self._validate_allocation(request, allocation, diagnostics)
        self._reporter.message(
            "story-plan-beat-allocation complete: "
            f"{len(allocation.get('briefs', []))} segment brief(s) allocated"
        )

        self._reporter.step("story-plan-entity-resolution")
        resolved_config, entity_decisions = self._resolve_entities(
            request, allocation
        )
        use_count = sum(1 for d in entity_decisions if d["decision"] == "use")
        extend_count = sum(1 for d in entity_decisions if d["decision"] == "extend")
        invent_count = sum(1 for d in entity_decisions if d["decision"] == "invent")
        self._reporter.message(
            "story-plan-entity-resolution complete: "
            f"{use_count} use, {extend_count} extend, {invent_count} invent"
        )
        for decision in entity_decisions:
            self._reporter.message(
                f"  entity {decision['entity_type']} {decision['entity_id']}: "
                f"{decision['decision']} ({decision['name']})"
            )

        self._reporter.step("Story plan - acting beats")
        acting = self._build_acting_in_batches(request, allocation, diagnostics)
        self._validate_acting(allocation, acting, diagnostics)
        self._reporter.message(
            "story-plan-acting complete: "
            f"{len(acting)} segment brief(s) covered"
        )

        candidate = self._assemble_candidate(
            request, allocation, acting, resolved_config
        )
        try:
            self._validate_payload(candidate, diagnostics)
        except StoryPlanValidationError as exc:
            validation_errors = list(exc.errors)
        else:
            validation_errors = []

        if validation_errors:
            self._reporter.step("story-plan-repair")
            for error in validation_errors:
                self._reporter.message(
                    f"  validation error: {error.get('code', '?')}: "
                    f"{error.get('message', '')}"
                )
            repaired_raw = self._job(
                "repair",
                self._repair_input(request, candidate, validation_errors),
            )
            repaired = self._normalize_job_output(
                "repair", repaired_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
            )
            candidate = self._assemble_candidate(
                request, repaired, acting, resolved_config
            )
            try:
                self._validate_payload(candidate, diagnostics)
            except StoryPlanValidationError as exc:
                self._raise_validation_failure(exc, diagnostics)
            self._reporter.message("story-plan-repair complete")

        if diagnostics:
            self._reporter.step("story-plan-diagnostics")
            for diag in diagnostics:
                self._reporter.warning(
                    f"{diag.get('code', '?')}: {diag.get('message', '')}"
                )

        plan = StoryPlan.model_validate(candidate, strict=False)
        self._reporter.table(
            "Story plan locked",
            ["Scenes", "Beats", "Acting briefs"],
            [[str(len(plan.segments)), str(len(plan.beats)), str(len(acting.get("briefs", [])))]],
        )
        live_prompts = self._build_live_prompts(request, plan, bible, diagnostics)
        return StoryPlanResult(
            plan=plan,
            acting=self._typed_acting(acting, plan),
            diagnostics=tuple(diagnostics),
            live_prompts=live_prompts,
        )

    def _build_acting_in_batches(
        self,
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
        diagnostics: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Generate acting in bounded deterministic batches with targeted retry."""
        raw_briefs = allocation.get("briefs")
        if not isinstance(raw_briefs, list):
            raw_briefs = []
        merged: dict[str, Any] = {"briefs": [], "character_arcs": []}
        ordered_briefs = [brief for brief in raw_briefs if isinstance(brief, Mapping)]
        total_briefs = len(ordered_briefs)
        total_batches = (total_briefs + self._acting_batch_size - 1) // self._acting_batch_size
        progress = SubStepProgress(
            self._reporter,
            "Story plan - acting",
            total_briefs,
            interval=1,
        )
        progress.update(0, detail=f"batch 1/{total_batches}", force=True)
        for start in range(0, len(ordered_briefs), self._acting_batch_size):
            batch = ordered_briefs[start:start + self._acting_batch_size]
            expected = [str(brief.get("brief_id", "")) for brief in batch]
            batch_number = start // self._acting_batch_size + 1
            batch_end = start + len(batch)
            self._reporter.message(
                f"story-plan-acting batch {batch_number}/{total_batches}: "
                f"requested {len(expected)} briefs (scenes {start + 1}-{batch_end}/{total_briefs})"
            )
            response = self._acting_batch(
                request, allocation, batch, diagnostics, expected
            )
            returned = self._acting_brief_ids(response)
            missing = [brief_id for brief_id in expected if brief_id not in returned]
            if missing:
                self._reporter.message(
                    f"story-plan-acting retry: requesting {len(missing)} missing briefs"
                )
                retry_batch = [
                    brief for brief in batch
                    if str(brief.get("brief_id", "")) in set(missing)
                ]
                retry = self._acting_batch(
                    request, allocation, retry_batch, diagnostics, missing
                )
                response = {
                    "briefs": [
                        *[brief for brief in response.get("briefs", []) if isinstance(brief, Mapping)],
                        *[brief for brief in retry.get("briefs", []) if isinstance(brief, Mapping)],
                    ],
                    "character_arcs": [
                        *response.get("character_arcs", []),
                        *retry.get("character_arcs", []),
                    ],
                }
            merged["briefs"].extend(
                brief for brief in response.get("briefs", [])
                if isinstance(brief, Mapping)
            )
            merged["character_arcs"].extend(
                arc for arc in response.get("character_arcs", [])
                if isinstance(arc, Mapping)
            )
            acting_by_id = {
                str(brief.get("brief_id", "")): brief
                for brief in response.get("briefs", [])
                if isinstance(brief, Mapping)
            }
            self._reporter.table(
                f"Acting briefs - batch {batch_number}/{total_batches}",
                ["Scene", "Time", "Objective", "Emotional turn", "Voice"],
                [
                    [
                        str(brief.get("segment_id", "")),
                        self._brief_time_range(brief),
                        self._brief_excerpt(acting_by_id.get(str(brief.get("brief_id", "")), {}).get("objective", "")),
                        self._brief_excerpt(acting_by_id.get(str(brief.get("brief_id", "")), {}).get("emotional_turn", "")),
                        self._brief_excerpt(acting_by_id.get(str(brief.get("brief_id", "")), {}).get("vocal_presentation", "offscreen")),
                    ]
                    for brief in batch
                ],
            )
            progress.update(batch_end, detail=f"batch {batch_number}/{total_batches}")
        # Model order is not authoritative.  The allocation order is the
        # canonical narrative order and must survive every batch/retry.
        by_id = {str(brief.get("brief_id")): brief for brief in merged["briefs"]}
        merged["briefs"] = [
            by_id[brief_id]
            for brief_id in (str(brief.get("brief_id", "")) for brief in ordered_briefs)
            if brief_id in by_id
        ]
        self._dedup_exclusive_allocations(merged["briefs"])
        return merged

    @staticmethod
    def _dedup_exclusive_allocations(briefs: list[Mapping[str, Any]]) -> None:
        """Keep at most one exclusive brief per beat (first in canonical order).

        The model may mark several briefs in one beat exclusive; the plan
        allows at most one.  Clear the rest so the allocation validates.
        """
        exclusive_seen: set[str] = set()
        for brief in briefs:
            if not isinstance(brief, Mapping):
                continue
            if brief.get("exclusive") and brief.get("beat_id"):
                beat_id = str(brief["beat_id"])
                if beat_id in exclusive_seen:
                    brief["exclusive"] = False
                else:
                    exclusive_seen.add(beat_id)

    @staticmethod
    def _acting_brief_ids(acting: Mapping[str, Any]) -> set[str]:
        return {
            str(brief.get("brief_id", ""))
            for brief in acting.get("briefs", [])
            if isinstance(brief, Mapping) and str(brief.get("brief_id", ""))
        }

    @staticmethod
    def _brief_time_range(brief: Mapping[str, Any]) -> str:
        def timestamp(value: Any) -> str:
            seconds = max(0, round(float(value)))
            return f"{seconds // 60:02d}:{seconds % 60:02d}"

        return f"{timestamp(brief.get('start_seconds', 0))}–{timestamp(brief.get('end_seconds', 0))}"

    @staticmethod
    def _brief_excerpt(value: Any, *, limit: int = 72) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= limit else f"{text[:limit - 1].rstrip()}…"

    def _acting_batch(
        self,
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
        batch: list[Mapping[str, Any]],
        diagnostics: list[Mapping[str, Any]],
        expected: list[str],
    ) -> dict[str, Any]:
        batch_allocation = {
            **dict(allocation),
            "briefs": [dict(brief) for brief in batch],
            "expected_brief_ids": list(expected),
        }
        raw = self._reporter.run_progress(
            "Story plan - writing acting",
            lambda: self._job("acting", self._acting_input(request, batch_allocation)),
        )
        acting = self._normalize_job_output(
            "acting", raw, diagnostics, _FORBIDDEN_AUDIO_DATA_KEYS
        )
        acting = self._coerce_typed_acting(request, batch_allocation, acting)
        raw_output = acting.get("briefs", [])
        output_ids = [
            str(brief.get("brief_id", ""))
            for brief in raw_output
            if isinstance(brief, Mapping)
        ]
        if len(output_ids) != len(set(output_ids)):
            raise StoryPlanError(
                "acting output contains duplicate brief ids",
                diagnostics=[{
                    "code": "duplicate_acting_brief",
                    "subject_id": "acting",
                    "message": "each requested brief must be returned at most once",
                }],
            )
        ids = set(output_ids)
        foreign = sorted(ids - set(expected))
        if foreign:
            raise StoryPlanError(
                "acting output contains foreign brief ids",
                diagnostics=[
                    {
                        "code": "foreign_acting_brief",
                        "subject_id": "acting",
                        "message": f"unexpected brief ids: {', '.join(foreign)}",
                    }
                ],
            )
        return dict(acting)

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
            "characters": [dict(character) for character in request.characters],
            "source_evidence": dict(request.source_evidence),
            "story_idea": request.story_idea,
            "locations": [dict(location) for location in request.locations],
            "props": [dict(prop) for prop in request.props],
            "guide": request.guide,
        }

    def _arc_skeleton_input(
        self, request: StoryPlanRequest, bible: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "song_title": request.song_title,
            "lyrics": request.lyrics,
            "narrative_bible": dict(bible),
            "characters": [dict(character) for character in request.characters],
            "locations": [dict(location) for location in request.locations],
            "props": [dict(prop) for prop in request.props],
            "terminal_window_seconds": request.terminal_window_seconds,
            "guide": request.guide,
        }

    def _allocation_input(
        self,
        request: StoryPlanRequest,
        bible: Mapping[str, Any],
        beats: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "song_title": request.song_title,
            "lyrics": request.lyrics,
            "beats": [dict(beat) for beat in beats],
            "segments": [segment.to_compact_dict() for segment in request.segments],
            "narrative_bible": dict(bible),
            "characters": [dict(character) for character in request.characters],
            "locations": [dict(location) for location in request.locations],
            "props": [dict(prop) for prop in request.props],
            "terminal_window_seconds": request.terminal_window_seconds,
            "guide": request.guide,
        }

    @staticmethod
    def _validate_arc(
        arc: Mapping[str, Any], diagnostics: list[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        """Validate the arc skeleton as a whole; return the beats list.

        Structural problems (missing beats, invalid phase) are hard fails;
        arc-order problems (no opening/resolution) are soft diagnostics so
        the bounded repair pass can fix them.
        """
        beats = arc.get("beats", arc.get("items")) if isinstance(arc, Mapping) else arc
        if not isinstance(beats, list) or not beats:
            raise StoryPlanError(
                "arc_skeleton output must contain a non-empty beats list",
                diagnostics=[
                    {
                        "code": "missing_beats",
                        "subject_id": "arc",
                        "message": "beats list missing or empty",
                    }
                ],
            )
        valid_phases = {"opening", "development", "climax", "resolution"}
        for index, beat in enumerate(beats):
            if not isinstance(beat, Mapping):
                raise StoryPlanError(
                    f"arc beat {index} must be an object",
                    diagnostics=[
                        {
                            "code": "invalid_arc_beat",
                            "subject_id": f"beat-{index:03d}",
                            "message": "arc beat is not an object",
                        }
                    ],
                )
            phase = str(beat.get("phase", "") or "")
            if phase not in valid_phases:
                raise StoryPlanError(
                    f"arc beat {index} has invalid phase {phase!r}",
                    diagnostics=[
                        {
                            "code": "invalid_arc_phase",
                            "subject_id": f"beat-{index:03d}",
                            "message": f"invalid phase {phase!r}",
                        }
                    ],
                )
        if str(beats[0].get("phase", "")) != "opening":
            diagnostics.append(
                {
                    "code": "arc_no_opening",
                    "subject_id": "arc",
                    "message": "first arc beat is not opening",
                }
            )
        if str(beats[-1].get("phase", "")) != "resolution":
            diagnostics.append(
                {
                    "code": "arc_no_resolution",
                    "subject_id": "arc",
                    "message": "last arc beat is not resolution",
                }
            )
        return [dict(beat) for beat in beats]

    def _acting_input(
        self, request: StoryPlanRequest, allocation: Mapping[str, Any]
    ) -> dict[str, Any]:
        briefs = allocation.get("briefs")
        if not isinstance(briefs, list):
            briefs = []
        segment_ids = {
            str(brief.get("segment_id"))
            for brief in briefs
            if isinstance(brief, Mapping) and brief.get("segment_id")
        }
        beat_indices = {
            int(index)
            for brief in briefs
            if isinstance(brief, Mapping)
            for index in brief.get("beat_indices", [])
            if isinstance(index, int) and not isinstance(index, bool)
        }
        typed_beats = [
            dict(beat)
            for index, beat in enumerate(allocation.get("typed_beats", []))
            if isinstance(beat, Mapping) and (not beat_indices or index in beat_indices)
        ]
        return {
            "song_title": request.song_title,
            "song_language": request.song_language,
            "lyrics": request.lyrics,
            "briefs": [dict(brief) for brief in briefs],
            "expected_brief_ids": [
                str(brief.get("brief_id", ""))
                for brief in briefs
                if isinstance(brief, Mapping)
            ],
            "typed_beats": typed_beats,
            "segments": [
                segment.to_compact_dict()
                for segment in request.segments
                if not segment_ids or segment.segment_id in segment_ids
            ],
            "locations": [dict(location) for location in request.locations],
            "props": [dict(prop) for prop in request.props],
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
    def _coerce_typed_allocation(
        request: StoryPlanRequest, allocation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Map a small model-authored beat sheet to service-owned bindings."""
        if "briefs" in allocation:
            return allocation
        raw_beats = allocation.get("beats")
        raw_allocations = allocation.get("brief_allocations")
        if not isinstance(raw_beats, list):
            return allocation
        # The model sometimes returns brief_allocations as a dict keyed by
        # target instead of the signature's list of {target, ...} objects.
        # Normalize both shapes to (target, item) pairs.
        if isinstance(raw_allocations, Mapping):
            entries = [
                (str(target), item) for target, item in raw_allocations.items()
            ]
        elif isinstance(raw_allocations, list):
            entries = [
                (str(item.get("target", "")), item)
                for item in raw_allocations
                if isinstance(item, Mapping)
            ]
        else:
            entries = []
        segments = {segment.segment_id: segment for segment in request.segments}
        briefs: list[dict[str, Any]] = []
        for target, item in entries:
            if not isinstance(item, Mapping):
                continue
            segment = segments.get(target)
            if segment is None:
                continue
            index = item.get("beat_index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            briefs.append({
                "brief_id": f"brief-{target}", "segment_id": target,
                "start_seconds": segment.start_seconds, "end_seconds": segment.end_seconds,
                "beat_indices": [index],
                "required": [f"beat-{value + 1:03d}" for value in item.get("required_beat_indices", [])],
                "forbidden": [f"beat-{value + 1:03d}" for value in item.get("forbidden_beat_indices", [])],
            })
        return {"briefs": briefs, "typed_beats": raw_beats}

    @staticmethod
    def _allocation_from_narrative_contract(
        request: StoryPlanRequest,
        raw_beats: list[Any],
        contract: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        location_ids = [
            str(item.get("id") if isinstance(item, Mapping) else item).strip()
            for item in contract.get("location_order", [])
        ]
        canonical_locations = {str(item.get("id", "")) for item in request.locations}
        location_ids = [item for item in location_ids if item in canonical_locations]
        if not location_ids:
            return None

        restricted = contract.get("actor_allowed_locations", {})
        restricted = restricted if isinstance(restricted, Mapping) else {}
        canonical_characters = [str(item.get("id", "")) for item in request.characters]
        restricted_ids = {str(actor_id) for actor_id in restricted}
        base_characters = [item for item in canonical_characters if item and item not in restricted_ids]
        terminal = contract.get("terminal_states", {})
        terminal = terminal if isinstance(terminal, Mapping) else {}

        typed_beats: list[dict[str, Any]] = []
        for index, location_id in enumerate(location_ids):
            source = raw_beats[min(index, len(raw_beats) - 1)]
            source = source if isinstance(source, Mapping) else {}
            allowed = [
                str(actor_id)
                for actor_id, locations in restricted.items()
                if location_id in {
                    str(entry.get("id") if isinstance(entry, Mapping) else entry)
                    for entry in (locations or [])
                }
            ]
            typed_beats.append({
                "phase": ("opening", "development", "climax", "resolution")[
                    min(index, 3)
                ],
                "description": str(source.get("description", f"Journey through {location_id}.")),
                "location_id": location_id,
                "character_ids": [*base_characters, *allowed],
            })

        bindings = contract.get("milestone_bindings", [])
        if not isinstance(bindings, list):
            raise StoryPlanError("narrative contract milestone bindings must be a list")
        milestone_ids = [
            str(item.get("id") if isinstance(item, Mapping) else item).strip()
            for item in contract.get("milestone_order", [])
        ]
        milestone_ids = [milestone for milestone in milestone_ids if milestone]
        binding_ids = [
            str(binding.get("milestone_id", "")).strip()
            for binding in bindings
            if isinstance(binding, Mapping)
        ]
        if milestone_ids and (
            len(binding_ids) != len(bindings)
            or len(binding_ids) != len(set(binding_ids))
            or set(binding_ids) != set(milestone_ids)
        ):
            raise StoryPlanError(
                "narrative contract milestone bindings must contain every milestone exactly once"
            )
        by_location: dict[int, list[tuple[str, float]]] = {index: [] for index in range(len(location_ids))}
        for binding in bindings:
            if not isinstance(binding, Mapping):
                return None
            milestone = str(binding.get("milestone_id", "")).strip()
            location_id = str(binding.get("location_id", "")).strip()
            if not milestone or location_id not in location_ids:
                return None
            try:
                relative_position = float(binding.get("relative_position"))
            except (TypeError, ValueError):
                return None
            if not 0.0 <= relative_position <= 1.0:
                return None
            by_location[location_ids.index(location_id)].append((milestone, relative_position))

        segment_count = len(request.segments)
        assignments: list[int] = [
            min(len(location_ids) - 1, position * len(location_ids) // max(1, segment_count))
            for position in range(segment_count)
        ]
        milestone_by_position: dict[int, str] = {}
        for location_index, items in by_location.items():
            positions = [i for i, value in enumerate(assignments) if value == location_index]
            items.sort(key=lambda item: milestone_ids.index(item[0]))
            if len(items) > len(positions):
                raise StoryPlanError(
                    f"narrative contract has {len(items)} milestones but only "
                    f"{len(positions)} scenes at location {location_ids[location_index]}"
                )
            next_available = 0
            for item_index, (milestone, relative_position) in enumerate(items):
                desired = round(relative_position * (len(positions) - 1))
                latest = len(positions) - (len(items) - item_index)
                chosen = min(max(desired, next_available), latest)
                milestone_by_position[positions[chosen]] = milestone
                next_available = chosen + 1

        briefs: list[dict[str, Any]] = []
        for position, segment in enumerate(request.segments):
            beat_index = assignments[position]
            milestone_id = milestone_by_position.get(position)
            characters = list(typed_beats[beat_index]["character_ids"])
            if milestone_id:
                for actor_id, rule in terminal.items():
                    if isinstance(rule, Mapping) and str(rule.get("milestone", "")) == milestone_id:
                        actor = str(actor_id)
                        if actor in canonical_characters and actor not in characters:
                            characters.append(actor)
            required_actor_states = [
                {"character_id": str(actor_id), "state": str(rule.get("state", "")).strip()}
                for actor_id, rule in terminal.items()
                if (
                    isinstance(rule, Mapping)
                    and str(rule.get("milestone", "")) == milestone_id
                    and str(actor_id) in canonical_characters
                    and str(rule.get("state", "")).strip()
                )
            ]
            briefs.append({
                "brief_id": f"brief-{segment.segment_id}",
                "segment_id": segment.segment_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "beat_indices": [beat_index],
                "required": [milestone_id] if milestone_id else [],
                "forbidden": [],
                "character_ids": characters,
                "location_id": typed_beats[beat_index]["location_id"],
                "milestone_id": milestone_id,
                "required_actor_states": required_actor_states,
            })
        return {"briefs": briefs, "typed_beats": typed_beats}

    @staticmethod
    def _lead_character_id(request: StoryPlanRequest) -> str:
        """The lead character id: the singer if present, else the first."""
        for character in request.characters:
            if isinstance(character, Mapping) and character.get("is_singer"):
                return str(character.get("id", "")).strip()
        for character in request.characters:
            if isinstance(character, Mapping):
                return str(character.get("id", "")).strip()
        return ""

    def _coerce_typed_acting(
        self,
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
        acting: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Bind typed acting briefs to service-owned brief ids by target."""
        if "briefs" not in acting or not isinstance(acting.get("briefs"), list):
            return acting
        by_target = {
            str(item.get("segment_id")): str(item.get("brief_id"))
            for item in allocation.get("briefs", []) if isinstance(item, Mapping)
        }
        beat_by_index = {
            index: item
            for index, item in enumerate(allocation.get("typed_beats", []))
            if isinstance(item, Mapping)
        }
        beat_index_by_target = {
            str(item.get("segment_id")): next(iter(item.get("beat_indices", [])), None)
            for item in allocation.get("briefs", [])
            if isinstance(item, Mapping)
        }
        if not any(isinstance(item, Mapping) and "target" in item for item in acting["briefs"]):
            return acting
        briefs = []
        for item in acting["briefs"]:
            if not isinstance(item, Mapping):
                continue
            target = str(item.get("target", ""))
            actor_states = list(item.get("actor_states", []))
            if not actor_states:
                # Prefer the brief's own character_ids, then the beat's, then
                # the lead character so instrumental segments always get a state.
                candidate_ids: list[str] = []
                for character_id in list(item.get("character_ids", [])):
                    value = str(character_id).strip()
                    if value and value not in candidate_ids:
                        candidate_ids.append(value)
                beat = beat_by_index.get(beat_index_by_target.get(target))
                for character_id in (beat or {}).get("character_ids", []):
                    value = str(character_id).strip()
                    if value and value not in candidate_ids:
                        candidate_ids.append(value)
                if not candidate_ids:
                    lead = self._lead_character_id(request)
                    if lead:
                        candidate_ids.append(lead)
                actor_states = [
                    {
                        "character_id": character_id,
                        "inner_state": "present",
                        "physical_state": "",
                    }
                    for character_id in candidate_ids
                ]
            briefs.append({
                "brief_id": by_target.get(target, target),
                "target": target,
                "character_ids": list(item.get("character_ids", [])),
                "location_id": item.get("location_id"),
                "prop_ids": list(item.get("prop_ids", [])),
                "vocal_presentation": item.get("vocal_presentation", "offscreen"),
                "visual_direction": item.get("visual_direction", ""),
                "exclusive": False,
                "beat_id": item.get("beat_id"),
                "objective": item.get("objective", ""),
                "emotional_turn": item.get("emotional_turn", ""),
                "actor_states": actor_states,
            })
        return {**dict(acting), "briefs": briefs}

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
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        if isinstance(raw, list):
            # Some jobs (e.g. arc_skeleton) return a bare list; wrap it so
            # the forbidden-key check and downstream access work uniformly.
            raw = {"items": raw}
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

    def _resolve_entities(
        self,
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[Mapping[str, Any]]]:
        """Resolve entity refs in beats/briefs to config entities (L2).

        For each entity ref, determines the resolution:
        - use: config has a complete matching entity (non-empty description)
        - extend: config has a partial entity (empty description) -> fill it
        - invent: no config entity -> invent, constrained to the story's world

        Returns (resolved_config, decisions) where resolved_config has
        'characters', 'locations', 'props' lists and decisions is a list
        of {entity_type, entity_id, decision, name}.
        """
        typed_beats = [
            beat for beat in allocation.get("typed_beats", [])
            if isinstance(beat, Mapping)
        ]
        briefs = [
            brief for brief in allocation.get("briefs", [])
            if isinstance(brief, Mapping)
        ]

        character_refs: set[str] = set()
        location_refs: set[str] = set()
        prop_refs: set[str] = set()
        for beat in typed_beats:
            for cid in beat.get("character_ids", []):
                character_refs.add(str(cid))
            if beat.get("location_id"):
                location_refs.add(str(beat["location_id"]))
            for pid in beat.get("prop_ids", []):
                prop_refs.add(str(pid))
        for brief in briefs:
            for cid in brief.get("character_ids", []):
                character_refs.add(str(cid))
            if brief.get("location_id"):
                location_refs.add(str(brief["location_id"]))
            for pid in brief.get("prop_ids", []):
                prop_refs.add(str(pid))

        def _config_map(
            items: tuple[Mapping[str, Any], ...]
        ) -> dict[str, dict[str, Any]]:
            return {
                str(item.get("id", "")).strip(): dict(item)
                for item in items
                if isinstance(item, Mapping)
                and str(item.get("id", "")).strip()
                and str(item.get("name", "")).strip()
            }

        config_characters = _config_map(request.characters)
        config_locations = _config_map(request.locations)
        config_props = _config_map(request.props)

        decisions: list[Mapping[str, Any]] = []

        def _resolve(
            entity_type: str,
            refs: set[str],
            config: dict[str, dict[str, Any]],
            is_singer: bool = False,
        ) -> list[dict[str, Any]]:
            resolved = [dict(entry) for entry in config.values()]
            for ref in sorted(refs):
                if ref in config:
                    entity = config[ref]
                    if entity.get("description"):
                        decisions.append(
                            {
                                "entity_type": entity_type,
                                "entity_id": ref,
                                "decision": "use",
                                "name": entity.get("name", ref),
                            }
                        )
                    else:
                        for entry in resolved:
                            if entry.get("id") == ref:
                                entry["description"] = self._generate_entity_description(
                                    entity_type, ref, request
                                )
                                break
                        decisions.append(
                            {
                                "entity_type": entity_type,
                                "entity_id": ref,
                                "decision": "extend",
                                "name": entity.get("name", ref),
                            }
                        )
                else:
                    name = self._derive_entity_name(entity_type, ref)
                    new_entity: dict[str, Any] = {
                        "id": ref,
                        "name": name,
                        "description": self._generate_entity_description(
                            entity_type, ref, request
                        ),
                    }
                    if is_singer:
                        new_entity["is_singer"] = False
                    resolved.append(new_entity)
                    decisions.append(
                        {
                            "entity_type": entity_type,
                            "entity_id": ref,
                            "decision": "invent",
                            "name": name,
                        }
                    )
            # The config dicts can carry extra keys (e.g. visual_description,
            # image_prompt) that the canonical models forbid.  Normalize to the
            # allowed fields so the candidate validates.
            allowed = {
                "character": ("id", "name", "description", "is_singer"),
                "location": ("id", "name", "description"),
                "prop": ("id", "name", "description"),
            }[entity_type]
            normalized: list[dict[str, Any]] = []
            for entry in resolved:
                clean: dict[str, Any] = {
                    key: entry[key] for key in allowed if key in entry
                }
                if entity_type == "character" and "is_singer" not in clean:
                    clean["is_singer"] = False
                normalized.append(clean)
            return normalized

        resolved_characters = _resolve("character", character_refs, config_characters, is_singer=True)
        resolved_locations = _resolve("location", location_refs, config_locations)
        resolved_props = _resolve("prop", prop_refs, config_props)

        return (
            {
                "characters": resolved_characters,
                "locations": resolved_locations,
                "props": resolved_props,
            },
            decisions,
        )

    @staticmethod
    def _derive_entity_name(entity_type: str, entity_id: str) -> str:
        """Derive a human-readable name from an entity ID."""
        name = entity_id.replace("_", " ").replace("-", " ").strip()
        return name.title() if name else entity_type.title()

    def _generate_entity_description(
        self, entity_type: str, entity_id: str, request: StoryPlanRequest
    ) -> str:
        """Generate a description for an entity, constrained to the story's world."""
        context = str(request.source_evidence.get("creative_direction", "")).strip()
        if context:
            return f"A {entity_type} in the story: {context[:200]}"
        return f"A {entity_type} in the music video for {request.song_title}"

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
            "story plan validation failed after deterministic assembly",
            diagnostics=[
                {
                    "code": str(error.get("code", "validation_error")),
                    "subject_id": str(error.get("subject_id", "")),
                    "message": str(error.get("message", "")),
                }
                for error in diagnostics
            ],
        ) from exc

    @staticmethod
    def _describe_arc(allocation: Mapping[str, Any]) -> str:
        """Summarize the beat count and arc phase sequence for observability."""
        beats = allocation.get("typed_beats")
        if not isinstance(beats, list):
            return "0 beats, arc: (none)"
        phases: list[str] = []
        for beat in beats:
            if isinstance(beat, Mapping):
                phase = str(beat.get("phase", "") or "")
                if phase:
                    phases.append(phase)
        return f"{len(phases)} beats, arc: {' -> '.join(phases) or '(none)'}"

    @staticmethod
    def _assemble_candidate(
        request: StoryPlanRequest,
        allocation: Mapping[str, Any],
        acting: Mapping[str, Any],
        resolved_config: Mapping[str, Any] | None = None,
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
        if resolved_config is not None:
            characters = [
                dict(entry) for entry in resolved_config.get("characters", [])
                if isinstance(entry, Mapping)
            ]
            locations = [
                dict(entry) for entry in resolved_config.get("locations", [])
                if isinstance(entry, Mapping)
            ]
            props = [
                dict(entry) for entry in resolved_config.get("props", [])
                if isinstance(entry, Mapping)
            ]
        else:
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
            locations = [
                {
                    "id": str(location.get("id", "")).strip(),
                    "name": str(location.get("name", "")).strip(),
                    "description": str(location.get("description", "")),
                }
                for location in request.locations
                if isinstance(location, Mapping)
                and str(location.get("id", "")).strip()
                and str(location.get("name", "")).strip()
            ]
            props = [
                {
                    "id": str(prop.get("id", "")).strip(),
                    "name": str(prop.get("name", "")).strip(),
                    "description": str(prop.get("description", "")),
                }
                for prop in request.props
                if isinstance(prop, Mapping)
                and str(prop.get("id", "")).strip()
                and str(prop.get("name", "")).strip()
            ]
        segments: list[dict[str, Any]] = []
        beats = [
            {
                "id": f"beat-{index:03d}",
                "phase": str(item.get("phase", "development")),
                "description": str(item.get("description", "")),
                "character_ids": list(item.get("character_ids", [])),
                "location_id": item.get("location_id"),
                "prop_ids": list(item.get("prop_ids", [])),
            }
            for index, item in enumerate(allocation.get("typed_beats", []), start=1)
            if isinstance(item, Mapping)
        ]
        for beat, item in zip(
            beats,
            (item for item in allocation.get("typed_beats", []) if isinstance(item, Mapping)),
        ):
            if item.get("milestone_id"):
                beat["milestone_id"] = str(item["milestone_id"])
        beat_by_index = {
            index: item
            for index, item in enumerate(allocation.get("typed_beats", []))
            if isinstance(item, Mapping)
        }
        raw_briefs = allocation.get("briefs")
        if not isinstance(raw_briefs, list):
            raw_briefs = []
        canonical_character_ids = {character["id"] for character in characters}
        contract = request.source_evidence.get("narrative_contract") or {}
        if not isinstance(contract, Mapping):
            contract = {}
        terminal_rules = contract.get("terminal_states") or {}
        active_terminal_states: dict[str, str] = {}
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
            acting_by_id = {
                str(item.get("brief_id")): item
                for item in acting.get("briefs", [])
                if isinstance(item, Mapping)
            }
            acting_brief = acting_by_id.get(brief_id, {})
            beat_index = next(iter(raw.get("beat_indices", [])), None)
            binding = beat_by_index.get(beat_index, {}) if isinstance(beat_index, int) else {}
            # Narrative bindings come from allocation, never from the creative
            # acting response. Acting may only fill creative fields.
            bound_character_ids = list(raw.get("character_ids", binding.get("character_ids", [])))
            bound_character_set = set(bound_character_ids)
            cast_is_locked = bool(bound_character_set) or (
                bool(contract.get("location_order")) and "character_ids" in raw
            )
            bound_prop_ids = list(binding.get("prop_ids", []))
            bound_location_id = raw.get("location_id", binding.get("location_id"))
            bound_milestone_id = raw.get("milestone_id") or binding.get("milestone_id")
            actor_states_by_id = {
                str(state.get("character_id", "")): {
                    "character_id": str(state.get("character_id", "")),
                    "state": str(
                        state.get("state")
                        or state.get("inner_state")
                        or "present"
                    ),
                    "physical_state": str(state.get("physical_state", "")),
                }
                for state in acting_brief.get("actor_states", [])
                if (
                    isinstance(state, Mapping)
                    and str(state.get("character_id", "")) in canonical_character_ids
                    and (not cast_is_locked or str(state.get("character_id", "")) in bound_character_set)
                )
            }
            if isinstance(terminal_rules, Mapping):
                for actor_id, rule in terminal_rules.items():
                    if not isinstance(rule, Mapping):
                        continue
                    if rule.get("reset_event") and bound_milestone_id == rule["reset_event"]:
                        active_terminal_states.pop(str(actor_id), None)
                    if bound_milestone_id == rule.get("milestone"):
                        required_state = str(rule.get("state") or "").strip()
                        if required_state:
                            active_terminal_states[str(actor_id)] = required_state
            for actor_id, state in active_terminal_states.items():
                if not cast_is_locked or actor_id in bound_character_set:
                    actor_states_by_id[actor_id] = {
                        "character_id": actor_id,
                        "state": state,
                        "physical_state": "",
                    }
            for state in raw.get("required_actor_states", []):
                if not isinstance(state, Mapping):
                    continue
                actor_id = str(state.get("character_id", ""))
                required_state = str(state.get("state", "")).strip()
                if required_state and (not cast_is_locked or actor_id in bound_character_set):
                    actor_states_by_id[actor_id] = {
                        "character_id": actor_id,
                        "state": required_state,
                        "physical_state": "",
                    }
            segment = {
                    "id": brief_id,
                    "target": target,
                    "beat_id": (
                        f"beat-{raw['beat_indices'][0] + 1:03d}"
                        if beats and raw.get("beat_indices")
                        else None
                    ),
                    "character_ids": (
                        bound_character_ids
                        if binding
                        else list(acting_brief.get("character_ids", []))
                    ),
                    "location_id": (
                        bound_location_id
                        if binding
                        else acting_brief.get("location_id")
                    ),
                    "prop_ids": (
                        bound_prop_ids
                        if binding
                        else list(acting_brief.get("prop_ids", []))
                    ),
                    "vocal_presentation": str(acting_brief.get("vocal_presentation", "offscreen")),
                    "visual_direction": str(acting_brief.get("visual_direction", "")),
                    "objective": str(acting_brief.get("objective", "")),
                    "emotional_turn": str(acting_brief.get("emotional_turn", "")),
                    "actor_states": list(actor_states_by_id.values()),
                    "exclusive": False,
                    "audio_ref": {"segment_id": target, "fingerprint": fingerprint},
                }
            if bound_milestone_id:
                segment["milestone_id"] = str(bound_milestone_id)
            segments.append(segment)
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
            "locations": locations,
            "props": props,
            "beats": beats,
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

    def _live_prompts_input(
        self,
        request: StoryPlanRequest,
        plan: StoryPlan,
        bible: Mapping[str, Any],
    ) -> dict[str, Any]:
        briefs = [
            {
                "target": brief.target,
                "visual_direction": brief.visual_direction,
                "character_ids": list(brief.character_ids),
                "location_id": brief.location_id,
                "prop_ids": list(brief.prop_ids),
            }
            for brief in plan.segments
        ]
        creative_direction = str(
            request.source_evidence.get("creative_direction", "")
        ).strip()
        return {
            "creative_direction": creative_direction,
            "bible": dict(bible),
            "briefs": briefs,
            "expected_targets": [brief.target for brief in plan.segments],
        }

    def _build_live_prompts(
        self,
        request: StoryPlanRequest,
        plan: StoryPlan,
        bible: Mapping[str, Any],
        diagnostics: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """L3: live per-scene image + video prompts (soft-fail).

        Live prompts are an additive layer: a missing, malformed, or
        unvalidated prompt set is a diagnostic, never a hard failure, so an
        unconfigured or failing prompt job never blocks plan production.
        """
        self._reporter.step("story-plan-live-prompts")
        try:
            raw = self._job(
                "live_prompts", self._live_prompts_input(request, plan, bible)
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "code": "live_prompts_failed",
                    "subject_id": "live_prompts",
                    "message": f"live prompt job failed: {exc}",
                }
            )
            self._reporter.message("story-plan-live-prompts failed; continuing")
            return {}
        prompts = self._normalize_live_prompts(raw, plan, diagnostics)
        self._reporter.message(
            "story-plan-live-prompts complete: "
            f"{len(prompts)} scene prompt(s)"
        )
        return prompts

    @staticmethod
    def _normalize_live_prompts(
        raw: Any, plan: StoryPlan, diagnostics: list[Mapping[str, Any]]
    ) -> dict[str, dict[str, str]]:
        """Normalize the live-prompt job output keyed by segment target.

        Accepts a ``{prompts: [...]}`` mapping or a bare list; drops entries
        whose target is not a supplied segment or whose image_prompt is empty.
        """
        items = raw.get("prompts") if isinstance(raw, Mapping) else raw
        if not isinstance(items, list):
            items = []
        expected = {brief.target for brief in plan.segments}
        result: dict[str, dict[str, str]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            target = str(item.get("target", ""))
            image_prompt = str(item.get("image_prompt", "") or "").strip()
            if target not in expected or not image_prompt:
                continue
            result[target] = {
                "image_prompt": image_prompt,
                "video_prompt": str(item.get("video_prompt", "") or "").strip(),
            }
        if not result:
            diagnostics.append(
                {
                    "code": "live_prompts_empty",
                    "subject_id": "live_prompts",
                    "message": "no valid live prompts returned",
                }
            )
        return result


def compute_source_fingerprint(
    *,
    song_title: str,
    song_language: str,
    song_style: str,
    lyrics: str,
    sections: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    characters: Sequence[Mapping[str, Any]],
    creative_direction: str = "",
    story_idea: str = "",
    locations: Sequence[Mapping[str, Any]] = (),
    props: Sequence[Mapping[str, Any]] = (),
    narrative_contract: Mapping[str, Any] | None = None,
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
        "creative_direction": creative_direction,
        "story_idea": story_idea,
        "locations": [dict(location) for location in locations],
        "props": [dict(prop) for prop in props],
        "narrative_contract": dict(narrative_contract or {}),
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
