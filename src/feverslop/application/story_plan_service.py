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
from dataclasses import dataclass
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

        self._reporter.step("Story plan - story arc")
        self._reporter.message(
            "[cyan]Writing a compact beat sheet; scene timing will be bound deterministically…[/cyan]"
        )
        allocation_raw = self._reporter.run_progress(
            "Story plan - shaping the arc",
            lambda: self._job("beat_allocation", self._allocation_input(request, bible)),
        )
        allocation = self._normalize_job_output(
            "beat_allocation", allocation_raw, diagnostics, _FORBIDDEN_RENDER_KEYS
        )
        allocation = self._coerce_typed_allocation(request, allocation)
        self._validate_allocation(request, allocation, diagnostics)
        self._reporter.table(
            "Story arc - model-authored beats",
            ["#", "Phase", "What is happening"],
            [
                [str(index + 1), str(beat.get("phase", "development")), str(beat.get("description", ""))]
                for index, beat in enumerate(allocation.get("typed_beats", []))
                if isinstance(beat, Mapping)
            ],
        )
        self._reporter.message(
            f"[green]Bound {len(allocation.get('briefs', []))} scenes to the arc deterministically.[/green]"
        )
        window_rows: list[list[str]] = []
        for index, beat in enumerate(allocation.get("typed_beats", []), start=0):
            if not isinstance(beat, Mapping):
                continue
            bound = [
                brief for brief in allocation.get("briefs", [])
                if isinstance(brief, Mapping) and index in brief.get("beat_indices", [])
            ]
            if not bound:
                continue
            milestones = [
                str(brief["milestone_id"])
                for brief in bound if brief.get("milestone_id")
            ]
            window_rows.append([
                str(beat.get("location_id", f"beat-{index + 1}")),
                f"{bound[0].get('segment_id')} – {bound[-1].get('segment_id')}",
                ", ".join(milestones) or "—",
            ])
        self._reporter.table(
            "Story plan - locked windows",
            ["Location", "Scenes", "Milestones"],
            window_rows,
        )

        self._reporter.step("Story plan - acting beats")
        acting = self._build_acting_in_batches(request, allocation, diagnostics)
        self._validate_acting(allocation, acting, diagnostics)
        self._reporter.message("[green]Story plan acting complete.[/green]")

        candidate = self._assemble_candidate(request, allocation, acting)
        try:
            self._validate_payload(candidate, diagnostics)
        except StoryPlanValidationError as exc:
            validation_errors = list(exc.errors)
        else:
            validation_errors = []

        if validation_errors:
            self._reporter.warning(
                "The locally assembled plan did not satisfy its contract. "
                "No full-plan LLM repair was attempted; the diagnostics identify the failing boundary.",
                title="Story plan validation",
            )
            self._raise_validation_failure(
                StoryPlanValidationError(
                    "story plan candidate failed deterministic validation",
                    errors=validation_errors,
                ),
                diagnostics,
            )

        plan = StoryPlan.model_validate(candidate, strict=False)
        self._reporter.table(
            "Story plan locked",
            ["Scenes", "Beats", "Acting briefs"],
            [[str(len(plan.segments)), str(len(plan.beats)), str(len(acting.get("briefs", [])))]],
        )
        return StoryPlanResult(
            plan=plan,
            acting=self._typed_acting(acting, plan),
            diagnostics=tuple(diagnostics),
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
        return merged

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
        acting = self._coerce_typed_acting(batch_allocation, acting)
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

    def _allocation_input(
        self, request: StoryPlanRequest, bible: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "song_title": request.song_title,
            "lyrics": request.lyrics,
            "segment_count": len(request.segments),
            "narrative_bible": dict(bible),
            "characters": [dict(character) for character in request.characters],
            "locations": [dict(location) for location in request.locations],
            "props": [dict(prop) for prop in request.props],
            "terminal_window_seconds": request.terminal_window_seconds,
            "guide": request.guide,
        }

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

    # -- normalization and validation ---------------------------------------

    @staticmethod
    def _coerce_typed_allocation(
        request: StoryPlanRequest, allocation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Map a small model-authored beat sheet to service-owned bindings."""
        if "briefs" in allocation:
            return allocation
        raw_beats = allocation.get("beats")
        if not isinstance(raw_beats, list) or not raw_beats:
            return allocation
        contract = request.source_evidence.get("narrative_contract", {})
        if isinstance(contract, Mapping):
            contract_allocation = StoryPlanService._allocation_from_narrative_contract(
                request, raw_beats, contract,
            )
            if contract_allocation is not None:
                return contract_allocation
        typed_beats = [dict(item) for item in raw_beats if isinstance(item, Mapping)]
        if not typed_beats:
            return allocation
        for index, beat in enumerate(typed_beats):
            phase = str(beat.get("phase", "development")).strip().lower()
            beat["phase"] = "resolution" if index == len(typed_beats) - 1 else (
                phase if phase in {"opening", "development", "climax"} else "development"
            )

        terminal_segments = [
            segment for segment in request.segments
            if segment.end_seconds > request.terminal_window_seconds
        ]
        terminal_ids = {segment.segment_id for segment in terminal_segments}
        preterminal_segments = [
            segment for segment in request.segments if segment.segment_id not in terminal_ids
        ]
        nonterminal_beat_count = max(1, len(typed_beats) - 1)
        briefs: list[dict[str, Any]] = []
        for position, segment in enumerate(request.segments):
            if segment.segment_id in terminal_ids:
                index = len(typed_beats) - 1
            else:
                preterminal_position = preterminal_segments.index(segment)
                index = min(
                    nonterminal_beat_count - 1,
                    (preterminal_position * nonterminal_beat_count) // max(1, len(preterminal_segments)),
                )
            briefs.append({
                "brief_id": f"brief-{segment.segment_id}", "segment_id": segment.segment_id,
                "start_seconds": segment.start_seconds, "end_seconds": segment.end_seconds,
                "beat_indices": [index],
                "required": [f"beat-{index + 1:03d}"],
                "forbidden": [],
            })
        return {"briefs": briefs, "typed_beats": typed_beats}

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
        if len(location_ids) < 2:
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

        milestones = [
            str(item.get("id") if isinstance(item, Mapping) else item).strip()
            for item in contract.get("milestone_order", [])
        ]
        milestones = [item for item in milestones if item]
        by_location: dict[int, list[str]] = {index: [] for index in range(len(location_ids))}
        for milestone in milestones:
            lower = milestone.lower()
            location_index = next(
                (index for index, location_id in enumerate(location_ids) if location_id.lower() in lower),
                None,
            )
            if location_index is None:
                if "lich" in lower:
                    location_index = next((i for i, item in enumerate(location_ids) if "lich" in item.lower()), 0)
                elif "dragon" in lower:
                    location_index = next((i for i, item in enumerate(location_ids) if "dragon" in item.lower()), 0)
                elif any(token in lower for token in ("fountain", "water", "ascen", "transfig")):
                    location_index = len(location_ids) - 1
                else:
                    location_index = 0
            by_location[location_index].append(milestone)

        segment_count = len(request.segments)
        assignments: list[int] = [
            min(len(location_ids) - 1, position * len(location_ids) // max(1, segment_count))
            for position in range(segment_count)
        ]
        milestone_by_position: dict[int, str] = {}
        for location_index, items in by_location.items():
            positions = [i for i, value in enumerate(assignments) if value == location_index]
            for item_index, milestone in enumerate(items):
                if not positions:
                    continue
                target = positions[round(item_index * (len(positions) - 1) / max(1, len(items) - 1))]
                milestone_by_position[target] = milestone

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
            })
        return {"briefs": briefs, "typed_beats": typed_beats}

    @staticmethod
    def _coerce_typed_acting(allocation: Mapping[str, Any], acting: Mapping[str, Any]) -> Mapping[str, Any]:
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
                beat = beat_by_index.get(beat_index_by_target.get(target))
                actor_states = [
                    {
                        "character_id": str(character_id),
                        "inner_state": "present",
                        "physical_state": "",
                    }
                    for character_id in (beat or {}).get("character_ids", [])
                    if str(character_id).strip()
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
            "story plan validation failed after deterministic assembly",
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
            bound_prop_ids = list(binding.get("prop_ids", []))
            bound_location_id = raw.get("location_id", binding.get("location_id"))
            bound_milestone_id = raw.get("milestone_id") or binding.get("milestone_id")
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
                    "actor_states": [
                        {
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
                            and str(state.get("character_id", ""))
                            in canonical_character_ids
                        )
                    ],
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
