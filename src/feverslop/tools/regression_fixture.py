"""Load and evaluate deterministic failed-run regression fixtures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.domain.artifact_hash import sha256_file


SCHEMA = "feverslop.failed-run-regression/v1"
INVARIANT_GROUPS = (
    "vocal_delivery",
    "milestone_uniqueness",
    "chronology",
    "adjacent_continuity",
)


@dataclass(frozen=True)
class InvariantGroupResult:
    name: str
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class RegressionInvariantReport:
    groups: dict[str, InvariantGroupResult]

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            failure
            for group in self.groups.values()
            for failure in group.failures
        )

    @property
    def passed(self) -> bool:
        return not self.failures


def load_regression_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read regression fixture: {fixture_path}") from exc

    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported regression fixture schema: {payload.get('schema')!r}")
    projections = payload.get("projections")
    if not isinstance(projections, dict) or not projections:
        raise ValueError("regression fixture requires projections")
    for name, projection in projections.items():
        if not isinstance(projection, dict):
            raise ValueError(f"projection {name!r} must be an object")
        scenes = projection.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"projection {name!r} requires scenes")
        numbers = [scene.get("scene") for scene in scenes if isinstance(scene, dict)]
        if len(numbers) != len(scenes) or numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            raise ValueError(f"projection {name!r} scenes must have unique ascending numbers")
    for source in (payload.get("provenance") or {}).get("baseline_files") or ():
        source_path = Path(str(source.get("path") or ""))
        digest = str(source.get("sha256") or "")
        if not source_path.parts or source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("baseline provenance paths must be portable and relative")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid baseline SHA-256: {source_path.as_posix()}")
    return payload


def verify_baseline_provenance(
    fixture: dict[str, Any], repo_root: str | Path,
) -> tuple[Path, ...]:
    """Verify the preserved baseline when its ignored project is available."""
    root = Path(repo_root).resolve()
    provenance = fixture.get("provenance") or {}
    source_project = root / str(provenance.get("source_project") or "")
    if not source_project.is_dir():
        return ()

    checked: list[Path] = []
    for source in provenance.get("baseline_files") or ():
        relative = Path(str(source["path"]))
        path = root / relative
        if not path.is_file():
            raise ValueError(f"preserved regression baseline file is missing: {relative.as_posix()}")
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"preserved regression baseline hash mismatch: {relative.as_posix()}")
        checked.append(relative)
    return tuple(checked)


def evaluate_regression_invariants(
    fixture: dict[str, Any],
    projection_name: str,
    *,
    prepared_request: dict[str, Any] | None = None,
) -> RegressionInvariantReport:
    projections = fixture.get("projections") or {}
    if projection_name not in projections:
        raise ValueError(f"unknown regression projection: {projection_name}")
    projection = projections[projection_name]
    contract = fixture.get("invariant_contract") or {}
    failures = {
        "vocal_delivery": _vocal_delivery_failures(
            projection.get("vocal_case") or {}, prepared_request=prepared_request,
        ),
        "milestone_uniqueness": _milestone_failures(projection, contract),
        "chronology": _chronology_failures(projection, contract),
        "adjacent_continuity": _continuity_failures(projection, contract),
    }
    return RegressionInvariantReport({
        name: InvariantGroupResult(name=name, failures=tuple(failures[name]))
        for name in INVARIANT_GROUPS
    })


def _vocal_delivery_failures(
    case: dict[str, Any],
    *,
    prepared_request: dict[str, Any] | None,
) -> list[str]:
    evidence = case.get("timeline_evidence") or {}
    request = prepared_request or case.get("final_request") or {}
    if evidence.get("transcript_status") != "accepted" or not evidence.get("word_timestamps"):
        return []

    scene = int(case.get("scene") or 0)
    prefix = f"scene_{scene:03d}"
    failures: list[str] = []
    prompt = str(request.get("prompt") or "")
    lowered = prompt.casefold()
    has_singing = re.search(r"\b(sing|sings|singing|sung)\b", lowered) is not None
    has_lip_sync = "lip sync" in lowered or "lip-sync" in lowered
    contradictions = (
        "no sung vocal performance",
        "no vocal performance",
        "does not sing",
        "do not create lip-sync",
        "mouth closed",
    )
    if not has_singing or not has_lip_sync or any(item in lowered for item in contradictions):
        failures.append(
            f"{prefix}.h3.prompt: accepted timed vocal evidence did not survive as an "
            "uncontradicted singing and lip-sync directive"
        )

    performer_id = str(evidence.get("performer_id") or "")
    speaker_id = str(evidence.get("speaker_id") or "")
    binding = request.get("audio_subject_binding") or {}
    if binding.get("subject_id") != performer_id or binding.get("speaker_id") != speaker_id:
        failures.append(
            f"{prefix}.request.audio_subject_binding: expected {performer_id}/{speaker_id}"
        )

    audio_inputs = request.get("audio_inputs") or []
    vocals = next((item for item in audio_inputs if item.get("name") == "vocals"), None)
    if vocals is None or "conditioning" not in (vocals.get("roles") or []):
        failures.append(
            f"{prefix}.request.audio_inputs.vocals: expected a conditioning vocal guide"
        )
    return failures


def _milestone_failures(projection: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    one_shot = set(contract.get("one_shot_milestones") or ())
    seen: dict[str, int] = {}
    failures: list[str] = []
    previous_prop_ranks: dict[str, tuple[int, int]] = {}
    prop_orders = contract.get("prop_state_order") or {}

    for scene in projection.get("scenes") or ():
        number = int(scene["scene"])
        narrative = scene.get("narrative") or {}
        reset_events = set(narrative.get("reset_events") or ())
        for milestone in narrative.get("milestones") or ():
            if milestone in one_shot and milestone in seen and milestone not in reset_events:
                failures.append(
                    f"scene_{number:03d}.narrative.milestones: one-shot milestone "
                    f"{milestone!r} repeats scene_{seen[milestone]:03d} without a reset event"
                )
            seen.setdefault(milestone, number)
        for prop, state in (narrative.get("props") or {}).items():
            ordered_states = prop_orders.get(prop) or []
            if state not in ordered_states:
                continue
            rank = ordered_states.index(state)
            previous = previous_prop_ranks.get(prop)
            if previous is not None and rank < previous[0] and prop not in reset_events:
                failures.append(
                    f"scene_{number:03d}.narrative.props.{prop}: state {state!r} regresses "
                    f"from scene_{previous[1]:03d} without a reset event"
                )
            previous_prop_ranks[prop] = (rank, number)
    return failures


def _chronology_failures(projection: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    previous_rank: tuple[int, int] | None = None
    active_terminal_states: dict[str, tuple[str, int]] = {}
    terminal_contract = contract.get("terminal_states") or {}

    for scene in projection.get("scenes") or ():
        number = int(scene["scene"])
        narrative = scene.get("narrative") or {}
        rank = int(narrative.get("milestone_rank", -1))
        if previous_rank is not None and rank < previous_rank[0]:
            failures.append(
                f"scene_{number:03d}.narrative.milestone_rank: rank {rank} reverses "
                f"scene_{previous_rank[1]:03d} rank {previous_rank[0]}"
            )
        previous_rank = (rank, number)

        milestones = set(narrative.get("milestones") or ())
        cast_states = narrative.get("cast_states") or {}
        causal_events = set(narrative.get("causal_events") or ())
        for actor, rule in terminal_contract.items():
            if rule.get("milestone") in milestones:
                active_terminal_states[actor] = (str(rule.get("state") or ""), number)
            if actor not in active_terminal_states or actor not in cast_states:
                continue
            required_state, terminal_scene = active_terminal_states[actor]
            if cast_states[actor] != required_state and rule.get("reset_event") not in causal_events:
                failures.append(
                    f"scene_{number:03d}.narrative.cast_states.{actor}: state "
                    f"{cast_states[actor]!r} reverses terminal state {required_state!r} "
                    f"from scene_{terminal_scene:03d}"
                )
    return failures


def _continuity_failures(projection: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    actor_locations = contract.get("actor_allowed_locations") or {}
    for boundary in projection.get("boundaries") or ():
        previous = int(boundary["from_scene"])
        current = int(boundary["to_scene"])
        outgoing = boundary.get("outgoing") or {}
        incoming = boundary.get("incoming") or {}
        transition_events = set(boundary.get("transition_events") or ())

        for prop, outgoing_state in (outgoing.get("props") or {}).items():
            incoming_state = (incoming.get("props") or {}).get(prop)
            if incoming_state != outgoing_state and f"{prop}:{outgoing_state}->{incoming_state}" not in transition_events:
                failures.append(
                    f"scene_{current:03d}.incoming.props.{prop}: {incoming_state!r} is "
                    f"incompatible with scene_{previous:03d} outgoing state {outgoing_state!r}"
                )

        location = str(incoming.get("location") or "")
        for actor, state in (incoming.get("cast_states") or {}).items():
            allowed = actor_locations.get(actor)
            if state != "absent" and allowed and location not in allowed:
                failures.append(
                    f"scene_{current:03d}.incoming.cast_states.{actor}: actor is not allowed "
                    f"in location {location!r}"
                )

        if boundary.get("continuation_required"):
            request = boundary.get("request") or {}
            required = {
                "continuation_intents": request.get("continuation_intents"),
                "canonical_dependencies": request.get("canonical_dependencies"),
                "startframe_mode": request.get("startframe_mode"),
                "startframe_source_scene": request.get("startframe_source_scene"),
            }
            for field, value in required.items():
                if value is None or value == [] or value == "":
                    failures.append(
                        f"scene_{current:03d}.request.{field}: required continuation from "
                        f"scene_{previous:03d} is missing"
                    )
    return failures
