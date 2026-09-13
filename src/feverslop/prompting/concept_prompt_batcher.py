from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import LLMPort
from feverslop.prompting.music_video_modules import MusicVideoPromptModules
from feverslop.prompting.planning_payload import compact_planning_payload


def chunked(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


class ConceptPromptBatcher:
    """Robust concept-prompt generation for many music-video scenes.

    Why:
    Large single-call JSON generation often drops late keys like segment_040.
    This class generates concepts in smaller batches and validates that every
    stage1 segment receives exactly one concept.

    Continuity strategy:
    Every batch receives:
    - full story idea
    - global context
    - project steering
    - previous generated concepts summary
    - previous few concepts verbatim

    This keeps scene progression continuous while reducing JSON failure risk.
    """

    def __init__(
        self,
        llm: LLMPort,
        batch_size: int = 10,
        max_previous_concepts: int = 6,
        request_timeout_seconds: float | None = None,
        progress_callback: Callable[[str], None] | None = None,
        prompt_modules: MusicVideoPromptModules | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.llm = llm
        self.batch_size = batch_size
        self.max_previous_concepts = max_previous_concepts
        self.request_timeout_seconds = request_timeout_seconds
        self.progress_callback = progress_callback
        self.prompt_modules = prompt_modules or MusicVideoPromptModules(llm)

    def create_concept_prompts_batched(
        self,
        *,
        stage1_segments: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        all_results: dict[str, str] = {}
        previous_summary = ""
        batches = list(chunked(stage1_segments, self.batch_size))
        report = progress_callback or self.progress_callback

        for batch_number, (batch_start, batch) in enumerate(batches, start=1):
            batch_label = f"Concept batch {batch_number}/{len(batches)}"
            self._report(
                f"{batch_label}: generating scenes "
                f"{batch_start + 1}-{batch_start + len(batch)}",
                report,
            )
            batch_result = self._generate_batch(
                batch_index=batch_start // self.batch_size + 1,
                batch=batch,
                story_idea=story_idea,
                global_context=global_context,
                notes=notes,
                previous_concepts=self._last_concepts(all_results),
                previous_summary=previous_summary,
            )
            self._report(f"{batch_label}: response received, validating keys", report)

            expected_ids = [seg["segment_id"] for seg in batch]
            batch_result = self._repair_missing_or_extra_keys(
                expected_ids=expected_ids,
                result=batch_result,
                batch=batch,
                story_idea=story_idea,
                global_context=global_context,
                notes=notes,
                previous_concepts=self._last_concepts(all_results),
                previous_accepted_concepts=all_results,
                previous_summary=previous_summary,
                progress_callback=report,
            )

            all_results.update(batch_result)

            previous_summary = self._summarize_progress(
                story_idea=story_idea,
                global_context=global_context,
                concepts=all_results,
            )
            self._report(f"{batch_label}: complete ({len(all_results)} scenes total)", report)

        missing = [
            seg["segment_id"]
            for seg in stage1_segments
            if seg["segment_id"] not in all_results
        ]

        if missing:
            raise ValueError(f"Missing concept prompts after batched generation: {missing}")

        # Preserve stage1 order in output JSON.
        return {
            seg["segment_id"]: all_results[seg["segment_id"]]
            for seg in stage1_segments
        }

    def _report(self, message: str, callback: Callable[[str], None] | None = None) -> None:
        callback = callback or self.progress_callback
        if callback is not None:
            callback(message)

    def _generate_batch(
        self,
        *,
        batch_index: int,
        batch: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str,
        previous_concepts: dict,
        previous_summary: str,
    ) -> dict:
        payload = {
            "BATCH_INDEX": batch_index,
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "NOTES": notes,
            "PREVIOUS_PROGRESS_SUMMARY": previous_summary,
            "PREVIOUS_CONCEPTS": previous_concepts,
            "CURRENT_BATCH_SEGMENTS": compact_planning_payload(batch),
        }

        response = self.prompt_modules.concepts(
            payload, batch=True,
            silent_mode=bool(global_context.get("silent_mode", False)),
            timeout=self.request_timeout_seconds,
        )
        return response if isinstance(response, dict) else extract_json_object(str(response))

    def _repair_missing_or_extra_keys(
        self,
        *,
        expected_ids: list[str],
        result: dict,
        batch: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str,
        previous_concepts: dict,
        previous_accepted_concepts: dict,
        previous_summary: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        # Drop unexpected keys.
        repaired = {
            key: value
            for key, value in result.items()
            if key in expected_ids
        }

        missing = [segment_id for segment_id in expected_ids if segment_id not in repaired]
        invalid = self._invalid_concepts(
            repaired,
            global_context,
            previous_concepts=previous_accepted_concepts,
            expected_ids=expected_ids,
        )
        repair_ids = list(dict.fromkeys(missing + [item["segment_id"] for item in invalid]))

        if not repair_ids:
            return self._annotate_semantic_validation(
                {
                    segment_id: repaired[segment_id]
                    for segment_id in expected_ids
                },
                previous_concepts=previous_accepted_concepts,
                contract=_narrative_contract(global_context),
            )

        self._report(
            f"Concept batch: repairing {len(repair_ids)} missing or invalid scene "
            f"{'key' if len(repair_ids) == 1 else 'keys'}: {', '.join(repair_ids)}",
            progress_callback,
        )

        # One focused repair call for missing keys only.
        missing_segments = [
            seg
            for seg in batch
            if seg["segment_id"] in repair_ids
        ]

        payload = {
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "NOTES": notes,
            "PREVIOUS_PROGRESS_SUMMARY": previous_summary,
            "PREVIOUS_CONCEPTS": previous_concepts,
            "MISSING_SEGMENTS": compact_planning_payload(missing_segments),
            "INVALID_SEGMENTS": invalid,
            "EXPECTED_KEYS": repair_ids,
        }

        response = self.prompt_modules.repair_concepts(
            payload,
            timeout=self.request_timeout_seconds,
        )
        repair = response if isinstance(response, dict) else extract_json_object(str(response))

        for segment_id in repair_ids:
            value = repair.get(segment_id)
            if value is None:
                value = self._fallback_concept(segment_id, global_context) if segment_id in missing else repaired[segment_id]
            repaired[segment_id] = value

        ordered = {
            segment_id: repaired[segment_id]
            for segment_id in expected_ids
        }
        remaining_invalid = self._invalid_concepts(
            ordered,
            global_context,
            previous_concepts=previous_accepted_concepts,
            expected_ids=expected_ids,
        )
        if remaining_invalid:
            details = "; ".join(
                f"{item['segment_id']}: {item['reason']}"
                for item in remaining_invalid
            )
            raise ValueError(f"Concept semantic validation failed after repair: {details}")
        return self._annotate_semantic_validation(
            ordered,
            previous_concepts=previous_accepted_concepts,
            contract=_narrative_contract(global_context),
        )

    @staticmethod
    def _invalid_concepts(
        result: dict,
        global_context: dict,
        *,
        previous_concepts: dict | None = None,
        expected_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        actors = {
            str(actor.get("id") or "").strip().lower(): str(actor.get("name") or "").strip().lower()
            for actor in global_context.get("actors") or []
            if isinstance(actor, dict) and str(actor.get("id") or "").strip()
        }
        invalid = []
        accepted = dict(previous_concepts or {})
        for segment_id in expected_ids or list(result):
            if segment_id not in result:
                continue
            value = result[segment_id]
            reasons: list[str] = []
            prior_segment_id = ""
            if not isinstance(value, dict):
                accepted[segment_id] = value
                continue
            concept = str(value.get("concept") or value.get("prompt") or "").lower()
            references = value.get("references") or {}
            selected = [str(actor_id).strip().lower() for actor_id in references.get("actor_ids") or []]
            missing_names = [
                actors[actor_id]
                for actor_id in selected
                if actor_id in actors and actors[actor_id] and actors[actor_id] not in concept
            ]
            if any(phrase in concept for phrase in ("full band", "entire band", "whole band", "all band members")):
                missing_names.extend(
                    actors[actor_id]
                    for actor_id in actors
                    if actor_id not in selected and actors[actor_id]
                )
            if missing_names:
                reasons.append(
                    "Concept must name every selected actor; missing: "
                    + ", ".join(dict.fromkeys(missing_names))
                )

            semantic = _semantic_conflicts(
                value,
                accepted,
                contract=_narrative_contract(global_context),
            )
            if semantic:
                reasons.extend(semantic["reasons"])
                prior_segment_id = semantic["prior_segment_id"]
            if reasons:
                finding = {
                    "segment_id": segment_id,
                    "reason": "; ".join(reasons),
                    "current_concept": value,
                }
                if prior_segment_id:
                    finding["prior_segment_id"] = prior_segment_id
                invalid.append(finding)
                continue
            accepted[segment_id] = value
        return invalid

    @staticmethod
    def _annotate_semantic_validation(
        result: dict,
        *,
        previous_concepts: dict | None = None,
        contract: dict[str, Any] | None = None,
    ) -> dict:
        accepted = dict(previous_concepts or {})
        annotated = {}
        for segment_id, raw_value in result.items():
            if not isinstance(raw_value, dict) or not isinstance(raw_value.get("narrative"), dict):
                annotated[segment_id] = raw_value
                accepted[segment_id] = raw_value
                continue
            value = deepcopy(raw_value)
            narrative = value["narrative"]
            repeated = _one_shot_repeated_milestones(
                narrative,
                accepted,
                contract or {},
            )
            reset_events = _normalized_list(narrative.get("reset_events"))
            reprise_of = _matching_signature_segment(narrative, accepted)
            authorized = bool(reprise_of and repeated) and all(
                milestone in reset_events for milestone in repeated
            )
            value["semantic_validation"] = {
                "outcome": "accepted",
                "scene_id": segment_id,
                "story_beat": _normalize_semantic_value(narrative.get("story_beat")),
                "state_signature": _semantic_signature(narrative),
                "authorized_reprise": authorized,
                "authorized_reset_events": sorted(set(repeated) & set(reset_events)),
                "reprise_of": reprise_of if authorized else None,
            }
            annotated[segment_id] = value
            accepted[segment_id] = value
        return annotated

    def _last_concepts(self, concepts: dict[str, str]) -> dict[str, str]:
        if self.max_previous_concepts <= 0:
            return {}

        items = list(concepts.items())[-self.max_previous_concepts:]
        return dict(items)

    def _summarize_progress(
        self,
        *,
        story_idea: str,
        global_context: dict,
        concepts: dict[str, str],
    ) -> str:
        if not concepts:
            return ""

        recent = dict(list(concepts.items())[-12:])

        payload = {
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "RECENT_CONCEPTS": recent,
        }

        return self.prompt_modules.summary(
            payload,
            timeout=self.request_timeout_seconds,
        )

    @staticmethod
    def _fallback_concept(segment_id: str, global_context: dict | None = None) -> str:
        locations = (global_context or {}).get("locations") or []
        first_location = str(locations[0]).strip() if locations else ""
        if first_location:
            return (
                f"Continue the established visual story for {segment_id} in {first_location}, "
                f"preserving atmosphere, symbolic tension, and narrative progression."
            )
        return (
            f"Continue the established visual story for {segment_id}, preserving the same setting, "
            f"atmosphere, symbolic tension, and narrative progression."
        )


def save_concepts(path: str | Path, concepts: dict, *, artifact_store: ArtifactStore) -> Path:
    return artifact_store.write_json(path, concepts)


_SEMANTIC_DIMENSIONS = (
    "story_beat",
    "objective",
    "action",
    "action_phase",
    "milestones",
    "location",
    "cast_states",
    "props",
)


def _normalize_semantic_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if isinstance(value, dict):
        return {
            str(_normalize_semantic_value(key)): _normalize_semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        unique: dict[str, Any] = {}
        for item in value:
            normalized = _normalize_semantic_value(item)
            canonical = json.dumps(
                normalized,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            unique.setdefault(canonical, normalized)
        return [unique[key] for key in sorted(unique)]
    return value


def _normalized_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [
        normalized
        for item in value
        if (normalized := str(_normalize_semantic_value(item)))
    ]


def _semantic_state(narrative: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _normalize_semantic_value(narrative.get(field))
        for field in _SEMANTIC_DIMENSIONS
    }


def _semantic_signature(narrative: dict[str, Any]) -> str:
    payload = json.dumps(
        _semantic_state(narrative),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _narrative(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    narrative = value.get("narrative")
    return narrative if isinstance(narrative, dict) else {}


def _matching_signature_segment(
    narrative: dict[str, Any],
    concepts: dict[str, Any],
) -> str:
    signature = _semantic_signature(narrative)
    for segment_id, concept in concepts.items():
        prior = _narrative(concept)
        if prior and _semantic_signature(prior) == signature:
            return segment_id
    return ""


def _repeated_milestones(
    narrative: dict[str, Any],
    concepts: dict[str, Any],
) -> list[str]:
    current = set(_normalized_list(narrative.get("milestones")))
    prior = {
        milestone
        for concept in concepts.values()
        for milestone in _normalized_list(_narrative(concept).get("milestones"))
    }
    return sorted(current & prior)


def _first_milestone_segment(milestone: str, concepts: dict[str, Any]) -> str:
    return next(
        (
            segment_id
            for segment_id, concept in concepts.items()
            if milestone in _normalized_list(_narrative(concept).get("milestones"))
        ),
        "",
    )


def _one_shot_repeated_milestones(
    narrative: dict[str, Any],
    concepts: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    configured = {
        str(_normalize_semantic_value(item))
        for item in contract.get("one_shot_milestones") or ()
    }
    if not configured:
        return []
    return [
        milestone
        for milestone in _repeated_milestones(narrative, concepts)
        if milestone in configured
    ]


def _narrative_contract(global_context: dict[str, Any]) -> dict[str, Any]:
    for key in ("narrative_contract", "semantic_contract", "invariant_contract"):
        value = global_context.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _prop_regressions(
    narrative: dict[str, Any],
    concepts: dict[str, Any],
    contract: dict[str, Any],
) -> list[tuple[str, str, str, str]]:
    prop_orders = contract.get("prop_state_order") or {}
    if not isinstance(prop_orders, dict):
        return []
    normalized_orders = {
        str(_normalize_semantic_value(prop)): [
            str(_normalize_semantic_value(item))
            for item in states
        ]
        for prop, states in prop_orders.items()
        if isinstance(states, (list, tuple))
    }
    regressions = []
    props = narrative.get("props") or {}
    if not isinstance(props, dict):
        return []
    for prop, raw_state in props.items():
        normalized_prop = str(_normalize_semantic_value(prop))
        state = str(_normalize_semantic_value(raw_state))
        order = normalized_orders.get(normalized_prop, [])
        if not order or state not in order:
            continue
        for prior_segment_id, concept in reversed(list(concepts.items())):
            prior_props = _narrative(concept).get("props") or {}
            if not isinstance(prior_props, dict):
                continue
            normalized_prior_props = {
                str(_normalize_semantic_value(key)): item
                for key, item in prior_props.items()
            }
            if normalized_prop not in normalized_prior_props:
                continue
            prior_state = str(
                _normalize_semantic_value(normalized_prior_props[normalized_prop]),
            )
            if prior_state in order and order.index(state) < order.index(prior_state):
                regressions.append(
                    (normalized_prop, prior_state, state, prior_segment_id),
                )
            break
    return regressions


def _semantic_conflicts(
    value: dict[str, Any],
    prior_concepts: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    narrative = _narrative(value)
    if not narrative:
        return {}

    reset_events = set(_normalized_list(narrative.get("reset_events")))
    malformed_props = (
        narrative.get("props") is not None
        and not isinstance(narrative.get("props"), dict)
    )
    repeated = _one_shot_repeated_milestones(
        narrative,
        prior_concepts,
        contract,
    )
    unauthorized_milestones = [item for item in repeated if item not in reset_events]
    matching_segment = _matching_signature_segment(narrative, prior_concepts)
    authorized_signature = bool(matching_segment and repeated) and not unauthorized_milestones
    regressions = [] if malformed_props else _prop_regressions(
        narrative,
        prior_concepts,
        contract,
    )
    unauthorized_regressions = [
        item for item in regressions if item[0] not in reset_events
    ]

    reasons = ["narrative.props must be an object"] if malformed_props else []
    prior_segment_id = matching_segment
    if matching_segment and not authorized_signature:
        state = _semantic_state(narrative)
        reasons.append(
            f"semantic scene duplicates {matching_segment} "
            f"(story_beat {state['story_beat']!r}, action {state['action']!r}, "
            f"action_phase {state['action_phase']!r})"
        )
    for milestone in unauthorized_milestones:
        milestone_segment = _first_milestone_segment(milestone, prior_concepts)
        prior_segment_id = prior_segment_id or milestone_segment
        reasons.append(
            f"milestone {milestone!r} repeats {milestone_segment} "
            "without reset_events authorization"
        )
    for prop, previous_state, state, prop_segment in unauthorized_regressions:
        prior_segment_id = prior_segment_id or prop_segment
        reasons.append(
            f"prop {prop!r} state {state!r} regresses from {previous_state!r} "
            f"in {prop_segment} without reset_events authorization"
        )
    if not reasons:
        return {}
    return {"reasons": reasons, "prior_segment_id": prior_segment_id}
