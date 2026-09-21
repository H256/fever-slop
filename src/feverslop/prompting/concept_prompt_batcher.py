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


# One repair call carries at most this many scene keys so its structured
# output stays inside the per-scene token budget (see llm_policy
# concept_batch_max_tokens) and a truncated response can never strand a whole
# batch repair.
_KEYS_PER_REPAIR_CALL = 2

# Concept-batch request budget. Front-loading BOUNDARY_CONTEXT and a
# closed-vocabulary block onto every initial batch (issue #1247) adds tokens on
# top of the existing PREVIOUS_CONCEPTS + ACCEPTED_STATE_LEDGER + summary. This
# ceiling is deliberately well below the ~100k-token model limit so a regression
# that grows the request unbounded fails fast instead of silently truncating.
_REQUEST_TOKEN_CEILING = 80_000

# The closed-vocabulary block must stay bounded so it cannot grow the request
# unbounded as a project accumulates more cast/locations/props.
_VOCAB_MAX_PER_FIELD = 40
_VOCAB_MAX_TOTAL = 200


def _estimate_request_tokens(payload: dict[str, Any]) -> int:
    """Deterministic lower-bound estimate of a request's input token count.

    Uses the standard chars//4 heuristic on the compacted, evidence-stripped
    payload. It is a lower bound for prose but conservative enough for a budget
    ceiling: if the estimate is under the ceiling, the real count is too.
    """
    material = json.dumps(
        compact_planning_payload(payload),
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )
    return len(material) // 4


def _closed_vocabulary(
    global_context: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, list[str]]:
    """Allowed values for narrative state fields, derived from the contract and
    structured ids, so the model picks from a fixed set instead of inventing
    identifiers. Only fields with actual values are included, and the block is
    capped so it cannot grow the request unbounded.
    """
    vocab: dict[str, list[str]] = {}

    def add(field: str, values: list[str]) -> None:
        unique = list(dict.fromkeys(str(v) for v in values if str(v)))
        if unique:
            vocab[field] = unique[:_VOCAB_MAX_PER_FIELD]

    # Structured ids from the project (the canonical identifiers the model must
    # reuse verbatim in references and narrative state fields).
    add("location_ids", [
        str(item.get("id")) for item in (global_context.get("structured_locations") or [])
        if isinstance(item, dict) and item.get("id")
    ])
    add("actor_ids", [
        str(item.get("id")) for item in (global_context.get("actors") or [])
        if isinstance(item, dict) and item.get("id")
    ])
    add("prop_ids", [
        str(item.get("id")) for item in (global_context.get("props") or [])
        if isinstance(item, dict) and item.get("id")
    ])
    # Ordered/allowed state vocabularies from the narrative contract.
    add("milestones", _ordered_contract_ids(contract, "milestone_order"))
    add("locations", _ordered_contract_ids(contract, "location_order"))
    for field in ("allowed_actions", "allowed_action_phases", "allowed_cast_states"):
        raw = contract.get(field)
        if isinstance(raw, (list, tuple)):
            add(field, [str(_normalize_semantic_value(v)) for v in raw])
    # Enforce the total cap, dropping whole fields past the budget in a stable
    # order (structured ids first, then contract vocabularies).
    total = sum(len(v) for v in vocab.values())
    if total > _VOCAB_MAX_TOTAL:
        for field in list(vocab):
            if total <= _VOCAB_MAX_TOTAL:
                break
            keep = max(0, _VOCAB_MAX_TOTAL - total)
            vocab[field] = vocab[field][:keep]
            total -= len(vocab[field])
            if not vocab[field]:
                del vocab[field]
    return vocab


def _narrative_constraints(contract: dict[str, Any]) -> dict[str, Any]:
    """Explicit one-shot and terminal-state constraints for the model.

    The closed vocabulary tells the model which milestones exist; this block
    tells it which are one-shot (must not repeat without a reset_events
    authorization) and which terminal milestones require an explicit cast
    state.  Only present when the contract actually configures these
    constraints.
    """
    constraints: dict[str, Any] = {}
    one_shot = [
        str(_normalize_semantic_value(item))
        for item in contract.get("one_shot_milestones") or ()
        if str(_normalize_semantic_value(item))
    ]
    if one_shot:
        constraints["one_shot_milestones"] = one_shot[:_VOCAB_MAX_PER_FIELD]
    terminal_states = contract.get("terminal_states") or {}
    if isinstance(terminal_states, dict) and terminal_states:
        normalized: dict[str, Any] = {}
        for actor, rule in terminal_states.items():
            if not isinstance(rule, dict):
                continue
            entry: dict[str, Any] = {}
            milestone = _normalize_semantic_value(rule.get("milestone"))
            if milestone is not None:
                entry["milestone"] = str(milestone)
            state = _normalize_semantic_value(rule.get("state"))
            if state is not None:
                entry["state"] = str(state)
            reset_event = _normalize_semantic_value(rule.get("reset_event"))
            if reset_event is not None:
                entry["reset_event"] = str(reset_event)
            if entry:
                normalized[str(_normalize_semantic_value(actor))] = entry
        if normalized:
            constraints["terminal_states"] = dict(
                list(normalized.items())[:_VOCAB_MAX_PER_FIELD]
            )
    return constraints


def _derive_fix_instructions(reasons: list[str]) -> list[str]:
    """Derive concrete, actionable fix instructions from validation reasons.

    The raw reason text names the offending value and the colliding scene,
    but the repair model still has to infer the correction.  These
    instructions state the exact field change or reset_events authorization
    that resolves each reason, so a single repair pass converges instead of
    re-emitting the same value.
    """
    instructions: list[str] = []
    for reason in reasons:
        match = re.match(
            r"milestone '([^']+)' repeats (\S+) without reset_events authorization",
            reason,
        )
        if match:
            milestone, prior = match.groups()
            instructions.append(
                f"Remove milestone '{milestone}' from milestones (one-shot, "
                f"already used in {prior}), or add '{milestone}' to "
                f"reset_events."
            )
            continue
        match = re.match(
            r"prop '([^']+)' state '([^']+)' regresses from '([^']+)' in "
            r"(\S+) without reset_events authorization",
            reason,
        )
        if match:
            prop, state, previous, prior = match.groups()
            instructions.append(
                f"Set props['{prop}'] to a state at or after '{previous}' "
                f"(it was '{previous}' in {prior}), or add '{prop}' to "
                f"reset_events."
            )
            continue
        match = re.match(
            r"milestone '([^']+)' reverses current milestone '([^']+)'",
            reason,
        )
        if match:
            milestone, previous = match.groups()
            instructions.append(
                f"Remove milestone '{milestone}' (reverses '{previous}'); "
                f"use a milestone at or after '{previous}'."
            )
            continue
        match = re.match(
            r"milestone '([^']+)' observed before required predecessor "
            r"'([^']+)'",
            reason,
        )
        if match:
            milestone, predecessor = match.groups()
            instructions.append(
                f"Remove milestone '{milestone}' (skips required predecessor "
                f"'{predecessor}'); resolve '{predecessor}' in an earlier "
                f"scene."
            )
            continue
        match = re.match(
            r"location '([^']+)' reverses current location '([^']+)'",
            reason,
        )
        if match:
            current, previous = match.groups()
            instructions.append(
                f"Set location to a location at or after '{previous}' "
                f"(it was '{previous}'), or name a chronology exception in "
                f"causal_events."
            )
            continue
        match = re.match(
            r"terminal milestone '([^']+)' requires explicit cast state "
            r"'([^']+)' for actor '([^']+)'",
            reason,
        )
        if match:
            milestone, state, actor = match.groups()
            instructions.append(
                f"Set cast_states['{actor}'] to '{state}' (terminal "
                f"milestone '{milestone}' requires it)."
            )
            continue
        match = re.match(
            r"(?:terminal milestone '([^']+)'|actor '([^']+)') requires "
            r"terminal state '([^']+)' from (\S+); observed '([^']+)' "
            r"without causal event '([^']+)'",
            reason,
        )
        if match:
            required_state = match.group(3)
            observed = match.group(5)
            reset_event = match.group(6)
            instructions.append(
                f"Set the actor's cast state to '{required_state}', or add "
                f"causal event '{reset_event}' to authorize observed state "
                f"'{observed}'."
            )
            continue
        match = re.match(
            r"(\S+)\.incoming\.([^:]+): '([^']+)' is incompatible with "
            r"continuous (\S+) outgoing state '([^']+)'",
            reason,
        )
        if match:
            field = match.group(2)
            before = match.group(5)
            instructions.append(
                f"Set incoming.{field} to '{before}' (must match the "
                f"predecessor's continuous outgoing state), or name the "
                f"change in transition_events."
            )
            continue
        match = re.match(
            r"(\S+)\.incoming\.([^:]+)\.([^:]+): '([^']+)' is incompatible "
            r"with (\S+) outgoing state '([^']+)'",
            reason,
        )
        if match:
            field, key = match.group(2), match.group(3)
            before = match.group(6)
            instructions.append(
                f"Set incoming.{field}.{key} to '{before}' (must match the "
                f"predecessor's outgoing state), or name the change in "
                f"transition_events."
            )
            continue
        instructions.append(f"Fix: {reason}")
    return instructions


def _sequence_aware_repair_ids(
    missing: list[str],
    invalid: list[dict[str, Any]],
    expected_ids: list[str],
) -> list[str]:
    """Expand the repair set to the smallest contiguous sequence window.

    When a semantic conflict names a prior segment (a one-shot milestone
    repeated, a prop regression, a chronology violation), the affected
    sequence spans from the cited predecessor through the offending scene.
    Repairing only the two endpoints leaves the intervening scenes
    inconsistent.  This helper expands the repair set to include every
    segment between the minimum and maximum affected ID, so the repair
    model sees the full sequence and can reallocate preparation,
    irreversible event, completion, and aftermath coherently.

    The cited predecessor is only included in the window when it is itself
    invalid or missing; a valid predecessor anchors the boundary and does
    not need regeneration.
    """
    invalid_ids = {item["segment_id"] for item in invalid}
    missing_set = set(missing)
    affected = set(missing)
    for item in invalid:
        affected.add(item["segment_id"])
        prior = item.get("prior_segment_id")
        if prior and (prior in invalid_ids or prior in missing_set):
            affected.add(prior)
    if not affected:
        return []
    # Expand to the contiguous window between min and max affected IDs.
    indices = [expected_ids.index(seg) for seg in affected if seg in expected_ids]
    if not indices:
        return sorted(affected, key=expected_ids.index)
    lo, hi = min(indices), max(indices)
    window = set(expected_ids[lo:hi + 1])
    # Include any missing IDs that fall outside the window.
    window.update(seg for seg in missing if seg not in expected_ids)
    return sorted(window, key=expected_ids.index)


def _sequence_allocation_window(
    invalid: list[dict[str, Any]],
    expected_ids: list[str],
) -> list[str]:
    """Compute the read-only allocation window.

    Unlike the repair window, this includes valid cited predecessors so
    the model can see the original accepted occurrence.  The predecessor
    need not be regenerated; it is included for context only.
    """
    affected: set[str] = set()
    for item in invalid:
        affected.add(item["segment_id"])
        prior = item.get("prior_segment_id")
        if prior:
            affected.add(prior)
    if not affected:
        return []
    indices = [expected_ids.index(seg) for seg in affected if seg in expected_ids]
    if not indices:
        return sorted(affected, key=expected_ids.index)
    lo, hi = min(indices), max(indices)
    return expected_ids[lo:hi + 1]


def _sequence_allocation(
    invalid: list[dict[str, Any]],
    expected_ids: list[str],
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a compact sequence allocation for the repair prompt.

    When a one-shot milestone conflict or terminal-state mismatch is
    detected, the repair model needs to know the expected sequence
    structure: which scene is the preparation, which is the irreversible
    event, which is the completion, and which are the aftermath.  This
    allocation is immutable and given to every repair call in the window.

    The allocation window includes valid cited predecessors (read-only
    context) so the model can see the original accepted occurrence.  The
    irreversible event is the original accepted milestone occurrence, not
    the duplicate that triggered the conflict.
    """
    one_shot = set(
        str(_normalize_semantic_value(item))
        for item in contract.get("one_shot_milestones") or ()
        if str(_normalize_semantic_value(item))
    )
    terminal = contract.get("terminal_states") or {}
    if not isinstance(terminal, dict):
        terminal = {}
    # Collect the milestones and terminal states involved in the conflicts.
    conflicted_milestones: set[str] = set()
    conflicted_actors: set[str] = set()
    for item in invalid:
        reason = item.get("reason", "")
        for milestone in one_shot:
            if f"milestone '{milestone}' repeats" in reason:
                conflicted_milestones.add(milestone)
        for actor, rule in terminal.items():
            if not isinstance(rule, dict):
                continue
            state = str(_normalize_semantic_value(rule.get("state")))
            if state and f"requires terminal state '{state}'" in reason:
                conflicted_actors.add(str(_normalize_semantic_value(actor)))
    if not conflicted_milestones and not conflicted_actors:
        return None
    # Build the allocation window: includes valid cited predecessors for
    # read-only context, so the model can see the original accepted
    # occurrence.
    window_ids = _sequence_allocation_window(invalid, expected_ids)
    if not window_ids:
        return None
    # The irreversible event is the original accepted milestone occurrence
    # (the cited predecessor), not the duplicate that triggered the
    # conflict.  Completion is the scene with the terminal state;
    # everything before is preparation, everything after is aftermath.
    event_index = None
    completion_index = None
    for i, seg_id in enumerate(window_ids):
        item = next(
            (it for it in invalid if it["segment_id"] == seg_id),
            None,
        )
        if not item:
            continue
        reason = item.get("reason", "")
        # If the reason says "milestone X repeats segment_Y", then
        # segment_Y (the cited predecessor) is the irreversible event.
        if event_index is None and any(
            f"milestone '{m}' repeats" in reason for m in conflicted_milestones
        ):
            prior = item.get("prior_segment_id")
            if prior and prior in window_ids:
                event_index = window_ids.index(prior)
            else:
                event_index = i
        if completion_index is None and "requires terminal state" in reason:
            completion_index = i
    # Aftermath: everything after the completion, or after the event if
    # there is no completion.
    if completion_index is not None:
        aftermath = window_ids[completion_index + 1:]
    elif event_index is not None:
        aftermath = window_ids[event_index + 1:]
    else:
        aftermath = []
    allocation: dict[str, Any] = {
        "window": window_ids,
        "preparation": window_ids[:event_index] if event_index is not None else [],
        "irreversible_event": (
            [window_ids[event_index]] if event_index is not None else []
        ),
        "completion": (
            [window_ids[completion_index]] if completion_index is not None else []
        ),
        "aftermath": aftermath,
    }
    if conflicted_milestones:
        allocation["one_shot_milestones"] = sorted(conflicted_milestones)
    if conflicted_actors:
        allocation["terminal_actors"] = sorted(conflicted_actors)
    # Remove empty lists to keep the payload compact.
    return {k: v for k, v in allocation.items() if v}


def _ordered_contract_ids(contract: dict[str, Any], key: str) -> list[str]:
    return [item_id for item_id, _source in _ordered_contract_entries(contract, key)]


def _batch_boundary_anchor(
    first_segment_id: str,
    *,
    known: dict[str, Any],
    order_ids: list[str],
) -> dict[str, Any]:
    """Exact accepted `outgoing` boundary state the batch's first scene must
    anchor on.

    The initial batch otherwise only sees a compact ledger and the last few
    concepts verbatim, so it must *infer* the exact boundary tokens. Handing it
    the predecessor's accepted outgoing snapshot (the same canonical vocabulary
    the repair path already receives) lets the first scene restate boundary
    states verbatim instead of paraphrasing them.
    """
    positions = {segment_id: index for index, segment_id in enumerate(order_ids)}
    position = positions.get(first_segment_id)
    if position is None:
        return {}
    for step in range(1, len(order_ids)):
        neighbor_position = position - step
        if not 0 <= neighbor_position < len(order_ids):
            break
        neighbor_id = order_ids[neighbor_position]
        if neighbor_id not in known:
            continue
        state = _continuity_boundary_state(known[neighbor_id], "outgoing")
        if not state:
            continue
        anchor = {"scene_id": neighbor_id, "outgoing": state}
        if step > 1:
            anchor["neighbor_distance"] = step
        return anchor
    return {}


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
        semantic_enforcement: str = "warn",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if semantic_enforcement not in ("warn", "block"):
            raise ValueError(
                f"semantic_enforcement must be 'warn' or 'block', "
                f"got {semantic_enforcement!r}"
            )

        self.llm = llm
        self.batch_size = batch_size
        self.max_previous_concepts = max_previous_concepts
        self.request_timeout_seconds = request_timeout_seconds
        self.progress_callback = progress_callback
        self.prompt_modules = prompt_modules or MusicVideoPromptModules(llm)
        self.semantic_enforcement = semantic_enforcement
        self._checkpoint_path: Path | None = None
        self._checkpoint_store: ArtifactStore | None = None

    def enable_checkpoint(self, *, path: str | Path, artifact_store: ArtifactStore) -> None:
        """Persist accepted concepts after every batch so a crash can resume.

        The checkpoint is keyed by a fingerprint of every creative input; a
        fingerprint mismatch invalidates it instead of reusing stale scenes,
        and it is deleted once a complete run succeeded.
        """
        self._checkpoint_path = Path(path)
        self._checkpoint_store = artifact_store

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

        checkpoint_identity = None
        if self._checkpoint_path is not None and self._checkpoint_store is not None:
            checkpoint_identity = self._checkpoint_fingerprint(
                story_idea=story_idea,
                notes=notes,
                global_context=global_context,
                stage1_segments=stage1_segments,
            )
            restored, stale = self._load_checkpoint(checkpoint_identity)
            if stale:
                self._report("Ignoring stale concept checkpoint (inputs changed)", report)
            if restored:
                all_results.update(restored)
                self._report(
                    f"Resuming concept generation from checkpoint ({len(all_results)} scenes)",
                    report,
                )
                if all_results:
                    previous_summary = self._summarize_progress(
                        story_idea=story_idea,
                        global_context=global_context,
                        concepts=all_results,
                    )

        for batch_number, (batch_start, batch) in enumerate(batches, start=1):
            batch_label = f"Concept batch {batch_number}/{len(batches)}"
            expected_ids = [seg["segment_id"] for seg in batch]
            if (
                checkpoint_identity is not None
                and expected_ids
                and all(segment_id in all_results for segment_id in expected_ids)
            ):
                self._report(
                    f"{batch_label}: reused from concept checkpoint "
                    f"({len(expected_ids)} scenes)",
                    report,
                )
                continue
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
                accepted_ledger=_accepted_state_ledger(all_results),
                accepted=all_results,
                order_ids=[seg["segment_id"] for seg in stage1_segments],
            )
            self._report(f"{batch_label}: response received, validating keys", report)

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
            self._write_checkpoint(checkpoint_identity, all_results)

        missing = [
            seg["segment_id"]
            for seg in stage1_segments
            if seg["segment_id"] not in all_results
        ]

        if missing:
            raise ValueError(f"Missing concept prompts after batched generation: {missing}")

        self._clear_checkpoint()
        # Preserve stage1 order in output JSON.
        return {
            seg["segment_id"]: all_results[seg["segment_id"]]
            for seg in stage1_segments
        }

    def _checkpoint_fingerprint(
        self,
        *,
        story_idea: str,
        notes: str,
        global_context: dict,
        stage1_segments: list[dict],
    ) -> str:
        material = json.dumps(
            {
                "batch_size": self.batch_size,
                "story_idea": story_idea,
                "notes": notes,
                "global_context": compact_planning_payload(global_context),
                "segments": compact_planning_payload(stage1_segments),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _load_checkpoint(self, identity: str) -> tuple[dict[str, Any], bool]:
        """Return (restored concepts, stale) for the configured checkpoint."""
        assert self._checkpoint_store is not None and self._checkpoint_path is not None
        try:
            data = self._checkpoint_store.read_json(self._checkpoint_path)
        except FileNotFoundError:
            return {}, False
        except (OSError, ValueError):
            return {}, True
        if (
            isinstance(data, dict)
            and data.get("identity") == identity
            and isinstance(data.get("concepts"), dict)
        ):
            restored = {
                segment_id: value
                for segment_id, value in data["concepts"].items()
                if isinstance(segment_id, str) and isinstance(value, (dict, str)) and value
            }
            return restored, False
        return {}, True

    def _write_checkpoint(self, identity: str | None, concepts: dict[str, Any]) -> None:
        if identity is None or self._checkpoint_store is None or self._checkpoint_path is None:
            return
        self._checkpoint_store.write_json(
            self._checkpoint_path,
            {"identity": identity, "concepts": concepts},
        )

    def _clear_checkpoint(self) -> None:
        if self._checkpoint_path is None:
            return
        self._checkpoint_path.unlink(missing_ok=True)

    def _report(self, message: str, callback: Callable[[str], None] = None) -> None:
        callback = callback or self.progress_callback
        if callback is not None:
            callback(message)

    @staticmethod
    def _repair_source_message(
        total: int,
        missing: list[str],
        invalid: list[dict[str, Any]],
        repair_ids: list[str],
    ) -> str:
        # Missing keys (truncation/omitted output) and invalid keys (failed
        # semantic validation) are separate failure sources; log them apart so
        # a truncated response is distinguishable from a tuning problem.
        invalid_parts = [
            f"{item['segment_id']} ({item['reason']})" for item in invalid
        ]
        return (
            f"Concept batch: repairing {total} scene "
            f"{'key' if total == 1 else 'keys'} "
            f"({len(missing)} missing, {len(invalid)} invalid): "
            f"{', '.join(repair_ids)}"
            + (f" [missing: {', '.join(missing)}]" if missing else "")
            + (f" [invalid: {'; '.join(invalid_parts)}]" if invalid_parts else "")
        )

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
        accepted_ledger: list[dict[str, str]] | None = None,
        accepted: dict | None = None,
        order_ids: list[str] | None = None,
    ) -> dict:
        accepted = accepted or {}
        order_ids = order_ids or []
        payload = {
            "BATCH_INDEX": batch_index,
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "NOTES": notes,
            "PREVIOUS_PROGRESS_SUMMARY": previous_summary,
            "PREVIOUS_CONCEPTS": previous_concepts,
            # The validator rejects a full semantic-state collision with ANY
            # accepted scene, not just the recent window; this compact ledger
            # gives the model the same collision surface it will be judged on.
            "ACCEPTED_STATE_LEDGER": accepted_ledger or [],
            "CURRENT_BATCH_SEGMENTS": compact_planning_payload(batch),
        }
        # Front-load the exact boundary vocabulary so the batch's first scene
        # anchors on verbatim tokens instead of inferred ones (issue #1247).
        # Only present when there is an accepted predecessor to anchor on.
        first_segment_id = batch[0].get("segment_id") if batch else None
        if first_segment_id:
            anchor = _batch_boundary_anchor(
                first_segment_id, known=accepted, order_ids=order_ids,
            )
            if anchor:
                payload["BOUNDARY_CONTEXT"] = {first_segment_id: anchor}
        # Constrain the output to a closed vocabulary derived from the narrative
        # contract and structured ids so the model picks instead of inventing.
        contract = _narrative_contract(global_context)
        vocabulary = _closed_vocabulary(global_context, contract)
        if vocabulary:
            payload["CLOSED_VOCABULARY"] = vocabulary
        # Explicit one-shot and terminal-state constraints so the model knows
        # which milestones must not repeat and which require an explicit cast
        # state, instead of discovering them only through repair.
        constraints = _narrative_constraints(contract)
        if constraints:
            payload["NARRATIVE_CONSTRAINTS"] = constraints
        # Guard the request budget: front-loaded context must not push the
        # request toward the ~100k-token model limit.
        estimated = _estimate_request_tokens(payload)
        if estimated > _REQUEST_TOKEN_CEILING:
            raise ValueError(
                f"Concept batch request exceeds token budget "
                f"({estimated} > {_REQUEST_TOKEN_CEILING} estimated tokens); "
                "bound the added context before generating"
            )

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
        repair_ids = _sequence_aware_repair_ids(
            missing,
            invalid,
            expected_ids,
        )

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
            self._repair_source_message(
                len(repair_ids),
                missing,
                invalid,
                repair_ids,
            ),
            progress_callback,
        )

        repaired.update(self._repair_scenes(
            repair_ids=repair_ids,
            missing_ids=set(missing),
            invalid_items=invalid,
            repaired=repaired,
            accepted=previous_accepted_concepts,
            expected_ids=expected_ids,
            batch=batch,
            story_idea=story_idea,
            global_context=global_context,
            notes=notes,
            previous_concepts=previous_concepts,
            previous_summary=previous_summary,
            progress_callback=progress_callback,
        ))

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
        # A repaired predecessor can strand an already-accepted successor whose
        # boundary states no longer match token-for-token. Give those collateral
        # scenes exactly one bounded follow-up repair; scenes whose own repair
        # response was invalid never get a second attempt.
        remaining_ids = [item["segment_id"] for item in remaining_invalid]
        collateral_ids = [segment_id for segment_id in remaining_ids if segment_id not in repair_ids]
        if collateral_ids:
            self._report(
                f"Concept batch: repairing {len(collateral_ids)} scene "
                f"{'key' if len(collateral_ids) == 1 else 'keys'} broken by adjacent "
                f"repairs: {', '.join(collateral_ids)}",
                progress_callback,
            )
            repaired.update(self._repair_scenes(
                repair_ids=collateral_ids,
                missing_ids=set(),
                invalid_items=[
                    item for item in remaining_invalid
                    if item["segment_id"] in set(collateral_ids)
                ],
                repaired=repaired,
                accepted={**previous_accepted_concepts, **ordered},
                expected_ids=expected_ids,
                batch=batch,
                story_idea=story_idea,
                global_context=global_context,
                notes=notes,
                previous_concepts=self._last_concepts({
                    **previous_accepted_concepts,
                    **ordered,
                }),
                previous_summary=previous_summary,
                progress_callback=progress_callback,
            ))
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
            if self.semantic_enforcement == "block":
                raise ValueError(
                    f"Concept semantic validation failed after repair: {details}"
                )
            # Warn mode: preserve structurally valid concepts, persist
            # unresolved diagnostics, emit a visible warning, and continue.
            self._report(
                f"[yellow]Warning: {len(remaining_invalid)} semantic "
                f"validation issue(s) remain after repair; continuing with "
                f"unresolved diagnostics. "
                f"Segments: {', '.join(item['segment_id'] for item in remaining_invalid)}. "
                f"Details: {details}[/yellow]",
                progress_callback,
            )
            return self._annotate_semantic_validation(
                ordered,
                previous_concepts=previous_accepted_concepts,
                contract=_narrative_contract(global_context),
                unresolved=remaining_invalid,
            )
        return self._annotate_semantic_validation(
            ordered,
            previous_concepts=previous_accepted_concepts,
            contract=_narrative_contract(global_context),
        )

    def _repair_scenes(
        self,
        *,
        repair_ids: list[str],
        missing_ids: set[str],
        invalid_items: list[dict[str, Any]],
        repaired: dict[str, Any],
        accepted: dict[str, Any],
        expected_ids: list[str],
        batch: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str,
        previous_concepts: dict,
        previous_summary: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        # Repair in small sequential chunks: each call carries a bounded number
        # of scene keys plus the exact accepted boundary states beside them, so
        # the model regenerates canonical values and the response always fits
        # its per-scene token budget.
        invalid_by_id = {item["segment_id"]: item for item in invalid_items}
        target_ids = set(repair_ids)
        expected_set = set(expected_ids)
        sequence_ids = [
            *(segment_id for segment_id in accepted if segment_id not in expected_set),
            *expected_ids,
        ]
        # States the repair model must not reproduce: every accepted scene it
        # can see (prior batches plus valid in-batch survivors) in the same
        # compact form the batch generator receives.
        ledger_concepts = {
            **accepted,
            **{
                segment_id: value
                for segment_id, value in repaired.items()
                if segment_id not in target_ids
            },
        }
        repaired_values: dict[str, Any] = {}
        for start in range(0, len(repair_ids), _KEYS_PER_REPAIR_CALL):
            chunk_ids = repair_ids[start:start + _KEYS_PER_REPAIR_CALL]
            chunk_missing = {segment_id for segment_id in chunk_ids if segment_id in missing_ids}
            payload = {
                "STORY_IDEA": story_idea,
                "GLOBAL_CONTEXT": global_context,
                "NOTES": notes,
                "PREVIOUS_PROGRESS_SUMMARY": previous_summary,
                "PREVIOUS_CONCEPTS": previous_concepts,
                "ACCEPTED_STATE_LEDGER": _accepted_state_ledger(ledger_concepts),
                "MISSING_SEGMENTS": compact_planning_payload(
                    [seg for seg in batch if seg["segment_id"] in chunk_missing]
                ),
                "INVALID_SEGMENTS": [
                    invalid_by_id[segment_id]
                    for segment_id in chunk_ids
                    if segment_id in invalid_by_id
                ],
                "EXPECTED_KEYS": chunk_ids,
                "BOUNDARY_CONTEXT": _boundary_context(
                    chunk_ids,
                    known={**accepted, **repaired, **repaired_values},
                    order_ids=sequence_ids,
                    excluded=target_ids,
                ),
            }
            # Explicit one-shot and terminal-state constraints so the repair
            # model knows which milestones must not repeat and which require
            # an explicit cast state.
            constraints = _narrative_constraints(_narrative_contract(global_context))
            if constraints:
                payload["NARRATIVE_CONSTRAINTS"] = constraints
            # Sequence allocation: when a one-shot or terminal-state conflict
            # is detected, the repair model needs the expected sequence
            # structure (preparation, irreversible event, completion,
            # aftermath) to reallocate the window coherently.
            allocation = _sequence_allocation(
                invalid_items,
                sequence_ids,
                _narrative_contract(global_context),
            )
            if allocation:
                payload["SEQUENCE_ALLOCATION"] = allocation
            response = self.prompt_modules.repair_concepts(
                payload,
                timeout=self.request_timeout_seconds,
            )
            repair = response if isinstance(response, dict) else extract_json_object(str(response))
            absent = [segment_id for segment_id in chunk_ids if repair.get(segment_id) is None]
            if absent:
                # Keep the documented one-shot fallback semantics, but never
                # hide that a truncated or malformed repair response stranded
                # these keys on their stale/fallback values.
                self._report(
                    f"Concept batch: repair response incomplete for "
                    f"{', '.join(absent)}",
                    progress_callback,
                )
            for segment_id in chunk_ids:
                value = repair.get(segment_id)
                if value is None:
                    value = (
                        self._fallback_concept(segment_id, global_context)
                        if segment_id in missing_ids
                        else repaired[segment_id]
                    )
                repaired_values[segment_id] = value
        return repaired_values

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
        contract = _narrative_contract(global_context)
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
                contract=contract,
                segment_id=segment_id,
            )
            if semantic:
                reasons.extend(semantic["reasons"])
                prior_segment_id = semantic["prior_segment_id"]
            if reasons:
                finding = {
                    "segment_id": segment_id,
                    "reason": "; ".join(reasons),
                    "fix_instructions": _derive_fix_instructions(reasons),
                    "current_concept": value,
                }
                if prior_segment_id:
                    finding["prior_segment_id"] = prior_segment_id
                    # The reason text only names a few fields; give the repair
                    # prompt the full normalized state of the scene this one
                    # collides with, which may be outside every visible window.
                    prior_narrative = _narrative(accepted.get(prior_segment_id))
                    if prior_narrative:
                        finding["prior_segment_state"] = _semantic_state(prior_narrative)
                invalid.append(finding)
                continue
            # Store the annotated form so later scenes are validated against
            # the same boundary states the annotated chronology gate enforces.
            accepted[segment_id] = ConceptPromptBatcher._annotate_semantic_validation(
                {segment_id: value},
                previous_concepts=accepted,
                contract=contract,
            )[segment_id]
        return invalid

    @staticmethod
    def _annotate_semantic_validation(
        result: dict,
        *,
        previous_concepts: dict | None = None,
        contract: dict[str, Any] | None = None,
        unresolved: list[dict[str, Any]] | None = None,
    ) -> dict:
        accepted = dict(previous_concepts or {})
        unresolved_ids = {
            item["segment_id"] for item in (unresolved or [])
        }
        unresolved_details = {
            item["segment_id"]: item.get("reason", "")
            for item in (unresolved or [])
        }
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
            is_unresolved = segment_id in unresolved_ids
            value["semantic_validation"] = {
                "outcome": "warning" if is_unresolved else "accepted",
                "scene_id": segment_id,
                "story_beat": _normalize_semantic_value(narrative.get("story_beat")),
                "state_signature": _semantic_signature(narrative),
                "authorized_reprise": authorized,
                "authorized_reset_events": sorted(set(repeated) & set(reset_events)),
                "reprise_of": reprise_of if authorized else None,
                "unresolved_diagnostic": (
                    unresolved_details.get(segment_id) if is_unresolved else None
                ),
                "chronology": _chronology_evidence(
                    segment_id,
                    narrative,
                    accepted,
                    contract or {},
                ),
                "continuity": _adjacent_continuity_plan(
                    segment_id,
                    narrative,
                    accepted,
                    contract or {},
                ),
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


def validate_and_annotate_concept_chronology(
    concepts: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete ordered concept sequence before downstream prompts."""
    final_segment_id = next(reversed(concepts), "")
    terminal_milestones = set(
        _normalized_list(contract.get("terminal_milestones") or ["story_complete"])
    )
    for segment_id, value in concepts.items():
        if segment_id == final_segment_id:
            continue
        premature = terminal_milestones.intersection(
            _normalized_list(_narrative(value).get("milestones"))
        )
        if premature:
            milestone = sorted(premature)[0]
            raise ValueError(
                "Concept chronology validation failed: terminal milestone "
                f"{milestone!r} is only valid on final segment {final_segment_id!r}"
            )
    accepted: dict[str, Any] = {}
    failures: list[str] = []
    for segment_id, value in concepts.items():
        if isinstance(value, dict):
            conflict = _semantic_conflicts(
                value,
                accepted,
                contract=contract,
                segment_id=segment_id,
            )
            if conflict:
                failures.append(f"{segment_id}: {'; '.join(conflict['reasons'])}")
                continue
        accepted[segment_id] = value
    if failures:
        raise ValueError("Concept chronology validation failed: " + failures[0])
    allocated = {
        milestone
        for value in concepts.values()
        for milestone in _normalized_list(_narrative(value).get("milestones"))
    }
    for milestone, source in _ordered_contract_entries(contract, "milestone_order"):
        if milestone not in allocated:
            raise ValueError(
                "Concept chronology validation failed: required milestone "
                f"{milestone!r} is unallocated (source: {source})",
            )
    return ConceptPromptBatcher._annotate_semantic_validation(
        concepts,
        contract=contract,
    )


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


_LEDGER_FIELDS = ("story_beat", "action", "action_phase", "location")


def _accepted_state_ledger(concepts: dict[str, Any]) -> list[dict[str, str]]:
    """Compact do-not-reproduce map of accepted scene states for the model.

    Validation compares the full semantic signature of a scene against every
    accepted scene, while the prompt otherwise only shows the last few
    concepts verbatim. This ledger gives generation and repair the same
    collision surface in a few tokens per scene.
    """
    ledger: list[dict[str, str]] = []
    for segment_id, value in concepts.items():
        narrative = _narrative(value)
        if not narrative:
            continue
        entry: dict[str, str] = {"scene_id": str(segment_id)}
        for field in _LEDGER_FIELDS:
            state = _normalized_scalar(narrative.get(field))
            if state:
                entry[field] = state
        ledger.append(entry)
    return ledger


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


def _ordered_contract_entries(
    contract: dict[str, Any], key: str,
) -> list[tuple[str, str]]:
    entries = []
    for raw in contract.get(key) or ():
        if isinstance(raw, dict):
            item_id = str(_normalize_semantic_value(raw.get("id")))
            source = str(raw.get("source") or key)
        else:
            item_id = str(_normalize_semantic_value(raw))
            source = key
        if item_id:
            entries.append((item_id, source))
    return entries


def _approved_chronology_exception(
    narrative: dict[str, Any],
    contract: dict[str, Any],
    dimension: str,
) -> str:
    causal_events = set(_normalized_list(narrative.get("causal_events")))
    exceptions = contract.get("chronology_exceptions") or {}
    if not isinstance(exceptions, dict):
        return ""
    for event in causal_events:
        rule = exceptions.get(event)
        if not isinstance(rule, dict):
            continue
        allowed = set(_normalized_list(rule.get("allows")))
        if dimension in allowed:
            return event
    return ""


def _chronology_conflicts(
    narrative: dict[str, Any],
    prior_concepts: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[str], str, str]:
    reasons: list[str] = []
    prior_segment_id = next(reversed(prior_concepts), "")
    approved_exception = ""

    milestone_entries = _ordered_contract_entries(contract, "milestone_order")
    milestone_ids = [item_id for item_id, _source in milestone_entries]
    highest_rank = -1
    for concept in prior_concepts.values():
        for milestone in _normalized_list(_narrative(concept).get("milestones")):
            if milestone in milestone_ids:
                highest_rank = max(highest_rank, milestone_ids.index(milestone))

    for milestone in _normalized_list(narrative.get("milestones")):
        if milestone not in milestone_ids:
            continue
        rank = milestone_ids.index(milestone)
        exception = _approved_chronology_exception(
            narrative, contract, "milestone_order",
        )
        if rank < highest_rank and not exception:
            previous_id = milestone_ids[highest_rank]
            source = milestone_entries[rank][1]
            reasons.append(
                f"milestone {milestone!r} reverses current milestone {previous_id!r}; "
                f"observed rank {rank} after rank {highest_rank} (source: {source})",
            )
        elif rank > highest_rank + 1:
            missing_id, source = milestone_entries[highest_rank + 1]
            reasons.append(
                f"milestone {milestone!r} observed before required predecessor "
                f"{missing_id!r} is unresolved (source: {source})",
            )
        else:
            approved_exception = approved_exception or exception
            highest_rank = max(highest_rank, rank)

    location_entries = _ordered_contract_entries(contract, "location_order")
    location_ids = [item_id for item_id, _source in location_entries]
    current_location = str(_normalize_semantic_value(narrative.get("location")))
    previous_location = ""
    for concept in reversed(list(prior_concepts.values())):
        candidate = str(_normalize_semantic_value(_narrative(concept).get("location")))
        if candidate in location_ids:
            previous_location = candidate
            break
    if current_location in location_ids and previous_location in location_ids:
        current_rank = location_ids.index(current_location)
        previous_rank = location_ids.index(previous_location)
        exception = _approved_chronology_exception(
            narrative, contract, "location_order",
        )
        if current_rank < previous_rank and not exception:
            source = location_entries[current_rank][1]
            reasons.append(
                f"location {current_location!r} reverses current location "
                f"{previous_location!r}; observed rank {current_rank} after rank "
                f"{previous_rank} (source: {source})",
            )
        else:
            approved_exception = approved_exception or exception

    terminal_contract = contract.get("terminal_states") or {}
    cast_states = narrative.get("cast_states") or {}
    if isinstance(terminal_contract, dict) and isinstance(cast_states, dict):
        for actor, raw_rule in terminal_contract.items():
            if not isinstance(raw_rule, dict):
                continue
            terminal_milestone = str(_normalize_semantic_value(raw_rule.get("milestone")))
            required_state = str(_normalize_semantic_value(raw_rule.get("state")))
            reset_event = str(_normalize_semantic_value(raw_rule.get("reset_event")))
            terminal_segment = ""
            for segment_id, concept in prior_concepts.items():
                prior_narrative = _narrative(concept)
                if terminal_milestone in _normalized_list(prior_narrative.get("milestones")):
                    terminal_segment = segment_id
                if reset_event in _normalized_list(prior_narrative.get("causal_events")):
                    terminal_segment = ""
            current_milestones = _normalized_list(narrative.get("milestones"))
            current_terminal = terminal_milestone in current_milestones
            if current_terminal:
                terminal_segment = terminal_segment or "current scene"
            normalized_states = {
                str(_normalize_semantic_value(key)): str(_normalize_semantic_value(value))
                for key, value in cast_states.items()
            }
            normalized_actor = str(_normalize_semantic_value(actor))
            if current_terminal and normalized_actor not in normalized_states:
                source = str(raw_rule.get("source") or "terminal_states")
                reasons.append(
                    f"terminal milestone {terminal_milestone!r} requires explicit "
                    f"cast state {required_state!r} for actor {normalized_actor!r} "
                    f"(source: {source})",
                )
                continue
            if not terminal_segment or normalized_actor not in normalized_states:
                continue
            observed_state = normalized_states[normalized_actor]
            causal_events = set(_normalized_list(narrative.get("causal_events")))
            if observed_state != required_state and reset_event not in causal_events:
                source = str(raw_rule.get("source") or "terminal_states")
                subject = (
                    f"terminal milestone {terminal_milestone!r}"
                    if terminal_segment == "current scene"
                    else f"actor {normalized_actor!r}"
                )
                reasons.append(
                    f"{subject} requires terminal state "
                    f"{required_state!r} from {terminal_segment}; observed "
                    f"{observed_state!r} without causal event {reset_event!r} "
                    f"(source: {source})",
                )
                prior_segment_id = terminal_segment
            elif observed_state != required_state:
                approved_exception = approved_exception or reset_event

    return reasons, prior_segment_id, approved_exception


def _chronology_evidence(
    segment_id: str,
    narrative: dict[str, Any],
    prior_concepts: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    allocation: dict[str, str] = {}
    for prior_id, concept in prior_concepts.items():
        for milestone in _normalized_list(_narrative(concept).get("milestones")):
            allocation.setdefault(milestone, prior_id)
    for milestone in _normalized_list(narrative.get("milestones")):
        allocation.setdefault(milestone, segment_id)
    _reasons, _prior_id, approved_exception = _chronology_conflicts(
        narrative, prior_concepts, contract,
    )
    return {
        "scene_order": [*prior_concepts, segment_id],
        "milestone_allocation": allocation,
        "validation_result": "accepted",
        "approved_exception": approved_exception or None,
    }


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


_CONTINUITY_STATE_MAPS = ("cast_states", "props", "transformation_states")


def _normalized_scalar(value: Any) -> str:
    # A missing field must stay absent from the snapshot so boundary merging
    # inherits the predecessor's exact value; str(None) would inject the
    # literal "None" sentinel and break every continuous handoff.
    if value is None:
        return ""
    return str(_normalize_semantic_value(value))


def _continuity_snapshot(narrative: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    location = _normalized_scalar(narrative.get("location"))
    if location:
        snapshot["location"] = location
    for field in _CONTINUITY_STATE_MAPS:
        value = narrative.get(field)
        if isinstance(value, dict):
            snapshot[field] = dict(_normalize_semantic_value(value))
    for field in ("action", "action_phase"):
        value = _normalized_scalar(narrative.get(field))
        if value:
            snapshot[field] = value
    return snapshot


def _continuity_events(narrative: dict[str, Any]) -> list[str]:
    events = []
    for field in ("transition_events", "causal_events", "reset_events"):
        events.extend(_normalized_list(narrative.get(field)))
    return list(dict.fromkeys(events))


def _merge_continuity_state(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(previous)
    for field, value in current.items():
        if field in _CONTINUITY_STATE_MAPS and isinstance(value, dict):
            merged[field] = {**(merged.get(field) or {}), **value}
        else:
            merged[field] = value
    return merged


def _narrative_outgoing_snapshot(narrative: dict[str, Any]) -> dict[str, Any]:
    # Mirror the annotated outgoing merge used by _adjacent_continuity_plan:
    # a scene's explicit `outgoing` block overrides its narrative-level state
    # fields. Raw fallbacks that ignored it made batch-time validation see a
    # different boundary than the annotated chronology gate enforced later.
    outgoing = _continuity_snapshot(narrative)
    explicit = narrative.get("outgoing")
    if isinstance(explicit, dict):
        outgoing = _merge_continuity_state(outgoing, _continuity_snapshot(explicit))
    return outgoing


def _transition_authorized(
    events: set[str],
    *,
    field: str,
    key: str,
    previous: str,
    current: str,
) -> bool:
    return bool(events.intersection({
        key,
        f"{key}:{previous}->{current}",
        f"{field}.{key}:{previous}->{current}",
    }))


def _adjacent_continuity_plan(
    segment_id: str,
    narrative: dict[str, Any],
    prior_concepts: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    predecessor_id = next(reversed(prior_concepts), "")
    previous_number = re.search(r"(\d+)$", predecessor_id)
    current_number = re.search(r"(\d+)$", segment_id)
    if (
        previous_number is not None
        and current_number is not None
        and int(current_number.group(1)) != int(previous_number.group(1)) + 1
    ):
        predecessor_id = ""
    previous_narrative = (
        _narrative(prior_concepts[predecessor_id]) if predecessor_id else {}
    )
    previous_validation = (
        prior_concepts.get(predecessor_id, {}).get("semantic_validation") or {}
        if predecessor_id and isinstance(prior_concepts.get(predecessor_id), dict)
        else {}
    )
    previous_continuity = previous_validation.get("continuity") or {}
    previous_outgoing = (
        deepcopy(previous_continuity.get("outgoing"))
        if isinstance(previous_continuity.get("outgoing"), dict)
        else _narrative_outgoing_snapshot(previous_narrative)
    )
    current_snapshot = _continuity_snapshot(narrative)
    explicit_incoming = narrative.get("incoming")
    incoming_delta = (
        _continuity_snapshot(explicit_incoming)
        if isinstance(explicit_incoming, dict)
        else {}
    )
    incoming = _merge_continuity_state(previous_outgoing, incoming_delta)
    outgoing = _merge_continuity_state(incoming, current_snapshot)
    explicit_outgoing = narrative.get("outgoing")
    if isinstance(explicit_outgoing, dict):
        outgoing = _merge_continuity_state(
            outgoing,
            _continuity_snapshot(explicit_outgoing),
        )

    transition = str(
        _normalize_semantic_value(narrative.get("transition_from_previous"))
        or "cut"
    )
    requires_continuation = bool(predecessor_id and transition == "continuous")
    return {
        "schema": "feverslop.narrative-continuity/v1",
        "scene_id": segment_id,
        "predecessor_id": predecessor_id or None,
        "transition": transition,
        "requires_continuation": requires_continuation,
        "continuation_intent": (
            str(outgoing.get("action") or incoming.get("action") or "continuous_action")
            if requires_continuation else None
        ),
        "transition_events": _continuity_events(narrative),
        "incoming": incoming,
        "outgoing": outgoing,
        "validation_result": "compatible",
    }


def _adjacent_continuity_conflicts(
    segment_id: str,
    narrative: dict[str, Any],
    prior_concepts: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[str], str]:
    plan = _adjacent_continuity_plan(
        segment_id,
        narrative,
        prior_concepts,
        contract,
    )
    predecessor_id = str(plan.get("predecessor_id") or "")
    if not predecessor_id:
        return [], ""
    previous = (
        prior_concepts.get(predecessor_id, {}).get("semantic_validation", {})
        .get("continuity", {})
        .get("outgoing")
        if isinstance(prior_concepts.get(predecessor_id), dict)
        else None
    )
    if not isinstance(previous, dict):
        previous = _narrative_outgoing_snapshot(_narrative(prior_concepts[predecessor_id]))
    incoming = plan["incoming"]
    outgoing = plan["outgoing"]
    events = set(plan["transition_events"])
    reasons: list[str] = []

    prop_orders = contract.get("prop_state_order") or {}
    for prop, previous_state in (previous.get("props") or {}).items():
        current_state = str((outgoing.get("props") or {}).get(prop) or previous_state)
        order = [
            str(_normalize_semantic_value(item))
            for item in (prop_orders.get(prop) or ())
        ] if isinstance(prop_orders, dict) else []
        if (
            previous_state in order
            and current_state in order
            and order.index(current_state) > order.index(previous_state) + 1
            and not _transition_authorized(
                events,
                field="props",
                key=prop,
                previous=previous_state,
                current=current_state,
            )
        ):
            reasons.append(
                f"{segment_id}.incoming.props.{prop}: {current_state!r} skips required "
                f"state after {predecessor_id} outgoing state {previous_state!r}",
            )

    allowed_locations = contract.get("actor_allowed_locations") or {}
    location = str(outgoing.get("location") or incoming.get("location") or "")
    if isinstance(allowed_locations, dict) and location:
        for actor, state in (outgoing.get("cast_states") or {}).items():
            allowed = {
                str(_normalize_semantic_value(item))
                for item in allowed_locations.get(actor) or ()
            }
            if allowed and state not in {"absent", "ascended_absent", "disappeared"} and location not in allowed:
                reasons.append(
                    f"{segment_id}.incoming.cast_states.{actor}: actor is not allowed "
                    f"at location {location!r}",
                )

    if plan["requires_continuation"]:
        for field in ("location", "action", "action_phase"):
            before = str(previous.get(field) or "")
            after = str(incoming.get(field) or "")
            if before and after and before != after:
                reasons.append(
                    f"{segment_id}.incoming.{field}: {after!r} is incompatible with "
                    f"continuous {predecessor_id} outgoing state {before!r}",
                )
        for field in _CONTINUITY_STATE_MAPS:
            for key, before in (previous.get(field) or {}).items():
                after = (incoming.get(field) or {}).get(key)
                if after is not None and after != before and not _transition_authorized(
                    events,
                    field=field,
                    key=key,
                    previous=str(before),
                    current=str(after),
                ):
                    reasons.append(
                        f"{segment_id}.incoming.{field}.{key}: {after!r} is incompatible "
                        f"with {predecessor_id} outgoing state {before!r}",
                    )
    return reasons, predecessor_id


def _continuity_boundary_state(value: Any, direction: str) -> dict[str, Any]:
    """Return the state a scene exposes at one boundary, mirroring validation."""
    if not isinstance(value, dict):
        return {}
    continuity = (value.get("semantic_validation") or {}).get("continuity") or {}
    state = continuity.get(direction)
    if isinstance(state, dict) and state:
        return state
    narrative = _narrative(value)
    if direction == "outgoing":
        return _narrative_outgoing_snapshot(narrative)
    explicit = narrative.get("incoming")
    return _continuity_snapshot(explicit) if isinstance(explicit, dict) else {}


def _boundary_context(
    target_ids: list[str],
    *,
    known: dict[str, Any],
    order_ids: list[str],
    excluded: set[str],
) -> dict[str, dict[str, Any]]:
    """Map each repair target to its accepted neighbours' exact boundary states.

    The repair prompt uses these snapshots as the canonical state vocabulary so
    regenerated scenes keep continuous boundaries token-identical instead of
    paraphrasing them. When the immediate neighbour is itself being repaired (or
    still missing), the lookup walks outward to the nearest accepted scene
    across the gap and marks it with ``neighbor_distance``; without this
    fallback a contiguous run of broken keys would repair every scene with no
    canonical vocabulary at all.
    """
    order = order_ids
    positions = {segment_id: index for index, segment_id in enumerate(order)}
    context: dict[str, dict[str, Any]] = {}
    for segment_id in target_ids:
        position = positions.get(segment_id)
        if position is None:
            continue
        entry: dict[str, Any] = {}
        for side, direction, offset in (
            ("predecessor", "outgoing", -1),
            ("successor", "incoming", 1),
        ):
            for step in range(1, len(order)):
                neighbor_position = position + offset * step
                if not 0 <= neighbor_position < len(order):
                    break
                neighbor_id = order[neighbor_position]
                if neighbor_id in excluded or neighbor_id not in known:
                    continue
                state = _continuity_boundary_state(known[neighbor_id], direction)
                if not state:
                    continue
                boundary = {"scene_id": neighbor_id, direction: state}
                if step > 1:
                    boundary["neighbor_distance"] = step
                entry[side] = boundary
                break
        if entry:
            context[segment_id] = entry
    return context


def _semantic_conflicts(
    value: dict[str, Any],
    prior_concepts: dict[str, Any],
    *,
    contract: dict[str, Any],
    segment_id: str = "current_scene",
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
        item for item in regressions
        if item[0] not in reset_events
        and not _approved_chronology_exception(
            narrative, contract, "prop_state_order",
        )
    ]

    chronology_reasons, chronology_segment, _approved_exception = _chronology_conflicts(
        narrative,
        prior_concepts,
        contract,
    )
    continuity_reasons, continuity_segment = _adjacent_continuity_conflicts(
        segment_id,
        narrative,
        prior_concepts,
        contract,
    )

    reasons = ["narrative.props must be an object"] if malformed_props else []
    prior_segment_id = matching_segment
    continuous_predecessor = (
        str(_normalize_semantic_value(narrative.get("transition_from_previous")))
        == "continuous"
        and matching_segment == next(reversed(prior_concepts), "")
    )
    if matching_segment and not authorized_signature and not continuous_predecessor:
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
    if chronology_reasons:
        prior_segment_id = prior_segment_id or chronology_segment
        reasons.extend(chronology_reasons)
    if continuity_reasons:
        prior_segment_id = prior_segment_id or continuity_segment
        reasons.extend(continuity_reasons)
    if not reasons:
        return {}
    return {"reasons": reasons, "prior_segment_id": prior_segment_id}
