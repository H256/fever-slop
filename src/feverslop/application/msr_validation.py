"""Shared MSR prompt helpers and ``modules.vision()`` response validation.

Both the song (``msr_prompt_enrichment``) and movie (``movie_msr_enrichment``)
enrichment paths call a vision LLM to describe references and rewrite relay
prompts. The response parsing + relay validation used to be duplicated verbatim
in each module; that duplication is the concrete outgrowth tracked in #1182.
This module owns that logic once. It is a leaf module (no import of either
enrichment module) so both can depend on it without a cycle.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def clean_segment_prompt(prompt: str) -> str:
    """Collapse whitespace and strip frame-locking phrasing from a relay prompt."""
    cleaned = " ".join(str(prompt or "").replace("\n", " ").split())
    cleaned = re.sub(r"(?is)\bStart frame:\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\bLock the first frame\b.*?(?:\.|$)", "", cleaned)
    cleaned = re.sub(r"(?is)\bpreserve same shot\b", "", cleaned)
    cleaned = re.sub(r"(?is)\bpreserve the same shot\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
    return cleaned


def is_valid_segment_prompt(prompt: str, relay: dict) -> bool:
    """State-consistent relay prompt check (no frame-locking, lip-sync rules)."""
    lower = prompt.lower()
    banned = (
        "preserve same subject",
        "keep identity",
        "keep same subject",
        "lock first frame",
        "start frame",
    )
    if not prompt or any(text in lower for text in banned):
        return False
    state = str(relay.get("state") or "").strip().lower()
    if state == "singing":
        return "sing" in lower and ("lip sync" in lower or "lip-sync" in lower)
    if state == "dialogue":
        return ("speak" in lower or "talk" in lower or "say" in lower) and (
            "lip sync" in lower or "lip-sync" in lower
        )
    return "lip sync" not in lower and "lip-sync" not in lower


def validate_msr_vision_response(
    data: Any,
    *,
    expected_pairs: set[tuple[str, str]],
    relays: list[dict],
) -> tuple[dict[tuple[str, str], str], dict[int, str]] | None:
    """Parse and validate a ``modules.vision()`` response.

    ``expected_pairs`` is the set of ``(reference id, type)`` tuples the caller
    asked the vision model to describe; ``relays`` is the list of input relay
    specs each rewritten prompt is validated against (a single-element list for
    the movie single-relay path, the full relay list for the song path).

    Returns ``(descriptions, prompts)`` where ``descriptions`` maps
    ``(reference id, type)`` to its cleaned description and ``prompts`` maps each
    relay index to its cleaned, state-validated prompt. Returns ``None`` when the
    response is structurally invalid, references the wrong set, or any relay
    fails validation.
    """
    try:
        parsed_references = data.get("references")
        parsed_relays = data.get("relays")
        if not isinstance(parsed_references, list) or not isinstance(parsed_relays, list):
            raise ValueError("missing lists")
        descriptions: dict[tuple[str, str], str] = {}
        for item in parsed_references:
            pair = (str(item.get("id") or ""), str(item.get("type") or ""))
            description = str(item.get("description") or "").strip()
            if not all(pair) or not description or pair in descriptions:
                raise ValueError("invalid reference")
            descriptions[pair] = description
        if set(descriptions) != expected_pairs:
            raise ValueError("reference mismatch")
        prompts: dict[int, str] = {}
        for item in parsed_relays:
            index = int(item["index"])
            if index in prompts or not 0 <= index < len(relays):
                raise ValueError("invalid relay index")
            prompt = clean_segment_prompt(str(item.get("prompt") or ""))
            if not is_valid_segment_prompt(prompt, relays[index]):
                raise ValueError("invalid relay prompt")
            prompts[index] = prompt
        if set(prompts) != set(range(len(relays))):
            raise ValueError("missing relay index")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        logger.debug("MSR vision response failed validation: %s", exc)
        return None
    return descriptions, prompts
