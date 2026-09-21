"""Plain-language guidance for H3 scene block reasons.

Maps raw reason codes from H3 readiness checks to human-readable causes
with concrete next actions so a blocked scene stops being an opaque
``h3.fallback.plan_missing``.
"""

from __future__ import annotations


def h3_block_guide(
    reason_codes: list[str],
    *,
    truncation_suspected: bool = False,
) -> str:
    """Return plain-language guidance for a blocked H3 scene.

    Parameters
    ----------
    reason_codes:
        Raw reason codes from ``readiness["reason_codes"]``.
    truncation_suspected:
        Whether the truncation heuristic (unclosed JSON in the LLM output)
        flagged the last planner attempt.

    Returns a single-line guidance string suitable for console output or
    embedding in an error message.  When the cause is known the guidance
    ends with a concrete action (raise budget, run ``--replan-failed``,
    correct inputs).  Unknown codes get a generic fallback.
    """
    if not reason_codes:
        return "The scene failed preparation; retry or correct its inputs."

    codes_lower = {c.lower() for c in reason_codes}
    causes: list[str] = []
    actions: list[str] = []

    if "h3.fallback.plan_missing" in codes_lower or "h3.plan.missing" in codes_lower:
        causes.append("the planner did not produce a usable typed plan")
    elif "h3.recovery.exhausted" in codes_lower:
        causes.append("all recovery attempts (generation, repair, fallback) are exhausted")
    elif "h3.prompt.truncated" in codes_lower:
        causes.append("the planner response was truncated by the output token budget")
    elif "h3.fact.missing" in codes_lower:
        causes.append("a locked scene fact is absent from the generated prompt")
    elif "h3.fact.conflict" in codes_lower:
        causes.append("the generated prompt contradicts a locked scene fact")
    elif "h3.performance.lyrics_mismatch" in codes_lower:
        causes.append("the vocal performance words do not match the expected lyrics")
    else:
        # Unknown code — generic fallback
        causes.append(f"scene validation failed ({', '.join(reason_codes[:2])})")

    cause_text = "; ".join(causes)

    # Determine the best actionable advice.
    if truncation_suspected:
        actions.append(
            "the model response was truncated; "
            "raise ``llm.prompt_planner_max_tokens`` in config (e.g. to 131072) or retry."
        )
    elif any("plan_missing" in c or "plan_invalid" in c for c in codes_lower):
        actions.append(
            "the planner keeps failing on the same input — "
            "retry with ``--replan-failed`` (fresh recovery budget) or correct the scene's "
            "lyrics, timing, or cast."
        )
    elif "exhausted" in cause_text:
        actions.append(
            "all internal retries are spent — correct the scene's inputs "
            "(lyrics, timing, cast) and retry."
        )
    else:
        actions.append(
            "correct the scene's inputs (lyrics, timing, cast) and retry."
        )

    action_text = "; ".join(actions)
    return f"{cause_text}. {action_text}"
