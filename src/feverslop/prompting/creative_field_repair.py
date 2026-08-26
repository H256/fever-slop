from __future__ import annotations

from collections.abc import Mapping

from feverslop.prompting.dspy_h3_models import CreativeFieldIssue, CreativeShotPayload


def repair_creative_fields(fields: Mapping[str, str], rejected_fields: list[str], replacements: Mapping[str, str]) -> dict[str, str]:
    """Apply bounded repairs only to rejected creative fields; locked fields are untouched."""
    result = {str(key): str(value) for key, value in fields.items()}
    for field in rejected_fields:
        key = str(field).strip()
        if key and key in replacements:
            result[key] = str(replacements[key]).strip()
    return result


def repair_creative_payloads(
    shots: list[CreativeShotPayload],
    field_issues: list[CreativeFieldIssue | Mapping[str, str]],
    replacements: Mapping[tuple[str, str], str],
) -> tuple[CreativeShotPayload, ...]:
    """Apply replacements only to judge-addressed creative fields."""
    by_id = {shot.shot_id: shot for shot in shots}
    for raw_issue in field_issues:
        issue = raw_issue if isinstance(raw_issue, CreativeFieldIssue) else CreativeFieldIssue.model_validate(raw_issue)
        shot = by_id.get(issue.shot_id)
        replacement = replacements.get((issue.shot_id, issue.field))
        if shot is None or replacement is None or not str(replacement).strip():
            continue
        by_id[issue.shot_id] = shot.model_copy(update={issue.field: str(replacement).strip()})
    return tuple(by_id[shot.shot_id] for shot in shots)
