from __future__ import annotations

from collections.abc import Mapping


def repair_creative_fields(fields: Mapping[str, str], rejected_fields: list[str], replacements: Mapping[str, str]) -> dict[str, str]:
    """Apply bounded repairs only to rejected creative fields; locked fields are untouched."""
    result = {str(key): str(value) for key, value in fields.items()}
    for field in rejected_fields:
        key = str(field).strip()
        if key and key in replacements:
            result[key] = str(replacements[key]).strip()
    return result
