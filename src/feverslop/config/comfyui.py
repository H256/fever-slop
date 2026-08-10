from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_str(raw: dict, key: str) -> str:
    """Extract a required string config field; raises ValueError if missing or non-string."""
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"ComfyUIModelOverride requires '{key}' field")
    return value


@dataclass(frozen=True)
class ComfyUIModelOverride:
    workflow: str
    node_id: str
    node_title: str
    input: str
    expected_value: str
    replacement: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ComfyUIModelOverride":
        return cls(
            workflow=_require_str(raw, "workflow"),
            node_id=_require_str(raw, "node_id"),
            node_title=_require_str(raw, "node_title"),
            input=_require_str(raw, "input"),
            expected_value=_require_str(raw, "expected_value"),
            replacement=_require_str(raw, "replacement"),
        )
