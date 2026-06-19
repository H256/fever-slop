from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
            workflow=str(raw["workflow"]),
            node_id=str(raw["node_id"]),
            node_title=str(raw["node_title"]),
            input=str(raw["input"]),
            expected_value=str(raw["expected_value"]),
            replacement=str(raw["replacement"]),
        )
