from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from feverslop.domain.scene_workspace import (
    SceneDocumentConflict,
    SceneLtxPromptField,
    SceneMedia,
)

__all__ = [
    "SceneDocumentConflict",
    "SceneDocumentPort",
    "SceneDocumentSnapshot",
    "SceneLtxPromptField",
    "SceneMediaPort",
]


@dataclass(frozen=True)
class SceneDocumentSnapshot:
    scenes: tuple[Mapping[str, Any], ...]
    revision: str

    def __post_init__(self) -> None:
        frozen_scenes: list[Mapping[str, Any]] = []
        for scene in self.scenes:
            if not isinstance(scene, Mapping):
                raise TypeError("Scene document scene must be a JSON object")
            frozen_scenes.append(_freeze_json(scene))
        object.__setattr__(
            self,
            "scenes",
            tuple(frozen_scenes),
        )

    def to_scenes(self) -> list[dict[str, Any]]:
        return [_thaw_json(scene) for scene in self.scenes]


class SceneDocumentPort(Protocol):
    def load(self, project_id: str) -> SceneDocumentSnapshot:
        """Load the canonical scene document and its optimistic-lock revision."""

    def patch_scene(
        self,
        project_id: str,
        scene_number: int,
        changes: Mapping[str, object],
        expected_revision: str,
    ) -> SceneDocumentSnapshot:
        """Apply validated canonical scene changes or raise a revision conflict."""


class SceneMediaPort(Protocol):
    def load_media(self, project_id: str) -> Mapping[int, SceneMedia]:
        """Return display-only media facts keyed by scene number."""


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Scene document contains non-JSON value: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
