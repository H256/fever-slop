from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from feverslop.domain.scene_workspace import SceneMedia


class SceneLtxPromptField(str, Enum):
    ORIGINAL_STYLE_I2V_PROMPT = "original_style_i2v_prompt"
    I2V_PROMPT_FROM_T2I = "i2v_prompt_from_t2i"
    BASE_PROMPT = "base_prompt"


@dataclass(frozen=True)
class SceneDocumentSnapshot:
    scenes: tuple[Mapping[str, Any], ...]
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenes",
            tuple(_freeze_json(scene) for scene in self.scenes),
        )

    def to_scenes(self) -> list[dict[str, Any]]:
        return [_thaw_json(scene) for scene in self.scenes]


class SceneDocumentConflict(RuntimeError):
    def __init__(
        self,
        project_id: str,
        expected_revision: str,
        actual_revision: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        detail = f"Scene document changed for project {project_id!r}"
        if actual_revision is not None:
            detail += (
                f" (expected revision {expected_revision!r}, "
                f"found {actual_revision!r})"
            )
        super().__init__(detail)


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
