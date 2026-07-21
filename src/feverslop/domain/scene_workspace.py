from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


SceneDisplayStatus = Literal["missing", "planned", "rendered", "failed"]


@dataclass(frozen=True)
class SceneMedia:
    thumbnail_path: str | None = None
    workflow_path: str | None = None
    video_path: str | None = None
    failure_message: str | None = None

    @property
    def status(self) -> SceneDisplayStatus:
        if self.failure_message:
            return "failed"
        if self.video_path:
            return "rendered"
        if self.workflow_path:
            return "planned"
        return "missing"


@dataclass(frozen=True)
class SceneWorkspaceItem:
    scene_number: int
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    performance_state: str = ""
    shot_description: str = ""
    image_prompt: str = ""
    video_prompt: str = ""
    reference_ids: tuple[str, ...] = ()
    media: SceneMedia = field(default_factory=SceneMedia)
    _raw_scene: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_ids", tuple(self.reference_ids))
        object.__setattr__(self, "_raw_scene", _freeze_json(self._raw_scene))

    @classmethod
    def from_scene(
        cls,
        scene: Mapping[str, Any],
        *,
        media: SceneMedia | None = None,
    ) -> SceneWorkspaceItem:
        raw_scene = dict(scene)
        ltx = raw_scene.get("ltx")
        ltx = ltx if isinstance(ltx, Mapping) else {}
        z_image = raw_scene.get("z_image")
        z_image = z_image if isinstance(z_image, Mapping) else {}
        return cls(
            scene_number=int(raw_scene["scene"]),
            start_seconds=_float_value(raw_scene, "abs_start_seconds", "start"),
            end_seconds=_float_value(raw_scene, "abs_end_seconds", "end"),
            performance_state=str(raw_scene.get("type") or ""),
            shot_description=str(raw_scene.get("shot_description") or ""),
            image_prompt=str(z_image.get("prompt") or ""),
            video_prompt=str(
                ltx.get("original_style_i2v_prompt")
                or ltx.get("i2v_prompt_from_t2i")
                or ltx.get("base_prompt")
                or ""
            ),
            reference_ids=_reference_ids(raw_scene),
            media=media or SceneMedia(),
            _raw_scene=raw_scene,
        )

    @property
    def status(self) -> SceneDisplayStatus:
        return self.media.status

    @property
    def raw_scene(self) -> dict[str, Any]:
        return _thaw_json(self._raw_scene)

    def to_scene(self) -> dict[str, Any]:
        return _thaw_json(self._raw_scene)


@dataclass(frozen=True)
class SceneWorkspace:
    items: tuple[SceneWorkspaceItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    @classmethod
    def from_scenes(
        cls,
        scenes: Iterable[Mapping[str, Any]],
        *,
        media_by_scene: Mapping[int, SceneMedia] | None = None,
    ) -> SceneWorkspace:
        media_by_scene = media_by_scene or {}
        items: list[SceneWorkspaceItem] = []
        seen: set[int] = set()
        for scene in scenes:
            scene_number = int(scene["scene"])
            if scene_number in seen:
                raise ValueError(f"Duplicate scene number: {scene_number}")
            seen.add(scene_number)
            items.append(
                SceneWorkspaceItem.from_scene(
                    scene,
                    media=media_by_scene.get(scene_number),
                )
            )
        return cls(items=tuple(items))

    def to_scenes(self) -> list[dict[str, Any]]:
        return [item.to_scene() for item in self.items]


def _float_value(scene: Mapping[str, Any], preferred: str, fallback: str) -> float:
    value = scene.get(preferred)
    if value is None:
        value = scene.get(fallback)
    return float(value or 0.0)


def _reference_ids(scene: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    _append_reference_values(values, scene.get("actor_ids"))

    references = scene.get("references")
    if isinstance(references, Mapping):
        for field_name in ("actor_ids", "location_id", "prop_ids"):
            _append_reference_values(values, references.get(field_name))

    movie_references = scene.get("reference_ids")
    if isinstance(movie_references, Mapping):
        for field_name in ("actors", "location", "props"):
            _append_reference_values(values, movie_references.get(field_name))
    else:
        _append_reference_values(values, movie_references)

    return tuple(dict.fromkeys(values))


def _append_reference_values(target: list[str], value: Any) -> None:
    candidates = value if isinstance(value, (list, tuple)) else (value,)
    target.extend(candidate for candidate in candidates if isinstance(candidate, str) and candidate)


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
    raise TypeError(f"Raw scene payload contains non-JSON value: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
