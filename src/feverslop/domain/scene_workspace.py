from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
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
    _raw_scene: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_scene(
        cls,
        scene: Mapping[str, Any],
        *,
        media: SceneMedia | None = None,
    ) -> SceneWorkspaceItem:
        raw_scene = deepcopy(dict(scene))
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
        return deepcopy(self._raw_scene)

    def to_scene(self) -> dict[str, Any]:
        return deepcopy(self._raw_scene)


@dataclass(frozen=True)
class SceneWorkspace:
    items: tuple[SceneWorkspaceItem, ...]

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
    values: list[Any] = []
    actor_ids = scene.get("actor_ids")
    if isinstance(actor_ids, (list, tuple)):
        values.extend(actor_ids)

    reference_ids = scene.get("reference_ids")
    if isinstance(reference_ids, Mapping):
        for value in reference_ids.values():
            if isinstance(value, (list, tuple)):
                values.extend(value)
            else:
                values.append(value)
    elif isinstance(reference_ids, (list, tuple)):
        values.extend(reference_ids)

    return tuple(dict.fromkeys(str(value) for value in values if value not in (None, "")))
