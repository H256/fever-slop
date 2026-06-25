from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSet:
    z_image_prompt: str
    i2v_prompt: str


@dataclass(frozen=True)
class RenderResult:
    scene_number: int
    output_path: Path

    def as_manifest_entry(self) -> dict:
        return {
            "scene": self.scene_number,
            "output_path": str(self.output_path),
        }


@dataclass(frozen=True)
class RenderScene:
    data: dict

    @classmethod
    def from_dict(cls, data: dict) -> "RenderScene":
        return cls(data=dict(data))

    def to_dict(self) -> dict:
        return dict(self.data)

    @property
    def scene_number(self) -> int:
        return int(self.data["scene"])

    @property
    def z_image_prompt(self) -> str:
        return str(self.data.get("z_image", {}).get("prompt", ""))

    @property
    def width(self) -> int:
        return int(self.data.get("width", 0))

    @property
    def height(self) -> int:
        return int(self.data.get("height", 0))

    @property
    def video_prompt(self) -> str:
        ltx = self.data.get("ltx", {})
        return str(
            ltx.get("original_style_i2v_prompt")
            or ltx.get("i2v_prompt_from_t2i")
            or ltx.get("base_prompt")
            or ""
        )


@dataclass(frozen=True)
class RenderPlan:
    scenes: list[RenderScene]

    @classmethod
    def from_dicts(cls, scenes: list[dict]) -> "RenderPlan":
        return cls([RenderScene.from_dict(scene) for scene in scenes])

    def to_dicts(self) -> list[dict]:
        return [scene.to_dict() for scene in self.scenes]

    def select(
        self,
        *,
        scene_numbers: set[int] | None = None,
        limit: int | None = None,
    ) -> "RenderPlan":
        scenes = self.scenes
        if scene_numbers is not None:
            scenes = [scene for scene in scenes if scene.scene_number in scene_numbers]
        if limit is not None:
            scenes = scenes[:limit]
        return RenderPlan(list(scenes))
