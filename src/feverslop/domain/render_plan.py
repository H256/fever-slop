from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.domain.canonical_render_plan import PromptRole, resolve_effective_role
from feverslop.domain.subject_directives import SubjectDirectivePlan
from feverslop.errors import FeverSlopDataError


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
    def from_dict(cls, data: dict) -> RenderScene:
        return cls(data=dict(data))

    def to_dict(self) -> dict:
        return dict(self.data)

    def effective_role(self, role: str, *, legacy_value: Any) -> Any:
        return resolve_effective_role(self.data, role, legacy_value=legacy_value)

    @property
    def scene_number(self) -> int:
        if "scene" not in self.data:
            raise FeverSlopDataError("render scene is missing required key: 'scene'")
        return int(self.data["scene"])

    @property
    def z_image_prompt(self) -> str:
        legacy = str(self.data.get("z_image", {}).get("prompt", ""))
        if not self._has_canonical_role(PromptRole.Z_IMAGE):
            return legacy
        return str(resolve_effective_role(
            self.data,
            PromptRole.Z_IMAGE,
            legacy_value=legacy,
        ))

    @property
    def width(self) -> int:
        return int(self.data.get("width", 0))

    @property
    def height(self) -> int:
        return int(self.data.get("height", 0))

    @property
    def video_prompt(self) -> str:
        h3 = self.data.get("h3", {})
        h3_prompt = str(h3.get("prompt") or "")
        if h3_prompt or self._has_canonical_role(PromptRole.H3_VIDEO):
            return str(resolve_effective_role(
                self.data,
                PromptRole.H3_VIDEO,
                legacy_value=h3_prompt,
            ))
        ltx = self.data.get("ltx", {})
        i2v_prompt = str(
            ltx.get("original_style_i2v_prompt")
            or ltx.get("i2v_prompt_from_t2i")
            or "",
        )
        if i2v_prompt or self._has_canonical_role(PromptRole.LTX_I2V):
            return str(resolve_effective_role(
                self.data,
                PromptRole.LTX_I2V,
                legacy_value=i2v_prompt,
            ))
        base_prompt = str(ltx.get("base_prompt") or "")
        if not self._has_canonical_role(PromptRole.LTX_BASE):
            return base_prompt
        return str(resolve_effective_role(
            self.data,
            PromptRole.LTX_BASE,
            legacy_value=base_prompt,
        ))

    def _has_canonical_role(self, role: PromptRole) -> bool:
        canonical = self.data.get("canonical")
        roles = canonical.get("roles") if isinstance(canonical, dict) else None
        return isinstance(roles, dict) and str(role) in roles

    @property
    def subject_directive_plan(self) -> SubjectDirectivePlan | None:
        payload = self.data.get("subject_directives")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise FeverSlopDataError("render scene subject_directives must be an object")
        return SubjectDirectivePlan.from_dict(payload)


@dataclass(frozen=True)
class RenderPlan:
    scenes: tuple[RenderScene, ...]

    @classmethod
    def from_dicts(cls, scenes: list[dict]) -> RenderPlan:
        return cls(tuple(RenderScene.from_dict(scene) for scene in scenes))

    def to_dicts(self) -> list[dict]:
        return [scene.to_dict() for scene in self.scenes]

    def select(
        self,
        *,
        scene_numbers: set[int] | None = None,
        limit: int | None = None,
    ) -> RenderPlan:
        scenes = self.scenes
        if scene_numbers is not None:
            by_technical_id = {
                str(
                    scene.data.get("technical_segment_id")
                    or scene.data.get("segment_id")
                    or "",
                ).strip(): scene
                for scene in scenes
            }
            selected = set(scene_numbers)
            changed = True
            while changed:
                changed = False
                for scene in scenes:
                    if scene.scene_number not in selected:
                        continue
                    predecessor_id = str(
                        scene.data.get("continuation_predecessor_id") or "",
                    ).strip()
                    predecessor = by_technical_id.get(predecessor_id)
                    if predecessor is not None and predecessor.scene_number not in selected:
                        selected.add(predecessor.scene_number)
                        changed = True
            scenes = tuple(
                scene
                for scene in scenes
                if scene.scene_number in selected
                or int(scene.data.get("semantic_scene", scene.scene_number)) in selected
            )
        if limit is not None:
            scenes = scenes[:limit]
        return RenderPlan(scenes)
