from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class WorkflowSelection:
    path: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path, *, root: str | Path) -> WorkflowSelection:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Selected workflow does not exist: {resolved}")
        base = Path(root).resolve()
        try:
            label = resolved.relative_to(base).as_posix()
        except ValueError:
            label = resolved.as_posix()
        return cls(
            path=label,
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ProjectRenderSettings:
    width: int
    height: int
    video_workflow: WorkflowSelection | None = None
    reference_hero_workflow: WorkflowSelection | None = None
    reference_edit_workflow: WorkflowSelection | None = None
    reference_generation: str = "image_views"
    reference_sequence_workflow: WorkflowSelection | None = None

    def apply_to_scene(self, scene: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(scene))
        result["width"] = int(self.width)
        result["height"] = int(self.height)

        render_settings = result.get("render_settings")
        render_settings = deepcopy(dict(render_settings)) if isinstance(render_settings, Mapping) else {}
        if self.video_workflow is None:
            render_settings.pop("video_workflow", None)
        else:
            render_settings["video_workflow"] = self.video_workflow.to_dict()
        if render_settings:
            result["render_settings"] = render_settings
        else:
            result.pop("render_settings", None)

        references = result.get("references")
        references = deepcopy(dict(references)) if isinstance(references, Mapping) else {}
        reference_payload = {
            "generation": self.reference_generation,
            "hero": self.reference_hero_workflow.to_dict() if self.reference_hero_workflow else None,
            "edit": self.reference_edit_workflow.to_dict() if self.reference_edit_workflow else None,
            "sequence": (
                self.reference_sequence_workflow.to_dict()
                if self.reference_generation == "sequence_sheet" and self.reference_sequence_workflow
                else None
            ),
        }
        if (
            self.reference_generation != "image_views"
            or self.reference_hero_workflow is not None
            or self.reference_edit_workflow is not None
            or self.reference_sequence_workflow is not None
        ):
            payload = reference_payload
            references["generator_fingerprint"] = _fingerprint(payload)
        else:
            references.pop("generator_fingerprint", None)
        if references:
            result["references"] = references
        else:
            result.pop("references", None)
        return result

    def apply_to_scenes(self, scenes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.apply_to_scene(scene) for scene in scenes]


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
