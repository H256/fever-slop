from __future__ import annotations

import copy
from typing import Any


class ReviewTimelineState:
    def __init__(self, scenes: list[dict[str, Any]], *, container_key: str | None, original: Any):
        self.scenes = scenes
        self.container_key = container_key
        self.original = original
        self.dirty = False
        self._undo: list[list[dict[str, Any]]] = []
        self._redo: list[list[dict[str, Any]]] = []

    @classmethod
    def from_document(cls, document: Any) -> ReviewTimelineState:
        container_key = None
        raw_scenes: list[Any] = []
        if isinstance(document, list):
            raw_scenes = document
        elif isinstance(document, dict):
            if isinstance(document.get("shots"), list):
                container_key = "shots"
                raw_scenes = document["shots"]
            elif isinstance(document.get("scenes"), list):
                container_key = "scenes"
                raw_scenes = document["scenes"]
        scenes = []
        for index, value in enumerate(raw_scenes):
            scene = copy.deepcopy(value) if isinstance(value, dict) else {}
            scene["scene"] = int(scene.get("scene", index + 1))
            scenes.append(scene)
        return cls(scenes, container_key=container_key, original=copy.deepcopy(document))

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def move(self, source_index: int, target_index: int) -> bool:
        if not (0 <= source_index < len(self.scenes) and 0 <= target_index < len(self.scenes)):
            return False
        if source_index == target_index:
            return False
        self._push_undo()
        scene = self.scenes.pop(source_index)
        self.scenes.insert(target_index, scene)
        self.dirty = True
        return True

    def trim(self, scene_number: int, raw_in_seconds: float, raw_out_seconds: float) -> bool:
        start = max(0.0, float(raw_in_seconds))
        end = float(raw_out_seconds)
        if end <= start:
            return False
        scene = next((item for item in self.scenes if int(item["scene"]) == int(scene_number)), None)
        if scene is None:
            return False
        self._push_undo()
        fps = float(scene.get("fps") or 24)
        edit = dict(scene.get("edit") or {})
        edit.update({
            "raw_in_frame": round(start * fps),
            "raw_out_frame": round(end * fps),
            "raw_in_seconds": start,
            "raw_out_seconds": end,
            "studio_stale": True,
            "studio_stale_reason": "clip trim changed",
        })
        scene["edit"] = edit
        self.dirty = True
        return True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self.scenes))
        self.scenes = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self.scenes))
        self.scenes = self._redo.pop()
        self.dirty = True
        return True

    def document(self) -> Any:
        if self.container_key is None:
            return copy.deepcopy(self.scenes)
        result = copy.deepcopy(self.original) if isinstance(self.original, dict) else {}
        result[self.container_key] = copy.deepcopy(self.scenes)
        return result

    def items(self, videos: list[str]) -> list[dict[str, Any]]:
        result = []
        start = 0.0
        for scene in self.scenes:
            scene_number = int(scene["scene"])
            fps = float(scene.get("fps") or 24)
            edit = dict(scene.get("edit") or {})
            duration = float(scene.get("duration_seconds") or 0)
            if "raw_in_frame" in edit and "raw_out_frame" in edit:
                duration = max(0.0, (float(edit["raw_out_frame"]) - float(edit["raw_in_frame"])) / fps)
            final_clip = find_scene_clip(videos, scene_number, raw=False)
            raw_clip = find_scene_clip(videos, scene_number, raw=True)
            result.append({
                "scene": scene_number,
                "start": start,
                "end": start + duration,
                "duration": duration,
                "clip": final_clip or raw_clip,
                "final_clip": final_clip,
                "raw_clip": raw_clip,
                "status": "final" if final_clip else "raw" if raw_clip else "missing",
                "preview": _preview(scene),
                "raw_in_seconds": float(edit.get("raw_in_seconds", edit.get("raw_in_frame", 0) / fps)),
                "raw_out_seconds": float(edit.get("raw_out_seconds", edit.get("raw_out_frame", duration * fps) / fps)),
                "stale": bool(edit.get("studio_stale")),
            })
            start += duration
        return result

    def mark_saved(self) -> None:
        self.original = self.document()
        self.dirty = False

    def _push_undo(self) -> None:
        self._undo.append(copy.deepcopy(self.scenes))
        self._undo = self._undo[-30:]
        self._redo.clear()


def find_scene_clip(videos: list[str], scene_number: int, *, raw: bool) -> str:
    padded = f"{scene_number:04d}"
    candidates = []
    for path in videos:
        if f"/scene_{padded}" not in path or "_debug/" in path:
            continue
        is_raw = "_raw" in path or "/raw/" in path
        if is_raw == raw:
            candidates.append(path)
    return next((path for path in candidates if "/final/" in path), candidates[0] if candidates else "")


def _preview(scene: dict[str, Any]) -> str:
    ltx = scene.get("ltx") if isinstance(scene.get("ltx"), dict) else {}
    metadata = scene.get("metadata") if isinstance(scene.get("metadata"), dict) else {}
    return str(
        ltx.get("base_prompt")
        or ltx.get("original_style_i2v_prompt")
        or scene.get("description")
        or scene.get("action")
        or metadata.get("lyrics")
        or ""
    )
