from __future__ import annotations

from pathlib import Path
from typing import Any
import json


class JsonArtifactStore:
    def read_json(self, path: str | Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path, data: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_render_plan(self, path: str | Path) -> list[dict]:
        data = self.read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Render plan must be a JSON list: {path}")
        return data

    def write_render_plan(self, path: str | Path, scenes: list[dict]) -> Path:
        return self.write_json(path, scenes)
