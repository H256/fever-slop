from __future__ import annotations

import json
from pathlib import Path

from feverslop.path_utils import coerce_local_path


def parse_scene_list(value: str | None) -> set[int] | None:
    if not value:
        return None

    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"Scene range is reversed: {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def load_render_plan_subset(
    render_plan_path: str | Path,
    scene_numbers: set[int] | None,
    limit: int | None,
) -> list[dict]:
    render_plan = json.loads(coerce_local_path(render_plan_path).read_text(encoding="utf-8"))
    if scene_numbers is not None:
        render_plan = [scene for scene in render_plan if int(scene["scene"]) in scene_numbers]
    if limit is not None:
        render_plan = render_plan[:limit]
    return render_plan
