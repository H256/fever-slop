from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def generate_movie_storyboard_page(*, project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    visual_plan = json.loads((project_dir / "movie" / "visual_plan.json").read_text(encoding="utf-8"))
    storyboard_dir = project_dir / "output" / "movie" / "storyboard"
    final_dir = storyboard_dir / "final"
    rows = [_shot_html(shot, final_dir=final_dir) for shot in visual_plan.get("shots", [])]
    output = storyboard_dir / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_page_html(rows), encoding="utf-8")
    return output


def _shot_html(shot: dict[str, Any], *, final_dir: Path) -> str:
    scene_number = int(shot.get("scene") or 0)
    image_name = f"scene_{scene_number:04}.png"
    image_path = final_dir / image_name
    image_html = f'<img src="final/{image_name}" alt="scene {scene_number:04}">' if image_path.exists() else '<div class="missing">Missing frame</div>'
    actor_ids = ", ".join(str(item) for item in shot.get("selected_actor_ids", []) if str(item).strip())
    return f"""
    <article class="shot">
      <div class="frame">{image_html}</div>
      <div class="meta">
        <h2>{_escape(str(shot.get("shot_id") or f"shot_{scene_number:04}"))}</h2>
        <p><strong>View</strong> {_escape(str(shot.get("view_id") or ""))}</p>
        <p><strong>Actors</strong> {_escape(actor_ids)}</p>
        <p><strong>Video</strong> {_escape(str(shot.get("video_prompt") or ""))}</p>
      </div>
    </article>
    """.strip()


def _page_html(rows: list[str]) -> str:
    body = "\n".join(rows) if rows else '<p class="empty">No movie visual shots.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Movie Storyboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .shot {{ display: grid; grid-template-columns: minmax(240px, 420px) 1fr; gap: 18px; padding: 18px 0; border-bottom: 1px solid #333; }}
    .frame {{ aspect-ratio: 16 / 9; background: #222; display: grid; place-items: center; }}
    img {{ width: 100%; height: 100%; object-fit: contain; }}
    h1, h2, p {{ margin: 0 0 10px; }}
    .meta {{ min-width: 0; }}
    .missing, .empty {{ color: #aaa; }}
  </style>
</head>
<body>
  <main>
    <h1>Movie Storyboard</h1>
    {body}
  </main>
</body>
</html>
"""


def _escape(value: str) -> str:
    return html.escape(value, quote=True)
