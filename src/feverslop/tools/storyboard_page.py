from __future__ import annotations

import argparse
import html
import json
import os
from importlib.resources import files as _files
from pathlib import Path

from jinja2 import BaseLoader, Environment, select_autoescape

from feverslop.path_utils import coerce_local_path

_RENDER_ENV = Environment(
    loader=BaseLoader(),
    autoescape=select_autoescape(default=True),
)


def parse_scene_list(value: str | None) -> set[int] | None:
    if not value:
        return None

    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            result.update(range(start, end + 1))
        else:
            result.add(int(part))

    return result


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _format_seconds(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return ""


def _select_scenes(
    render_plan: list[dict],
    scene_numbers: set[int] | None,
    limit: int | None,
) -> list[dict]:
    scenes = render_plan
    if scene_numbers is not None:
        scenes = [scene for scene in scenes if int(scene["scene"]) in scene_numbers]
    if limit is not None:
        scenes = scenes[:limit]
    return scenes


def _relative_href(target: Path, output_html: Path) -> str:
    relative = os.path.relpath(target, start=output_html.parent)
    return Path(relative).as_posix()


def _field(label: str, value: object, code: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    class_name = "field-value code" if code else "field-value"
    return (
        '<div class="field">'
        f'<div class="field-label">{_escape(label)}</div>'
        f'<div class="{class_name}">{_escape(text)}</div>'
        "</div>"
    )


def _relay_details(relay_prompts: list[dict]) -> str:
    if not relay_prompts:
        return ""

    rows = []
    for relay in relay_prompts:
        frame_start = _escape(relay.get("frame_start", ""))
        frame_end = _escape(relay.get("frame_end", ""))
        state = _escape(relay.get("state", ""))
        prompt = _escape(relay.get("prompt", ""))
        rows.append(
            '<div class="relay-row">'
            f'<div class="relay-meta">{frame_start}-{frame_end} / {state}</div>'
            f'<div class="code">{prompt}</div>'
            "</div>",
        )

    return (
        "<details>"
        "<summary>Relay prompts</summary>"
        f"{''.join(rows)}"
        "</details>"
    )


def _scene_card(scene: dict, storyboard_dir: Path, output_html: Path, allow_missing_images: bool) -> str:
    scene_number = int(scene["scene"])
    metadata = scene.get("metadata", {}) or {}
    z_image = scene.get("z_image", {}) or {}
    ltx = scene.get("ltx", {}) or {}

    image_path = storyboard_dir / f"scene_{scene_number:04}.png"
    image_name = image_path.name
    if not image_path.exists() and not allow_missing_images:
        raise FileNotFoundError(f"Missing storyboard image: {image_path}")

    image_href = _escape(_relative_href(image_path, output_html))
    caption = str(metadata.get("base_concept") or z_image.get("prompt") or "").strip()
    start = _format_seconds(scene.get("abs_start_seconds"))
    end = _format_seconds(scene.get("abs_end_seconds"))
    duration = _format_seconds(scene.get("duration_seconds"))
    scene_type = str(metadata.get("type") or "").strip()

    chips = []
    if start and end:
        chips.append(f"{_escape(start)} - {_escape(end)}")
    if duration:
        chips.append(f"{_escape(duration)}")
    if scene_type:
        chips.append(_escape(scene_type))
    chips_html = "".join(f'<span class="chip">{chip}</span>' for chip in chips)

    if image_path.exists():
        prompt_title = _escape(z_image.get("prompt", ""))
        image_html = (
            f'<a class="image-link" href="{image_href}" target="_blank" rel="noopener" title="{prompt_title}">'
            f'<img src="{image_href}" alt="Scene {scene_number:04} storyboard frame">'
            "</a>"
        )
    else:
        image_html = f'<div class="missing-image">Missing image: {_escape(image_name)}</div>'

    prompt_details = (
        "<details>"
        "<summary>Z-Image / T2I prompt</summary>"
        f'<div class="code">{_escape(z_image.get("prompt", ""))}</div>'
        "</details>"
        "<details>"
        "<summary>LTX / I2V prompt</summary>"
        f'<div class="code">{_escape(ltx.get("i2v_prompt_from_t2i") or ltx.get("original_style_i2v_prompt") or ltx.get("base_prompt") or "")}</div>'
        "</details>"
        f'{_relay_details(ltx.get("prompt_relay", []) or [])}'
    )

    return (
        '<article class="scene-card">'
        '<div class="scene-header">'
        f"<h2>Scene {scene_number:04}</h2>"
        f'<div class="chips">{chips_html}</div>'
        "</div>"
        f"{image_html}"
        '<div class="scene-content">'
        f'<p class="caption">{_escape(caption)}</p>'
        f'{_field("Lyrics", metadata.get("lyrics"))}'
        f'{_field("Camera motion", metadata.get("camera_motion"))}'
        f'{_field("Character motion", metadata.get("character_motion"))}'
        f"{prompt_details}"
        "</div>"
        "</article>"
    )


def _render_html(title: str, scenes_html: str, scene_count: int) -> str:
    template_text = _files("feverslop.tools.templates").joinpath("storyboard.html").read_text(encoding="utf-8")
    template = _RENDER_ENV.from_string(template_text)
    # scenes_html is pre-rendered HTML from _scene_card; mark safe to avoid double-escaping
    return template.render(title=title, scenes_html=scenes_html, scene_count=scene_count)


def generate_storyboard_page(
    render_plan_path: str | Path,
    storyboard_dir: str | Path,
    output_html: str | Path | None = None,
    title: str = "Storyboard Review",
    scene_numbers: set[int] | None = None,
    limit: int | None = None,
    allow_missing_images: bool = False,
) -> Path:
    render_plan_path = coerce_local_path(render_plan_path)
    storyboard_dir = coerce_local_path(storyboard_dir)
    output_html = coerce_local_path(output_html) if output_html else storyboard_dir / "index.html"

    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
    scenes = _select_scenes(render_plan, scene_numbers=scene_numbers, limit=limit)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    scenes_html = "\n      ".join(
        _scene_card(
            scene=scene,
            storyboard_dir=storyboard_dir,
            output_html=output_html,
            allow_missing_images=allow_missing_images,
        )
        for scene in scenes
    )
    output_html.write_text(
        _render_html(title=title, scenes_html=scenes_html, scene_count=len(scenes)),
        encoding="utf-8",
    )
    return output_html


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML storyboard review page from render_plan.json.",
    )
    parser.add_argument("--render-plan", required=True)
    parser.add_argument("--storyboard-dir", required=True)
    parser.add_argument("--output-html", default=None)
    parser.add_argument("--title", default="Storyboard Review")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenes", default=None, help="Example: 1,2,5-8")
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_html = generate_storyboard_page(
        render_plan_path=args.render_plan,
        storyboard_dir=args.storyboard_dir,
        output_html=args.output_html,
        title=args.title,
        scene_numbers=parse_scene_list(args.scenes),
        limit=args.limit,
        allow_missing_images=args.allow_missing_images,
    )
    print(output_html)


if __name__ == "__main__":
    main()
