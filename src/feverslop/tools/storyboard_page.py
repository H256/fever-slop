from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


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
            "</div>"
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
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <link rel="preconnect" href="https://api.fontshare.com">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://api.fontshare.com/v2/css?f[]=general-sans@600,700&display=swap" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #6366F1;
      --primary-hover: #4F46E5;
      --background: #FAFAFA;
      --surface: #FFFFFF;
      --text-primary: #0A0A0A;
      --text-secondary: #6B6B6B;
      --neutral: #9C9C9C;
      --border: #E8E8EC;
      --warning: #F59E0B;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--background);
      color: var(--text-primary);
      font-family: "DM Sans", Arial, sans-serif;
      font-size: 15px;
      line-height: 1.5;
    }}

    a {{
      color: var(--primary);
    }}

    a:focus-visible,
    summary:focus-visible {{
      outline: none;
      box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
      border-radius: 6px;
    }}

    .page {{
      width: 100%;
      max-width: none;
      margin: 0 auto;
      padding: 48px 24px 64px;
    }}

    .page-header {{
      margin-bottom: 32px;
    }}

    h1,
    h2 {{
      font-family: "General Sans", Inter, Arial, sans-serif;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin: 0;
    }}

    h1 {{
      font-size: clamp(40px, 7vw, 72px);
      line-height: 0.95;
    }}

    .page-meta {{
      color: var(--text-secondary);
      margin-top: 12px;
    }}

    .header-row {{
      align-items: end;
      display: flex;
      gap: 24px;
      justify-content: space-between;
    }}

    .mode-switch {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      display: inline-flex;
      padding: 4px;
    }}

    .mode-button {{
      background: transparent;
      border: 0;
      border-radius: 6px;
      color: var(--text-secondary);
      cursor: pointer;
      font-family: "DM Sans", Arial, sans-serif;
      font-size: 13px;
      font-weight: 500;
      min-height: 32px;
      padding: 0 12px;
      transition: background 200ms ease, color 200ms ease, transform 200ms ease;
    }}

    .mode-button:hover {{
      color: var(--primary-hover);
      transform: translateY(-1px);
    }}

    .mode-button:focus-visible {{
      outline: none;
      box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
    }}

    body.mode-full .mode-button[data-mode-button="full"],
    body.mode-compact .mode-button[data-mode-button="compact"] {{
      background: var(--primary);
      color: #FFFFFF;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(max(280px, calc((100vw - 144px) / 5)), 1fr));
      gap: 24px;
    }}

    .scene-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      transition: transform 200ms ease, box-shadow 200ms ease;
    }}

    .scene-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }}

    .scene-header {{
      padding: 20px 20px 16px;
      border-bottom: 1px solid var(--border);
    }}

    h2 {{
      font-size: 24px;
      line-height: 1.1;
    }}

    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}

    .chip {{
      border-radius: 9999px;
      background: #F3F4F6;
      color: var(--text-secondary);
      font-size: 12px;
      padding: 4px 12px;
    }}

    .image-link {{
      display: block;
      background: #F3F4F6;
      line-height: 0;
    }}

    .image-link:hover {{
      outline: 2px solid var(--primary);
      outline-offset: -2px;
    }}

    img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }}

    .missing-image {{
      display: grid;
      min-height: 220px;
      place-items: center;
      background: #FFF7ED;
      color: var(--warning);
      font-weight: 500;
    }}

    .scene-content {{
      padding: 20px;
    }}

    .caption {{
      margin: 0 0 20px;
      color: var(--text-primary);
      font-size: 15px;
    }}

    .field {{
      border-top: 1px solid var(--border);
      padding-top: 12px;
      margin-top: 12px;
    }}

    .field-label {{
      color: var(--neutral);
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}

    .field-value {{
      color: var(--text-secondary);
    }}

    details {{
      border-top: 1px solid var(--border);
      margin-top: 12px;
      padding-top: 12px;
    }}

    summary {{
      color: var(--primary);
      cursor: pointer;
      font-weight: 500;
    }}

    summary:hover {{
      color: var(--primary-hover);
    }}

    .code {{
      color: var(--text-secondary);
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      margin-top: 8px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }}

    .relay-row {{
      border-top: 1px solid var(--border);
      margin-top: 12px;
      padding-top: 12px;
    }}

    .relay-meta {{
      color: var(--neutral);
      font-size: 12px;
      margin-bottom: 4px;
    }}

    body.mode-compact .scene-header,
    body.mode-compact .scene-content {{
      display: none;
    }}

    body.mode-compact .image-link:hover {{
      outline-offset: -3px;
    }}

    @media (max-width: 720px) {{
      .header-row {{
        align-items: start;
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body class="mode-full">
  <main class="page">
    <header class="page-header">
      <div class="header-row">
        <h1>{_escape(title)}</h1>
        <div class="mode-switch" aria-label="Storyboard view mode">
          <button class="mode-button" type="button" data-mode-button="full" aria-pressed="true">Full</button>
          <button class="mode-button" type="button" data-mode-button="compact" aria-pressed="false">Compact</button>
        </div>
      </div>
      <div class="page-meta">{scene_count} scene blocks</div>
    </header>
    <section class="grid">
      {scenes_html}
    </section>
  </main>
  <script>
    (() => {{
      const buttons = Array.from(document.querySelectorAll('[data-mode-button]'));
      const setMode = (mode) => {{
        document.body.classList.toggle('mode-full', mode === 'full');
        document.body.classList.toggle('mode-compact', mode === 'compact');
        buttons.forEach((button) => {{
          button.setAttribute('aria-pressed', String(button.dataset.modeButton === mode));
        }});
        localStorage.setItem('storyboardPageMode', mode);
      }};
      const savedMode = localStorage.getItem('storyboardPageMode');
      if (savedMode === 'compact') {{
        setMode('compact');
      }}
      buttons.forEach((button) => {{
        button.addEventListener('click', () => setMode(button.dataset.modeButton));
      }});
    }})();
  </script>
</body>
</html>
"""


def generate_storyboard_page(
    render_plan_path: str | Path,
    storyboard_dir: str | Path,
    output_html: str | Path | None = None,
    title: str = "Storyboard Review",
    scene_numbers: set[int] | None = None,
    limit: int | None = None,
    allow_missing_images: bool = False,
) -> Path:
    render_plan_path = Path(render_plan_path)
    storyboard_dir = Path(storyboard_dir)
    output_html = Path(output_html) if output_html else storyboard_dir / "index.html"

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
        description="Generate a static HTML storyboard review page from render_plan.json."
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
