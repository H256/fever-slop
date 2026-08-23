import json
import tempfile
import unittest
from pathlib import Path

from feverslop.tools.storyboard_page import (
    _render_html,
    generate_storyboard_page,
    parse_scene_list,
)


def _scene(scene_number: int, base_concept: str = "A story beat.") -> dict:
    return {
        "scene": scene_number,
        "abs_start_seconds": 1.25,
        "abs_end_seconds": 4.5,
        "duration_seconds": 3.25,
        "z_image": {"prompt": "Z prompt <unsafe>"},
        "ltx": {
            "base_prompt": "Base prompt",
            "i2v_prompt_from_t2i": "I2V prompt",
            "prompt_relay": [
                {
                    "frame_start": 0,
                    "frame_end": 12,
                    "state": "singing",
                    "prompt": "Relay prompt",
                },
            ],
        },
        "metadata": {
            "type": "mixed",
            "lyrics": "Lyrics & words",
            "base_concept": base_concept,
            "camera_motion": "Slow push-in",
            "character_motion": "Raises one hand",
        },
    }


class StoryboardPageTests(unittest.TestCase):
    def test_generates_scene_blocks_with_clickable_images_and_caption(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            (storyboard_dir / "scene_0001.png").write_bytes(b"fake png")

            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([_scene(1)]), encoding="utf-8")
            output_html = storyboard_dir / "index.html"

            generate_storyboard_page(
                render_plan_path=render_plan,
                storyboard_dir=storyboard_dir,
                output_html=output_html,
                title="Review Page",
            )

            html = output_html.read_text(encoding="utf-8")

        self.assertIn("<h1>Review Page</h1>", html)
        self.assertIn("Scene 0001", html)
        self.assertIn('href="scene_0001.png"', html)
        self.assertIn('src="scene_0001.png"', html)
        self.assertIn("A story beat.", html)
        self.assertIn("Lyrics &amp; words", html)
        self.assertIn("Z prompt &lt;unsafe&gt;", html)
        self.assertIn("fontshare.com", html)
        self.assertIn("#6366F1", html)

    def test_page_container_uses_full_available_width_with_max_five_cards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            (storyboard_dir / "scene_0001.png").write_bytes(b"fake png")

            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([_scene(1)]), encoding="utf-8")
            output_html = storyboard_dir / "index.html"

            generate_storyboard_page(render_plan, storyboard_dir, output_html)

            html = output_html.read_text(encoding="utf-8")

        self.assertIn("width: 100%;", html)
        self.assertIn("max-width: none;", html)
        self.assertIn(
            "grid-template-columns: repeat(auto-fit, minmax(max(280px, calc((100vw - 144px) / 5)), 1fr));",
            html,
        )

    def test_page_includes_full_compact_mode_switch_and_prompt_tooltip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            (storyboard_dir / "scene_0001.png").write_bytes(b"fake png")

            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([_scene(1)]), encoding="utf-8")
            output_html = storyboard_dir / "index.html"

            generate_storyboard_page(render_plan, storyboard_dir, output_html)

            html = output_html.read_text(encoding="utf-8")

        self.assertIn('<body class="mode-full">', html)
        self.assertIn('<div class="mode-switch" aria-label="Storyboard view mode">', html)
        self.assertIn('data-mode-button="full"', html)
        self.assertIn('data-mode-button="compact"', html)
        self.assertIn("body.mode-compact .scene-header", html)
        self.assertIn("body.mode-compact .scene-content", html)
        self.assertIn('title="Z prompt &lt;unsafe&gt;"', html)
        self.assertIn("localStorage.setItem('storyboardPageMode', mode);", html)

    def test_falls_back_to_z_image_prompt_when_base_concept_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            (storyboard_dir / "scene_0001.png").write_bytes(b"fake png")

            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([_scene(1, base_concept="")]), encoding="utf-8")
            output_html = storyboard_dir / "index.html"

            generate_storyboard_page(render_plan, storyboard_dir, output_html)

            html = output_html.read_text(encoding="utf-8")

        self.assertIn("Z prompt &lt;unsafe&gt;", html)

    def test_missing_images_fail_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([_scene(1)]), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "scene_0001.png"):
                generate_storyboard_page(render_plan, storyboard_dir, storyboard_dir / "index.html")

    def test_allow_missing_images_renders_placeholder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([_scene(1)]), encoding="utf-8")
            output_html = storyboard_dir / "index.html"

            generate_storyboard_page(
                render_plan,
                storyboard_dir,
                output_html,
                allow_missing_images=True,
            )

            html = output_html.read_text(encoding="utf-8")

        self.assertIn("Missing image: scene_0001.png", html)

    def test_limit_and_scene_selector_filter_scene_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            for scene_number in (1, 2, 3):
                (storyboard_dir / f"scene_{scene_number:04}.png").write_bytes(b"fake png")

            render_plan = temp / "render_plan.json"
            render_plan.write_text(
                json.dumps([_scene(1), _scene(2), _scene(3)]),
                encoding="utf-8",
            )
            output_html = storyboard_dir / "index.html"

            generate_storyboard_page(
                render_plan,
                storyboard_dir,
                output_html,
                scene_numbers=parse_scene_list("2-3"),
                limit=1,
            )

            html = output_html.read_text(encoding="utf-8")

        self.assertNotIn("Scene 0001", html)
        self.assertIn("Scene 0002", html)
        self.assertNotIn("Scene 0003", html)

    def test_render_html_escapes_title_and_preserves_raw_blocks(self):
        """Jinja2 template: title auto-escaped, scenes_html not double-escaped, CSS/JS raw."""
        html_output = _render_html(
            title="Test & <Page>",
            scenes_html='<article class="scene-card">Card</article>',
            scene_count=1,
        )
        self.assertIn("<!doctype html>", html_output)
        self.assertIn("<h1>Test &amp; &lt;Page&gt;</h1>", html_output)
        self.assertIn("1 scene blocks", html_output)
        self.assertIn('<article class="scene-card">Card</article>', html_output)
        self.assertIn("--primary: #6366F1", html_output)
        self.assertIn("localStorage.setItem('storyboardPageMode', mode);", html_output)


if __name__ == "__main__":
    unittest.main()
