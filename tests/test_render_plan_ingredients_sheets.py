import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from feverslop.application.render_plan_ingredients_sheets import enrich_render_plan_with_ingredients_sheets


def _create_minimal_png(path: Path) -> Path:
    """Create a minimal valid PNG file."""
    img = Image.new("RGB", (64, 64), "white")
    img.save(path)
    return path


class TestIngredientsEnrichment(unittest.TestCase):
    def test_enriches_song_scenes_with_ingredients_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            render_plan = [
                {
                    "scene": 1,
                    "abs_start_seconds": 0.0,
                    "duration_seconds": 4.0,
                    "fps": 24,
                    "frame_count": 96,
                    "width": 1280,
                    "height": 704,
                    "ltx": {
                        "i2v_prompt_from_t2i": "cinematic shot of artist singing",
                    },
                    "references": {
                        "actor_ids": ["artist_1"],
                        "location_id": "stage",
                        "actor_reference_descriptions": [{"id": "artist_1", "name": "Artist", "role": "singer", "visual_description": "young man with dark hair"}],
                        "location_reference_description": {"id": "stage", "name": "Stage", "visual_description": "dark stage with spotlights"},
                        "actor_sheet_paths": ["output/references/actors/artist_1/sheet.png"],
                        "actor_msr_paths": ["output/references/actors/artist_1/msr_sheet.png"],
                        "location_sheet_path": "output/references/locations/stage/sheet.png",
                    },
                },
            ]
            rp_path = tmp / "render_plan.json"
            rp_path.write_text(json.dumps(render_plan), encoding="utf-8")

            ref_dir = tmp / "output" / "references"
            actors_dir = ref_dir / "actors" / "artist_1"
            actors_dir.mkdir(parents=True, exist_ok=True)
            (actors_dir / "manifest.json").write_text(
                json.dumps({"id": "artist_1", "name": "Artist", "role": "singer", "visual_description": "young man with dark hair", "sheet_path": str(actors_dir / "sheet.png"), "msr_input_path": str(actors_dir / "msr_sheet.png")}),
                encoding="utf-8",
            )
            _create_minimal_png(actors_dir / "sheet.png")
            _create_minimal_png(actors_dir / "msr_sheet.png")

            locations_dir = ref_dir / "locations" / "stage"
            locations_dir.mkdir(parents=True, exist_ok=True)
            (locations_dir / "manifest.json").write_text(
                json.dumps({"id": "stage", "name": "Stage", "visual_description": "dark stage with spotlights", "sheet_path": str(locations_dir / "sheet.png"), "msr_background_path": str(locations_dir / "msr_background.png")}),
                encoding="utf-8",
            )
            _create_minimal_png(locations_dir / "sheet.png")
            _create_minimal_png(locations_dir / "msr_background.png")

            out_path = tmp / "render_plan_ingredients.json"
            result = enrich_render_plan_with_ingredients_sheets(
                render_plan_path=rp_path,
                references_dir=ref_dir,
                output_path=out_path,
            )

            self.assertTrue(result.exists())
            data = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 1)
            scene = data[0]
            self.assertIn("ingredients_scene_sheet", scene)
            self.assertIn("ingredients_scene_sheet_description", scene)
            self.assertIn("ingredients_target_prompt", scene)
            self.assertIn("ltx", scene)
            self.assertIn("ingredients_scene_sheet_description", scene["ltx"])
            self.assertIn("ingredients_target_prompt", scene["ltx"])
            self.assertIn("cinematic shot of artist singing", scene["ingredients_target_prompt"])
            self.assertIn("Use Character `artist_1` from Left", scene["ingredients_target_prompt"])
            self.assertIn("Use Setting `stage` from Right", scene["ingredients_target_prompt"])
            self.assertIn("Do not add or omit visible characters", scene["ingredients_target_prompt"])

    def test_calls_on_scene_complete_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            render_plan = [
                {
                    "scene": 1,
                    "width": 1280,
                    "height": 704,
                    "ltx": {},
                    "references": {"actor_ids": [], "location_id": ""},
                },
                {
                    "scene": 2,
                    "width": 1280,
                    "height": 704,
                    "ltx": {},
                    "references": {"actor_ids": [], "location_id": ""},
                },
            ]
            rp_path = tmp / "render_plan.json"
            rp_path.write_text(json.dumps(render_plan), encoding="utf-8")

            ref_dir = tmp / "output" / "references"
            ref_dir.mkdir(parents=True, exist_ok=True)

            callbacks = []
            out_path = tmp / "render_plan_ingredients.json"
            result = enrich_render_plan_with_ingredients_sheets(
                render_plan_path=rp_path,
                references_dir=ref_dir,
                output_path=out_path,
                on_scene_complete=lambda scene_num, idx, total: callbacks.append((scene_num, idx, total)),
            )

            self.assertTrue(result.exists())
            self.assertEqual(len(callbacks), 2)
            self.assertEqual(callbacks[0], (1, 1, 2))
            self.assertEqual(callbacks[1], (2, 2, 2))

    def test_empty_references_produces_empty_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            render_plan = [
                {
                    "scene": 1,
                    "width": 1280,
                    "height": 704,
                    "ltx": {},
                    "references": {"actor_ids": [], "location_id": ""},
                },
            ]
            rp_path = tmp / "render_plan.json"
            rp_path.write_text(json.dumps(render_plan), encoding="utf-8")

            ref_dir = tmp / "output" / "references"
            ref_dir.mkdir(parents=True, exist_ok=True)

            out_path = tmp / "render_plan_ingredients.json"
            result = enrich_render_plan_with_ingredients_sheets(
                render_plan_path=rp_path,
                references_dir=ref_dir,
                output_path=out_path,
            )

            self.assertTrue(result.exists())
            data = json.loads(result.read_text(encoding="utf-8"))
            scene = data[0]
            self.assertEqual(scene["ingredients_scene_sheet"], "")
            self.assertEqual(scene["ingredients_scene_sheet_description"], "")
            self.assertEqual(scene["ingredients_target_prompt"], "")
            self.assertTrue(scene["ltx"].get("native_audio"))
