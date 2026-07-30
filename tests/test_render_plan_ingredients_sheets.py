import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from feverslop.adapters.project_visual_consistency import ProjectReferenceManifestAdapter
from feverslop.application.render_plan_ingredients_sheets import enrich_render_plan_with_ingredients_sheets
from feverslop.application.visual_consistency_preflight import preflight_visual_consistency
from feverslop.config.video_settings import VideoSettings
from feverslop.errors import FeverSlopValidationError


def _create_minimal_png(path: Path) -> Path:
    """Create a minimal valid PNG file."""
    img = Image.new("RGB", (64, 64), "white")
    img.save(path)
    return path


class TestIngredientsEnrichment(unittest.TestCase):
    def test_equal_references_reuse_sheet_without_collapsing_song_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            references = project / "output" / "references"
            actor_dir = references / "actors" / "artist"
            location_dir = references / "locations" / "stage"
            actor_dir.mkdir(parents=True)
            location_dir.mkdir(parents=True)
            actor = _create_minimal_png(actor_dir / "sheet.png")
            location = _create_minimal_png(location_dir / "sheet.png")
            (actor_dir / "manifest.json").write_text(json.dumps({
                "id": "artist",
                "name": "Artist",
                "visual_description": "Artist wears a red jacket",
                "sheet_path": actor.relative_to(project).as_posix(),
            }), encoding="utf-8")
            (location_dir / "manifest.json").write_text(json.dumps({
                "id": "stage",
                "name": "Stage",
                "visual_description": "Stage has blue spotlights",
                "sheet_path": location.relative_to(project).as_posix(),
            }), encoding="utf-8")
            scenes = []
            for number, action, camera, lyric in (
                (1, "raises one hand", "slow dolly", "First line"),
                (2, "lowers both hands", "locked wide shot", "Second line"),
            ):
                scenes.append({
                    "scene": number,
                    "duration_seconds": 2,
                    "fps": 24,
                    "frame_count": 48,
                    "width": 1280,
                    "height": 704,
                    "metadata": {
                        "character_motion": action,
                        "camera_motion": camera,
                        "lyrics": lyric,
                    },
                    "references": {
                        "actor_ids": ["artist"],
                        "location_id": "stage",
                    },
                    "ltx": {"prompt_relay": [{
                        "frame_start": 0,
                        "frame_end": 47,
                        "state": "singing",
                        "prompt": f"{action}; {camera}; sings {lyric}.",
                    }]},
                })
            plan = project / "render_plan.json"
            plan.write_text(json.dumps(scenes), encoding="utf-8")

            output = enrich_render_plan_with_ingredients_sheets(
                plan,
                references,
                project / "ingredients.json",
            )
            runtime = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(
                runtime[0]["ingredients"]["sheet_path"],
                runtime[1]["ingredients"]["sheet_path"],
            )
            self.assertIn("raises one hand", runtime[0]["ltx"]["static_prompt"])
            self.assertIn("lowers both hands", runtime[1]["ltx"]["static_prompt"])
            self.assertIn("First line", runtime[0]["metadata"]["lyrics"])
            self.assertIn("Second line", runtime[1]["metadata"]["lyrics"])
            self.assertNotEqual(
                runtime[0]["ltx"]["prompt_relay"],
                runtime[1]["ltx"]["prompt_relay"],
            )

    def test_video_settings_override_render_plan_sheet_resolution(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "feverslop.application.render_plan_ingredients_sheets.ingredients_sheet_size",
            return_value=(2048, 1152),
        ) as sheet_size, patch(
            "feverslop.application.render_plan_ingredients_sheets.project_ingredients_runtime_scene",
            side_effect=lambda scene: scene,
        ):
            tmp = Path(tmp)
            render_plan = tmp / "render_plan.json"
            render_plan.write_text(
                json.dumps([{"scene": 1, "width": 1280, "height": 704, "references": {}}]),
                encoding="utf-8",
            )
            references = tmp / "references"
            references.mkdir()

            enrich_render_plan_with_ingredients_sheets(
                render_plan,
                references,
                tmp / "enriched.json",
                video_settings=VideoSettings(width=1024, height=576),
            )

            sheet_size.assert_called_once_with(1024, 576, 2.0)

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
                        "msr_prompt_relay": [
                            {
                                "frame_start": 0,
                                "frame_end": 95,
                                "state": "singing",
                                "prompt": "Artist sings immediately with precise lip sync.",
                            }
                        ],
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
            self.assertRegex(
                scene["ingredients"]["sheet_path"],
                r"^output/references/ingredients_sheets/by_signature/[0-9a-f]{64}\.png$",
            )
            self.assertIn("artist_1", scene["ingredients"]["global_prompt"])
            self.assertEqual("singing", scene["ltx"]["prompt_relay"][0]["state"])
            self.assertIn("sings immediately", scene["ltx"]["static_prompt"])
            self.assertEqual(
                "feverslop.visual-consistency/v1",
                scene["visual_consistency"]["schema"],
            )
            self.assertEqual(
                scene["ingredients"]["signature"],
                Path(scene["ingredients"]["sheet_path"]).stem,
            )
            self.assertRegex(
                scene["ingredients"]["sheet_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                {
                    "actors": [{
                        "id": "artist_1",
                        "path": "output/references/actors/artist_1/msr_sheet.png",
                    }],
                    "location": {
                        "id": "stage",
                        "path": (
                            "output/references/locations/stage/"
                            "msr_background.png"
                        ),
                    },
                },
                scene["visual_consistency_sources"],
            )
            self.assertEqual(
                1,
                scene["ingredients"]["global_prompt"].count(
                    "Continuity anchors (keep unchanged):"
                ),
            )
            self.assertNotIn("ingredients_scene_sheet_description", scene)
            self.assertNotIn("ingredients_target_prompt", scene)
            self.assertNotIn("msr_prompt_relay", scene["ltx"])
            self.assertNotIn("actor_reference_descriptions", scene["references"])
            snapshot = ProjectReferenceManifestAdapter(lambda _project_id: tmp).load(
                tmp.name
            )
            preflight = preflight_visual_consistency(
                data,
                snapshot,
                mode="ingredients",
                workflow_profile="ingredients-default",
                preflight_mode="strict",
                supports_continuous_transitions=False,
            )
            self.assertNotIn(
                "visual_consistency_fingerprint_mismatch",
                [issue.code for issue in preflight.issues],
            )

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
            with patch(
                "feverslop.application.render_plan_ingredients_sheets.project_ingredients_runtime_scene",
                side_effect=lambda scene: scene,
            ):
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

    def test_empty_references_reject_unrenderable_runtime_scene(self):
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
            with self.assertRaisesRegex(FeverSlopValidationError, "Scene 1.*global prompt"):
                enrich_render_plan_with_ingredients_sheets(
                    render_plan_path=rp_path,
                    references_dir=ref_dir,
                    output_path=out_path,
                )
