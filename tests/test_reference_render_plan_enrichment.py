import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from feverslop.application.reference_bible import (
    enrich_render_plan_with_reference_sheets,
)
from feverslop.application.render_plan_ingredients_sheets import (
    enrich_render_plan_with_ingredients_sheets,
)
from feverslop.errors import FeverSlopValidationError
from feverslop.prompting.ingredients_signatures import IngredientsVisionResult


class ReferenceRenderPlanEnrichmentTests(unittest.TestCase):
    def test_reports_all_missing_reference_manifests_with_stage_hint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan_path = temp / "render_plan.json"
            render_plan_path.write_text(json.dumps([
                {"scene": 1, "references": {"actor_ids": ["warrior_leader", "mage_companion"], "location_id": "crypt"}},
                {"scene": 2, "references": {"actor_ids": ["warrior_leader"], "location_id": "nave"}},
            ]), encoding="utf-8")

            with self.assertRaises(FeverSlopValidationError) as raised:
                enrich_render_plan_with_reference_sheets(render_plan_path, temp, temp / "enriched.json")

            message = str(raised.exception)
            self.assertIn("mage_companion", message)
            self.assertIn("warrior_leader", message)
            self.assertIn("crypt", message)
            self.assertIn("nave", message)
            self.assertIn("--stage msr_references", message)

    def test_enriches_reference_ids_with_sheet_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            temp = project / "output" / "references"
            actor_dir = temp / "actors" / "singer"
            location_dir = temp / "locations" / "stage"
            actor_dir.mkdir(parents=True)
            location_dir.mkdir(parents=True)
            actor_sheet = actor_dir / "sheet.png"
            location_sheet = location_dir / "sheet.png"
            actor_msr = actor_dir / "views" / "hero.png"
            location_msr = location_dir / "views" / "hero.png"
            actor_msr.parent.mkdir(parents=True)
            location_msr.parent.mkdir(parents=True)
            actor_sheet.write_bytes(b"actor")
            location_sheet.write_bytes(b"location")
            actor_msr.write_bytes(b"actor-msr")
            location_msr.write_bytes(b"location-msr")
            (actor_dir / "manifest.json").write_text(
                json.dumps({
                    "id": "singer",
                    "name": "Mara",
                    "role": "lead singer",
                    "visual_description": "silver-haired singer in a red coat",
                    "image_prompt": "character sheet of Mara",
                    "kind": "actor",
                    "sheet_path": "output/references/actors/singer/sheet.png",
                    "msr_input_path": "output/references/actors/singer/views/hero.png",
                }),
                encoding="utf-8",
            )
            (location_dir / "manifest.json").write_text(
                json.dumps({
                    "id": "stage",
                    "name": "Mirror Stage",
                    "visual_description": "black mirror stage with neon rain",
                    "image_prompt": "wide mirror stage",
                    "kind": "location",
                    "sheet_path": "output/references/locations/stage/sheet.png",
                    "msr_background_path": "output/references/locations/stage/views/hero.png",
                }),
                encoding="utf-8",
            )
            render_plan_path = project / "output" / "render" / "render_plan.json"
            render_plan_path.parent.mkdir(parents=True)
            render_plan_path.write_text(
                json.dumps([
                    {"scene": 1, "references": {"actor_ids": ["singer"], "location_id": "stage"}},
                ]),
                encoding="utf-8",
            )

            output_path = enrich_render_plan_with_reference_sheets(render_plan_path, temp, temp / "render_plan_refs.json")

            enriched = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(["output/references/actors/singer/sheet.png"], enriched[0]["references"]["actor_sheet_paths"])
            self.assertEqual("output/references/locations/stage/sheet.png", enriched[0]["references"]["location_sheet_path"])
            self.assertEqual(["output/references/actors/singer/views/hero.png"], enriched[0]["references"]["actor_msr_paths"])
            self.assertEqual("output/references/locations/stage/views/hero.png", enriched[0]["references"]["location_msr_path"])
            self.assertEqual("Mara", enriched[0]["references"]["actor_reference_descriptions"][0]["name"])
            self.assertEqual("lead singer", enriched[0]["references"]["actor_reference_descriptions"][0]["role"])
            self.assertEqual("Mirror Stage", enriched[0]["references"]["location_reference_description"]["name"])
            self.assertEqual(
                {
                    "actors": [
                        {
                            "id": "singer",
                            "path": "output/references/actors/singer/views/hero.png",
                        },
                    ],
                    "location": {
                        "id": "stage",
                        "path": "output/references/locations/stage/views/hero.png",
                    },
                },
                enriched[0].get("visual_consistency_sources"),
            )

    def test_fails_clearly_when_actor_manifest_is_malformed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_dir = temp / "actors" / "singer"
            actor_dir.mkdir(parents=True)
            (actor_dir / "manifest.json").write_text("{not json", encoding="utf-8")
            render_plan_path = temp / "render_plan.json"
            render_plan_path.write_text(
                json.dumps([{"scene": 1, "references": {"actor_ids": ["singer"]}}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FeverSlopValidationError, "actors/singer/manifest.json"):
                enrich_render_plan_with_reference_sheets(render_plan_path, temp, temp / "enriched.json")

    def test_fails_clearly_when_manifest_is_missing_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_dir = temp / "actors" / "singer"
            actor_dir.mkdir(parents=True)
            (actor_dir / "manifest.json").write_text(json.dumps({"name": "Mara"}), encoding="utf-8")
            render_plan_path = temp / "render_plan.json"
            render_plan_path.write_text(
                json.dumps([{"scene": 1, "references": {"actor_ids": ["singer"]}}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FeverSlopValidationError, "missing an id"):
                enrich_render_plan_with_reference_sheets(render_plan_path, temp, temp / "enriched.json")

    def test_fails_clearly_when_render_plan_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            with self.assertRaisesRegex(FeverSlopValidationError, "Cannot read render plan"):
                enrich_render_plan_with_reference_sheets(temp / "render_plan.json", temp, temp / "enriched.json")

    def test_fails_clearly_when_render_plan_is_not_a_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan_path = temp / "render_plan.json"
            render_plan_path.write_text(json.dumps({"scene": 1}), encoding="utf-8")

            with self.assertRaisesRegex(FeverSlopValidationError, "JSON array"):
                enrich_render_plan_with_reference_sheets(render_plan_path, temp, temp / "enriched.json")

    def test_reports_progress_per_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_dir = temp / "actors" / "singer"
            location_dir = temp / "locations" / "stage"
            actor_dir.mkdir(parents=True)
            location_dir.mkdir(parents=True)
            actor_sheet = actor_dir / "sheet.png"
            location_sheet = location_dir / "sheet.png"
            actor_sheet.write_bytes(b"actor")
            location_sheet.write_bytes(b"location")
            (actor_dir / "manifest.json").write_text(
                json.dumps({
                    "id": "singer",
                    "name": "Mara",
                    "kind": "actor",
                    "sheet_path": str(actor_sheet),
                }),
                encoding="utf-8",
            )
            (location_dir / "manifest.json").write_text(
                json.dumps({
                    "id": "stage",
                    "name": "Mirror Stage",
                    "kind": "location",
                    "sheet_path": str(location_sheet),
                }),
                encoding="utf-8",
            )
            render_plan_path = temp / "render_plan.json"
            render_plan_path.write_text(
                json.dumps([
                    {"scene": 1, "references": {"actor_ids": ["singer"], "location_id": "stage"}},
                    {"scene": 2, "references": {"actor_ids": ["singer"], "location_id": "stage"}},
                ]),
                encoding="utf-8",
            )
            events = []

            enrich_render_plan_with_reference_sheets(
                render_plan_path,
                temp,
                temp / "render_plan_refs.json",
                on_scene_complete=lambda scene, completed, total: events.append((scene, completed, total)),
            )

            self.assertEqual([(1, 1, 2), (2, 2, 2)], events)


class IngredientsVisionEnrichmentTests(unittest.TestCase):
    def test_analyzes_individual_references_and_stores_detailed_prompts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            references = project / "output" / "references"
            actor_dir = references / "actors" / "singer"
            location_dir = references / "locations" / "stage"
            actor_dir.mkdir(parents=True)
            location_dir.mkdir(parents=True)
            actor_sheet = actor_dir / "sheet.png"
            location_sheet = location_dir / "sheet.png"
            Image.new("RGB", (8, 8), "red").save(actor_sheet)
            Image.new("RGB", (8, 8), "blue").save(location_sheet)
            for root, data in (
                (actor_dir, {"id": "singer", "name": "Mara", "kind": "actor", "visual_description": "red coat", "image_prompt": "silver-haired singer", "sheet_path": "output/references/actors/singer/sheet.png"}),
                (location_dir, {"id": "stage", "name": "Mirror Stage", "kind": "location", "visual_description": "neon rain", "image_prompt": "black mirror stage", "sheet_path": "output/references/locations/stage/sheet.png"}),
            ):
                (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
            plan = project / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 7,
                "concept": "defiant chorus",
                "duration_seconds": 5,
                "camera_motion": "slow orbit",
                "character_motion": "Mara raises one hand",
                "references": {"actor_ids": ["singer"], "location_id": "stage"},
                "ltx": {
                    "i2v_prompt_from_t2i": "Mara sings through the rain",
                    "msr_prompt_relay": [{
                        "frame_start": 0, "frame_end": 119, "state": "singing",
                        "prompt": "Mara sings immediately with precise lip sync.",
                    }],
                },
            }]), encoding="utf-8")

            class FakeIngredientsModule:
                def __init__(self, _llm, **_kwargs):
                    self.image_paths = []
                    self.user_prompt = ""

                def vision(self, payload, paths):
                    self.image_paths = paths
                    self.user_prompt = json.dumps(payload)
                    return IngredientsVisionResult(
                        references=[
                            {"id": "singer", "type": "actor", "t2i_description": "silver hair and a vivid red coat"},
                            {"id": "stage", "type": "location", "t2i_description": "black mirrors crossed by blue neon rain"},
                        ],
                        shot_invariants=" ".join(["stable cinematic composition"] * 30),
                    )

            llm = object()
            module = FakeIngredientsModule(llm)
            events = []
            with patch("feverslop.application.ingredients_vision_prompt.IngredientsPromptModules", return_value=module):
                output = enrich_render_plan_with_ingredients_sheets(
                    plan, references, project / "enriched.json", llm=llm,
                    on_analysis_status=lambda scene_id, refs: events.append((scene_id, refs)),
                )
            scene = json.loads(output.read_text(encoding="utf-8"))[0]

            self.assertEqual([actor_sheet, location_sheet], module.image_paths)
            self.assertNotIn("ingredients_sheets", str(module.image_paths[0]))
            self.assertIn("Character `singer`", scene["ingredients"]["global_prompt"])
            self.assertIn("silver hair", scene["ingredients"]["global_prompt"])
            self.assertIn("### Target Description", scene["ingredients"]["global_prompt"])
            self.assertIn("defiant chorus", module.user_prompt)
            self.assertIn("slow orbit", module.user_prompt)
            self.assertEqual([(7, [{"id": "singer", "type": "actor"}, {"id": "stage", "type": "location"}])], events)

            plan.write_text(json.dumps([{
                "scene": 7,
                "references": {"actor_ids": ["singer"], "location_id": "stage"},
                "ltx": {
                    "i2v_prompt_from_t2i": "",
                    "prompt_relay": [{
                        "frame_start": 0, "frame_end": 119, "state": "instrumental",
                        "prompt": "Mara remains silent with her mouth closed.",
                    }],
                },
            }]), encoding="utf-8")
            fallback_output = enrich_render_plan_with_ingredients_sheets(
                plan, references, project / "fallback.json", llm=None,
            )
            fallback_scene = json.loads(fallback_output.read_text(encoding="utf-8"))[0]
            self.assertIn("### Target Description", fallback_scene["ingredients"]["global_prompt"])
            self.assertIn("no vocal performance throughout", fallback_scene["ltx"]["static_prompt"].lower())
