from __future__ import annotations

from copy import deepcopy
import unittest

from feverslop.application.ingredients_render_plan import project_ingredients_runtime_scene
from feverslop.domain.visual_consistency import ReferenceAnchor, SceneConsistencyContract
from feverslop.domain.visual_consistency_runtime import scrub_prior_context
from feverslop.errors import FeverSlopValidationError


class IngredientsRenderPlanProjectionTests(unittest.TestCase):
    def test_scrub_removes_only_structural_continuity_directives(self):
        self.assertEqual(
            "Silk enters the tunnel.",
            scrub_prior_context(
                "Continue with same wardrobe from before; Silk enters the tunnel."
            ),
        )
        self.assertEqual(
            "",
            scrub_prior_context(
                "Continue with the same wardrobe from before."
            ),
        )
        self.assertEqual(
            "Silk enters the tunnel.",
            scrub_prior_context(
                "As before, Silk enters the tunnel."
            ),
        )

    def test_scrub_preserves_previous_scene_as_subject_or_comparison(self):
        for prompt in (
            "The critic reviews previous scene compositions.",
            "Unlike previous scene lighting, this room is bright.",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(prompt, scrub_prior_context(prompt))

    @staticmethod
    def _bloated_scene() -> dict:
        return {
            "scene": 6,
            "abs_start_seconds": 42.0,
            "abs_end_seconds": 52.7,
            "duration_seconds": 10.7,
            "fps": 24,
            "width": 1280,
            "height": 704,
            "frame_count": 257,
            "cut": {"transition": "hard"},
            "concept": "unused authoring concept",
            "z_image": {"prompt": "unused storyboard prompt"},
            "metadata": {
                "segment_id": "segment_006",
                "type": "vocals",
                "silent_mode": False,
                "lyrics": "Silver and iron",
                "base_concept": "duplicate concept",
                "camera_motion": "tracking",
            },
            "references": {
                "actor_ids": ["silk", "bobby"],
                "location_id": "mountain_tunnels",
                "actor_msr_paths": ["silk.png", "bobby.png"],
                "actor_reference_descriptions": [{"id": "silk", "role": "lead singer"}],
                "location_msr_path": "tunnel.png",
            },
            "ingredients_scene_sheet": "output/references/ingredients_sheets/scene_0006_ingredients.png",
            "ingredients_scene_sheet_anchors": [{"id": "silk", "position": "Left"}],
            "ingredients_scene_sheet_description": "Silk and Bobby in the mountain tunnels.",
            "ingredients_target_prompt": "A long chronological prompt that should be removed.",
            "ltx": {
                "base_prompt": "old base prompt",
                "i2v_prompt_from_t2i": "old i2v prompt",
                "prompt_relay": [
                    {"frame_start": 0, "frame_end": 33, "state": "singing", "prompt": "raw singing"},
                ],
                "msr_global_prompt": "duplicate MSR global prompt",
                "msr_prompt_relay": [
                    {
                        "frame_start": 0,
                        "frame_end": 33,
                        "state": "singing",
                        "prompt": "Silk sings immediately with precise lip sync.",
                    },
                    {
                        "frame_start": 33,
                        "frame_end": 57,
                        "state": "instrumental",
                        "prompt": "Silk and Bobby remain silent with closed mouths.",
                    },
                ],
                "ingredients_scene_sheet_description": "duplicate description",
                "ingredients_target_prompt": "duplicate target",
            },
        }

    def test_projection_keeps_only_ingredients_runtime_contract(self):
        source = self._bloated_scene()
        original = deepcopy(source)

        projected = project_ingredients_runtime_scene(source)

        self.assertEqual(
            {
                "scene",
                "abs_start_seconds",
                "abs_end_seconds",
                "duration_seconds",
                "fps",
                "width",
                "height",
                "frame_count",
                "cut",
                "metadata",
                "references",
                "ingredients",
                "ltx",
            },
            set(projected),
        )
        self.assertEqual(
            {
                "segment_id": "segment_006",
                "type": "vocals",
                "silent_mode": False,
                "lyrics": "Silver and iron",
            },
            projected["metadata"],
        )
        self.assertEqual(
            {"actor_ids": ["silk", "bobby"], "location_id": "mountain_tunnels"},
            projected["references"],
        )
        self.assertEqual(
            {
                "sheet_path": "output/references/ingredients_sheets/scene_0006_ingredients.png",
                "anchors": [{"id": "silk", "position": "Left"}],
                "global_prompt": "Silk and Bobby in the mountain tunnels.",
            },
            projected["ingredients"],
        )
        self.assertEqual(source["ltx"]["msr_prompt_relay"], projected["ltx"]["prompt_relay"])
        self.assertEqual(projected["ingredients"]["global_prompt"], projected["ltx"]["base_prompt"])
        self.assertIn("Silk sings immediately", projected["ltx"]["static_prompt"])
        self.assertIn("then instrumental", projected["ltx"]["static_prompt"].lower())
        self.assertTrue(projected["ltx"]["native_audio"])
        self.assertEqual(original, source)

    def test_projection_falls_back_to_raw_prompt_relay(self):
        scene = self._bloated_scene()
        del scene["ltx"]["msr_prompt_relay"]

        projected = project_ingredients_runtime_scene(scene)

        self.assertEqual(scene["ltx"]["prompt_relay"], projected["ltx"]["prompt_relay"])

    def test_projection_static_prompt_enforces_whole_scene_silence(self):
        scene = self._bloated_scene()
        scene["ltx"]["msr_prompt_relay"] = [
            {
                "frame_start": 0,
                "frame_end": 257,
                "state": "instrumental",
                "prompt": "The performers walk through the tunnel.",
            }
        ]

        projected = project_ingredients_runtime_scene(scene)

        self.assertIn("no vocal performance throughout", projected["ltx"]["static_prompt"].lower())
        self.assertIn("mouths remain closed", projected["ltx"]["static_prompt"].lower())

    def test_projection_binds_one_stable_anchor_block_and_contract(self):
        scene = self._bloated_scene()
        actor = ReferenceAnchor(
            id="silk",
            kind="actor",
            look_id="default",
            asset_role="identity-reference",
            asset_sha256="a" * 64,
            prompt_anchor=(
                "As before after the prior shot, Silk has a sharp black bob "
                "and a silver jacket."
            ),
        )
        location = ReferenceAnchor(
            id="mountain_tunnels",
            kind="location",
            look_id="default",
            asset_role="environment-reference",
            asset_sha256="b" * 64,
            prompt_anchor=(
                "Previous scene aside, the mountain tunnel has wet basalt walls "
                "and amber lamps."
            ),
        )
        contract = SceneConsistencyContract.create(
            scene=6,
            mode="ingredients",
            workflow_profile="ingredients-final",
            actors=(actor,),
            location=location,
            transition_from_previous="cut",
        )
        scene["visual_consistency"] = contract.to_dict()
        scene["ingredients_sheet_signature"] = "c" * 64
        scene["ingredients_sheet_layout_version"] = "scene-reference-grid/v1"
        scene["ingredients_sheet_size"] = [1280, 704]
        scene["ingredients_signature_references"] = [
            {"id": "silk", "type": "actor", "sha256": "a" * 64},
            {"id": "mountain_tunnels", "type": "location", "sha256": "b" * 64},
        ]
        scene["ingredients_scene_sheet_description"] = (
            "As before after the prior shot and previous scene; Silk enters the tunnel."
        )
        scene["ltx"]["msr_prompt_relay"][0]["prompt"] = (
            "Continue the action from the previous shot."
        )

        projected = project_ingredients_runtime_scene(scene)

        self.assertEqual(contract.to_dict(), projected.get("visual_consistency"))
        self.assertEqual("c" * 64, projected["ingredients"].get("signature"))
        for prompt in (
            projected["ingredients"]["global_prompt"],
            projected["ltx"]["base_prompt"],
            projected["ltx"]["static_prompt"],
        ):
            self.assertEqual(
                1,
                prompt.count("Continuity anchors (keep unchanged):"),
            )
            self.assertIn("sharp black bob", prompt)
            self.assertIn("wet basalt walls", prompt)
            self.assertNotIn("same as before", prompt.lower())
            self.assertNotIn("previous scene", prompt.lower())
            self.assertNotIn("prior shot", prompt.lower())
            self.assertNotIn("as before", prompt.lower())
        self.assertIn(
            "Silk and Bobby remain silent",
            projected["ltx"]["static_prompt"],
        )
        self.assertIn(
            "previous shot",
            projected["ltx"]["prompt_relay"][0]["prompt"].lower(),
        )

    def test_projection_rejects_missing_global_prompt(self):
        scene = self._bloated_scene()
        scene["ingredients_scene_sheet_description"] = ""
        scene["ltx"]["ingredients_scene_sheet_description"] = ""

        with self.assertRaisesRegex(FeverSlopValidationError, "Scene 6.*global prompt"):
            project_ingredients_runtime_scene(scene)

    def test_projection_rejects_empty_relay(self):
        scene = self._bloated_scene()
        scene["ltx"]["msr_prompt_relay"] = []
        scene["ltx"]["prompt_relay"] = []

        with self.assertRaisesRegex(FeverSlopValidationError, "Scene 6.*prompt relay"):
            project_ingredients_runtime_scene(scene)


if __name__ == "__main__":
    unittest.main()
