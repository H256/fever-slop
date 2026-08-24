from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.canonical_render_plan import (
    CANONICAL_SCHEMA,
    PromptRole,
    build_canonical_scene,
    resolve_effective_role,
    stable_scene_id,
    validate_canonical_plan,
)
from feverslop.domain.render_plan import RenderScene
from feverslop.errors import FeverSlopDataError
from feverslop.pipeline.render_plan_builder import build_render_plan


class CanonicalRoleResolutionTests(unittest.TestCase):
    def test_generated_value_is_effective_without_override(self):
        scene = {
            "canonical": build_canonical_scene(
                segment_id="segment-a",
                generated_roles={PromptRole.H3_VIDEO: "generated prompt"},
            ),
        }

        self.assertEqual(
            "generated prompt",
            resolve_effective_role(scene, PromptRole.H3_VIDEO),
        )

    def test_human_override_wins_without_changing_generated_value(self):
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.H3_VIDEO: "generated prompt"},
        )
        canonical["roles"][PromptRole.H3_VIDEO]["override"] = {
            "value": "human prompt",
            "provenance": {"source": "human"},
        }
        scene = {"canonical": canonical}

        self.assertEqual("human prompt", resolve_effective_role(scene, PromptRole.H3_VIDEO))
        self.assertEqual(
            "generated prompt",
            canonical["roles"][PromptRole.H3_VIDEO]["generated"]["value"],
        )

    def test_empty_override_is_an_actionable_validation_error(self):
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.H3_VIDEO: "generated prompt"},
        )
        canonical["roles"][PromptRole.H3_VIDEO]["override"] = {"value": "  "}

        with self.assertRaisesRegex(
            FeverSlopDataError,
            r"canonical\.roles\.h3\.video\.override\.value.*must not be empty",
        ):
            resolve_effective_role({"canonical": canonical}, PromptRole.H3_VIDEO)

    def test_explicit_legacy_value_is_used_for_legacy_scene(self):
        self.assertEqual(
            "legacy prompt",
            resolve_effective_role({}, PromptRole.H3_VIDEO, legacy_value="legacy prompt"),
        )

    def test_missing_role_without_legacy_value_is_an_error(self):
        scene = {
            "canonical": build_canonical_scene(
                segment_id="segment-a",
                generated_roles={},
            ),
        }

        with self.assertRaisesRegex(FeverSlopDataError, r"h3\.video.*no effective value"):
            resolve_effective_role(scene, PromptRole.H3_VIDEO)

    def test_effective_value_must_not_be_persisted(self):
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.H3_VIDEO: "generated prompt"},
        )
        canonical["roles"][PromptRole.H3_VIDEO]["effective"] = "stale prompt"

        with self.assertRaisesRegex(FeverSlopDataError, r"effective.*must not be persisted"):
            resolve_effective_role({"canonical": canonical}, PromptRole.H3_VIDEO)

    def test_malformed_role_reports_its_json_path(self):
        scene = {
            "canonical": {
                "schema": CANONICAL_SCHEMA,
                "scene_id": stable_scene_id("segment-a"),
                "segment_id": "segment-a",
                "roles": {PromptRole.H3_VIDEO: "not-an-object"},
            },
        }

        with self.assertRaisesRegex(
            FeverSlopDataError,
            r"canonical\.roles\.h3\.video must be an object",
        ):
            resolve_effective_role(scene, PromptRole.H3_VIDEO)

    def test_render_scene_exposes_the_shared_resolver_without_rewriting_legacy_accessors(self):
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.Z_IMAGE: "generated still"},
        )
        canonical["roles"][PromptRole.Z_IMAGE]["override"] = {"value": "human still"}
        scene = RenderScene.from_dict({
            "scene": 1,
            "z_image": {"prompt": "legacy still"},
            "canonical": canonical,
        })

        self.assertEqual(
            "human still",
            scene.effective_role(PromptRole.Z_IMAGE, legacy_value=scene.z_image_prompt),
        )
        self.assertEqual("legacy still", scene.z_image_prompt)


class CanonicalSceneIdentityTests(unittest.TestCase):
    def test_scene_id_is_deterministic_and_independent_of_array_order(self):
        first = build_canonical_scene(segment_id="segment-a", generated_roles={})
        second = build_canonical_scene(segment_id="segment-b", generated_roles={})

        reordered = [second, first]

        self.assertEqual(stable_scene_id("segment-a"), reordered[1]["scene_id"])
        self.assertNotEqual(first["scene_id"], second["scene_id"])

    def test_duplicate_scene_identity_is_rejected(self):
        canonical = build_canonical_scene(segment_id="segment-a", generated_roles={})
        scenes = [
            {"scene": 1, "canonical": canonical},
            {"scene": 2, "canonical": dict(canonical)},
        ]

        with self.assertRaisesRegex(FeverSlopDataError, r"duplicate canonical scene_id"):
            validate_canonical_plan(scenes)


class CanonicalRenderPlanBuilderTests(unittest.TestCase):
    def test_builder_adds_canonical_roles_without_removing_legacy_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenes = [{
                "scene": 1,
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "zimage_prompt": "generated still",
                "t2i_prompt": "generated base",
                "i2v_prompt_from_t2i": "generated video",
                "segment_id": "segment-a",
                "type": "vocals",
            }]
            relays = [{
                "scene": 1,
                "prompt_relay": [{
                    "frame_start": 0,
                    "frame_end": 48,
                    "state": "singing",
                }],
            }]
            h3 = [{
                "segment_id": "segment-a",
                "prompt": "generated h3",
                "performance_timing": [{"start": 0, "end": 48}],
            }]
            for name, value in (("scenes.json", scenes), ("relays.json", relays), ("h3.json", h3)):
                (temp / name).write_text(json.dumps(value), encoding="utf-8")

            output = build_render_plan(
                temp / "scenes.json",
                temp / "relays.json",
                temp / "base.json",
                VideoSettings(width=1280, height=720, fps=24),
                artifact_store=JsonArtifactStore(),
                h3_prompts_json=temp / "h3.json",
            )

            scene = json.loads(output.read_text(encoding="utf-8"))[0]
            self.assertEqual("generated still", scene["z_image"]["prompt"])
            self.assertEqual("generated h3", scene["h3"]["prompt"])
            self.assertEqual(CANONICAL_SCHEMA, scene["canonical"]["schema"])
            self.assertEqual("segment-a", scene["canonical"]["segment_id"])
            self.assertEqual(stable_scene_id("segment-a"), scene["canonical"]["scene_id"])
            self.assertEqual(
                "generated still",
                scene["canonical"]["roles"][PromptRole.Z_IMAGE]["generated"]["value"],
            )
            self.assertEqual(
                "generated h3",
                scene["canonical"]["roles"][PromptRole.H3_VIDEO]["generated"]["value"],
            )
            self.assertNotIn("effective", scene["canonical"]["roles"][PromptRole.H3_VIDEO])


if __name__ == "__main__":
    unittest.main()
