import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.scene_prompt_builder import ScenePromptBuilder, scene_prompt_word_limit
from tests.prompt_fakes import GeneralModulesFake


class ScenePromptBuilderTests(unittest.TestCase):
    def test_scene_prompt_overflow_is_trimmed_with_scene_aware_diagnostic(self):
        modules = GeneralModulesFake(i2v="word " * 60)
        messages = []
        builder = ScenePromptBuilder(object(), modules=modules)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=[{"segment_id": "segment_011", "scene": 11, "type": "instrumental"}],
                concept_prompts={"segment_011": "A camera crosses the battlefield."},
                scene_details={"segment_011": {}},
                global_context={
                    "subject": "a stone statue",
                    "story_idea": "A battle.",
                    "style": "cinematic",
                    "locations": ["battlefield"],
                    "prompt_guidance": {},
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
                status_callback=messages.append,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(50, len(data[0]["i2v_prompt_from_t2i"].split()))
        self.assertEqual(1, len(messages))
        self.assertIn("Scene 11 I2V prompt", messages[0])
        self.assertIn("60 words", messages[0])
        self.assertIn("trimmed to 50 words", messages[0])

    def test_scene_prompt_word_limit_uses_shared_default_for_missing_or_invalid_guidance(self):
        from feverslop.config.project_config import SCENE_PROMPT_WORD_COUNT_MAX

        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({}))
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({"prompt_guidance": {}}))
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({"prompt_guidance": {"word_count_max": None}}))
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({"prompt_guidance": {"word_count_max": "many"}}))
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({"prompt_guidance": {"word_count_max": 0}}))
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({"prompt_guidance": {"word_count_max": -10}}))
        self.assertEqual(42, scene_prompt_word_limit({"prompt_guidance": {"word_count_max": 42}}))

    def test_scene_prompt_uses_configured_word_count_max(self):
        modules = GeneralModulesFake(i2v="word " * 51)
        messages = []
        builder = ScenePromptBuilder(object(), modules=modules)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=[{"segment_id": "segment_003", "scene": 3, "type": "instrumental"}],
                concept_prompts={"segment_003": "A camera crosses the battlefield."},
                scene_details={"segment_003": {}},
                global_context={
                    "subject": "a stone statue",
                    "story_idea": "A battle.",
                    "style": "cinematic",
                    "locations": ["battlefield"],
                    "prompt_guidance": {"word_count_min": 40, "word_count_max": 50},
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
                status_callback=messages.append,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(50, len(data[0]["i2v_prompt_from_t2i"].split()))
        self.assertIn("51 words", messages[0])
        self.assertIn("trimmed to 50 words", messages[0])

    def test_scene_prompts_report_progress_after_each_scene(self):
        modules = GeneralModulesFake()
        progress = []
        builder = ScenePromptBuilder(object(), modules=modules)

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.build_scene_prompts(
                stage1_segments=[
                    {"segment_id": "segment_001", "type": "vocals"},
                    {"segment_id": "segment_002", "type": "instrumental"},
                ],
                concept_prompts={"segment_001": "one", "segment_002": "two"},
                scene_details={},
                global_context={
                    "subject": "singer",
                    "story_idea": "story",
                    "style": "cinematic",
                    "locations": ["stage"],
                    "prompt_guidance": {},
                },
                output_json_path=Path(temp_dir) / "scene_prompts.json",
                artifact_store=JsonArtifactStore(),
                progress_callback=lambda current, total: progress.append((current, total)),
            )

        self.assertEqual([(1, 2), (2, 2)], progress)

    def test_scene_prompts_include_explicit_t2i_and_i2v_fields(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)
        stage1_segments = [
            {
                "segment_id": "segment_001",
                "type": "vocals",
                "start": 0.0,
                "end": 4.0,
                "duration": 4.0,
                "lyrics": "hello",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=stage1_segments,
                concept_prompts={"segment_001": "A lone singer on a mountain peak."},
                scene_details={"segment_001": {"camera_motion": "slow push-in", "character_motion": "hair in the wind"}},
                global_context={
                    "subject": "a man with long hair",
                    "story_idea": "A mountain performance.",
                    "style": "cinematic realism",
                    "locations": ["mountain peak"],
                    "prompt_guidance": {},
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("T2I RESULT", data[0]["t2i_prompt"])
        self.assertEqual("T2I RESULT", data[0]["zimage_prompt"])
        self.assertEqual("T2I RESULT", data[0]["ltx_base_prompt"])
        self.assertEqual("I2V RESULT", data[0]["i2v_prompt_from_t2i"])
        self.assertEqual("I2V RESULT", data[0]["original_style_i2v_prompt"])
        self.assertGreaterEqual(len(modules.calls), 2)
        self.assertIn("T2I RESULT", modules.calls[-1].payload["t2i_prompt"])

    def test_silent_mode_uses_dialogue_free_i2v_policy_for_vocal_segments(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)

        builder.build_i2v_prompt_from_t2i(
            segment={"segment_id": "segment_001", "type": "vocals", "lyrics": "hello"},
            concept="Mara performs with grief on the mirror stage.",
            scene_details={"character_motion": "clutches her chest"},
            global_context={
                "subject": "Mara",
                "story_idea": "A wordless stage performance.",
                "style": "cinematic realism",
                "locations": ["Mirror Stage"],
                "prompt_guidance": {},
                "silent_mode": True,
            },
            t2i_prompt="Mara stands under a spotlight.",
        )

        system_prompt = modules.calls[0].guide.lower()
        payload = modules.calls[0].payload

        self.assertNotIn("singing with passion", system_prompt)
        self.assertNotIn("lip sync only during vocal intervals", system_prompt)
        self.assertIn("dialogue-free", system_prompt)
        self.assertIn("gaze, posture, hands, body movement", system_prompt)
        self.assertTrue(payload["silent_mode"])
        self.assertNotIn("singing with passion", payload["performance_policy"].lower())

    def test_scene_prompts_persist_silent_mode_for_render_plan(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=[
                    {
                        "segment_id": "segment_001",
                        "type": "vocals",
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "lyrics": "hello",
                    }
                ],
                concept_prompts={"segment_001": "Mara tells the story without dialogue."},
                scene_details={"segment_001": {}},
                global_context={
                    "subject": "Mara",
                    "story_idea": "A wordless stage performance.",
                    "style": "cinematic realism",
                    "locations": ["Mirror Stage"],
                    "prompt_guidance": {},
                    "silent_mode": True,
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(data[0]["silent_mode"])

    def test_zimage_prompt_payload_includes_location_constraint(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)

        builder.build_zimage_prompt(
            segment={"segment_id": "segment_001", "type": "instrumental"},
            concept="A youth kneels at a spring.",
            global_context={
                "subject": "a Greek youth",
                "story_idea": "A forest myth.",
                "style": "chiaroscuro",
                "locations": ["ancient forest", "secluded spring"],
                "location_constraint": "Allowed locations: ancient forest, secluded spring",
                "prompt_guidance": {},
            },
        )

        payload = modules.calls[0].payload

        self.assertEqual("Allowed locations: ancient forest, secluded spring", payload["location_constraint"])

    def test_build_scene_prompts_carries_reference_metadata_from_concept_dict(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)
        stage1_segments = [
            {
                "segment_id": "segment_001",
                "type": "vocals",
                "start": 0.0,
                "end": 4.0,
                "duration": 4.0,
                "lyrics": "hello",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=stage1_segments,
                concept_prompts={
                    "segment_001": {
                        "concept": "Mara sings on the mirror stage.",
                        "references": {"actor_ids": ["singer"], "location_id": "stage"},
                    }
                },
                scene_details={},
                global_context={
                    "subject": "Mara",
                    "story_idea": "A stage performance.",
                    "style": "cinematic realism",
                    "locations": ["Mirror Stage"],
                    "location_constraint": "Allowed locations: Mirror Stage",
                    "prompt_guidance": {},
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual({"actor_ids": ["singer"], "location_id": "stage"}, data[0]["references"])

    def test_build_scene_prompts_passes_selected_cast_to_t2i_and_i2v(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.build_scene_prompts(
                stage1_segments=[{"segment_id": "segment_005", "type": "mixed"}],
                concept_prompts={
                    "segment_005": {
                        "concept": "The party approaches the gate.",
                        "references": {
                            "actor_ids": ["warrior_lead", "mage_lead", "rogue_lead"],
                            "location_id": "cathedral",
                        },
                    }
                },
                scene_details={"segment_005": {"camera_motion": "dolly", "character_motion": "advance"}},
                global_context={
                    "subject": "warrior_lead",
                    "story_idea": "A party enters a ruined cathedral.",
                    "style": "cinematic realism",
                    "locations": ["Cathedral"],
                    "actors": [
                        {"id": "warrior_lead", "name": "The Warrior"},
                        {"id": "mage_lead", "name": "The Mage"},
                        {"id": "rogue_lead", "name": "The Rogue"},
                    ],
                    "structured_locations": [{"id": "cathedral", "name": "Cathedral"}],
                    "subject_mode": "multi",
                    "max_scene_actors": 4,
                },
                output_json_path=Path(temp_dir) / "scene_prompts.json",
                artifact_store=JsonArtifactStore(),
            )

        t2i_payload = modules.calls[0].payload
        i2v_payload = modules.calls[1].payload
        expected_ids = ["warrior_lead", "mage_lead", "rogue_lead"]
        self.assertEqual(expected_ids, t2i_payload["scene_cast"]["visible_actor_ids"])
        self.assertEqual("warrior_lead", t2i_payload["scene_cast"]["primary_actor_id"])
        self.assertTrue(t2i_payload["scene_cast"]["requires_group_staging"])
        self.assertEqual(expected_ids, i2v_payload["scene_cast"]["visible_actor_ids"])

    def test_h3_scene_prompt_builder_preserves_more_than_four_selected_actors(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)
        actor_ids = ["actor_1", "actor_2", "actor_3", "actor_4", "actor_5", "actor_6"]

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.build_scene_prompts(
                stage1_segments=[{"segment_id": "segment_006", "type": "mixed"}],
                concept_prompts={
                    "segment_006": {
                        "concept": "The full band performs together.",
                        "references": {"actor_ids": actor_ids, "location_id": "stage"},
                    }
                },
                scene_details={},
                global_context={
                    "subject": "actor_1",
                    "story_idea": "A band performs.",
                    "style": "cinematic realism",
                    "locations": ["Stage"],
                    "actors": [{"id": actor_id, "name": actor_id} for actor_id in actor_ids],
                    "structured_locations": [{"id": "stage", "name": "Stage"}],
                    "subject_mode": "multi",
                    "max_scene_actors": 8,
                    "video_pipeline": "minimax-h3-r2v",
                },
                output_json_path=Path(temp_dir) / "scene_prompts.json",
                artifact_store=JsonArtifactStore(),
            )

        self.assertEqual(actor_ids, modules.calls[0].payload["scene_cast"]["visible_actor_ids"])

    def test_single_subject_mode_forces_first_actor_reference(self):
        modules = GeneralModulesFake()
        builder = ScenePromptBuilder(object(), modules=modules)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=[{"segment_id": "segment_001", "scene": 1, "type": "vocals"}],
                concept_prompts={
                    "segment_001": {
                        "concept": "Mara and Jon stand on the mirror stage.",
                        "references": {"actor_ids": ["mara", "jon"], "location_id": "stage"},
                    }
                },
                scene_details={},
                global_context={
                    "subject": "Mara",
                    "story_idea": "A stage performance.",
                    "style": "cinematic realism",
                    "locations": ["Mirror Stage"],
                    "actors": [{"id": "mara", "name": "Mara"}, {"id": "jon", "name": "Jon"}],
                    "structured_locations": [{"id": "stage", "name": "Mirror Stage"}],
                    "subject_mode": "single",
                    "max_scene_actors": 1,
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual({"actor_ids": ["mara"], "location_id": "stage"}, data[0]["references"])


if __name__ == "__main__":
    unittest.main()
