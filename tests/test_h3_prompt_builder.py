import unittest
import json
from pathlib import Path

from feverslop.domain.render_plan import RenderScene
from feverslop.prompting.minimax_h3_prompt_style import (
    build_h3_video_system_prompt,
)


class FakeLLMPort:
    """Minimal LLM mock for testing."""
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append((system_prompt, prompt))
        return self.response


class FakeArtifactStore:
    """Minimal artifact store mock for testing."""
    def __init__(self):
        self.writes = {}
        self.reads = {}

    def write_json(self, path, data):
        self.writes[str(path)] = data
        return Path(str(path))

    def read_json(self, path):
        return self.writes.get(str(path), self.reads.get(str(path), []))


# ─── System Prompt Builder ───────────────────────────────────────────────────

class BuildH3VideoSystemPromptTests(unittest.TestCase):
    def test_base_mode_contains_three_fields(self):
        prompt = build_h3_video_system_prompt(mode="base")
        self.assertIn("integrated_multimodal_description", prompt)
        self.assertIn("overall_soundscape", prompt)
        self.assertIn("non_diegetic_music", prompt)

    def test_ref_mode_contains_six_sections(self):
        prompt = build_h3_video_system_prompt(mode="ref")
        self.assertIn("subject_definitions", prompt)
        self.assertIn("summary", prompt)
        self.assertIn("retention_analysis", prompt)
        self.assertIn("detailed_description", prompt)
        self.assertIn("overall_soundscape", prompt)
        self.assertIn("non_diegetic_music", prompt)

    def test_contains_camera_motion_vocabulary(self):
        prompt = build_h3_video_system_prompt(mode="base")
        for term in ("Zoom", "Pan", "Push", "Tracking"):
            self.assertIn(term, prompt, f"Camera term '{term}' missing")

    def test_music_video_sets_audio_fields_na(self):
        prompt = build_h3_video_system_prompt(mode="base", video_type="music_video")
        self.assertIn("N/A", prompt)

    def test_movie_type_not_music_video(self):
        prompt = build_h3_video_system_prompt(mode="base", video_type="movie")
        # Should NOT say N/A for both audio fields
        self.assertNotIn("N/A (the music video", prompt)

    def test_silent_mode_forbids_vocal_terms(self):
        prompt = build_h3_video_system_prompt(mode="base", silent_mode=True)
        self.assertIn("SILENT MODE", prompt)

    def test_base_returns_string(self):
        result = build_h3_video_system_prompt(mode="base")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 500)

    def test_ref_returns_string(self):
        result = build_h3_video_system_prompt(mode="ref")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 500)

    def test_ref_mode_no_references(self):
        prompt = build_h3_video_system_prompt(mode="ref", references=None)
        self.assertIsInstance(prompt, str)

    def test_ref_mode_with_references(self):
        refs = [{"label": "Alice", "type": "image"}]
        prompt = build_h3_video_system_prompt(mode="ref", references=refs)
        self.assertIn("Alice", prompt)
        self.assertIn("<Picture 1>", prompt)

    def test_ref_retention_markers(self):
        prompt = build_h3_video_system_prompt(mode="ref")
        for marker in ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"):
            self.assertIn(marker, prompt)


# ─── H3PromptBuilder ─────────────────────────────────────────────────────────

class H3PromptBuilderTests(unittest.TestCase):
    def _get_builder(self, response: str):
        from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
        builder = H3PromptBuilder(FakeLLMPort(response))
        return builder

    def test_produces_structured_output(self):
        llm_response = json.dumps({
            "integrated_multimodal_description": "A person standing in a room.",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        })
        builder = self._get_builder(llm_response)
        result = builder.build_h3_prompt(
            segment={"segment_id": "seg1", "type": "vocals"},
            concept="A person in a room.",
            scene_details={"camera_motion": "Push In", "character_motion": "standing"},
            global_context={
                "subject": "a person",
                "story_idea": "alone",
                "style": "cinematic",
                "locations": ["room"],
                "silent_mode": False,
                "location_constraint": "",
            },
            mode="base",
        )
        self.assertIn("integrated_multimodal_description", result)
        self.assertIn("prompt", result)
        self.assertIn("A person standing", result["prompt"])

    def test_forwards_system_prompt(self):
        llm_response = json.dumps({
            "integrated_multimodal_description": "X",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        })
        llm = FakeLLMPort(llm_response)
        from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
        builder = H3PromptBuilder(llm)
        builder.build_h3_prompt(
            segment={"segment_id": "s1", "type": "vocals"},
            concept="test",
            scene_details={},
            global_context={
                "subject": "person",
                "story_idea": "story",
                "style": "style",
                "locations": ["loc"],
                "silent_mode": False,
                "location_constraint": "",
            },
        )
        self.assertEqual(len(llm.calls), 1)
        system_prompt, payload = llm.calls[0]
        self.assertIn("integrated_multimodal_description", system_prompt)
        parsed = json.loads(payload)
        self.assertEqual(parsed["scene_concept"], "test")

    def test_fallback_on_parse_failure(self):
        response = "not valid json at all"
        builder = self._get_builder(response)
        result = builder.build_h3_prompt(
            segment={"segment_id": "s1", "type": "vocals"},
            concept="test",
            scene_details={},
            global_context={
                "subject": "person",
                "story_idea": "story",
                "style": "style",
                "locations": ["loc"],
                "silent_mode": False,
                "location_constraint": "",
            },
        )
        self.assertIn("prompt", result)
        self.assertIn("not valid json", result["prompt"])

    def test_ref_mode_output(self):
        llm_response = json.dumps({
            "subject_definitions": "Subject 1 is a person.",
            "summary": "A scene in a room.",
            "retention_analysis": "<Picture 1>: face=fully_preserved",
            "detailed_description": "A person in a room.",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        })
        builder = self._get_builder(llm_response)
        result = builder.build_h3_prompt(
            segment={"segment_id": "s1", "type": "vocals"},
            concept="test",
            scene_details={},
            global_context={
                "subject": "person",
                "story_idea": "story",
                "style": "style",
                "locations": ["loc"],
                "silent_mode": False,
                "location_constraint": "",
            },
            mode="ref",
        )
        self.assertIn("subject_definitions", result)
        self.assertIn("prompt", result)


# ─── H3PromptBuilder Batch ──────────────────────────────────────────────────

class H3PromptBuilderBatchTests(unittest.TestCase):
    def test_batch_writes_all_scenes(self):
        llm_response = json.dumps({
            "integrated_multimodal_description": "Scene content.",
            "overall_soundscape": "N/A",
            "non_diegetic_music": "N/A",
        })
        from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
        store = FakeArtifactStore()
        builder = H3PromptBuilder(FakeLLMPort(llm_response))
        builder.build_all_h3_prompts(
            stage1_segments=[
                {"segment_id": "seg1", "type": "vocals"},
                {"segment_id": "seg2", "type": "instrumental"},
            ],
            concept_prompts={"seg1": "concept one", "seg2": "concept two"},
            scene_details={"seg1": {}, "seg2": {}},
            global_context={
                "subject": "person",
                "story_idea": "story",
                "style": "style",
                "locations": ["loc"],
                "silent_mode": False,
                "location_constraint": "",
            },
            output_json_path="output.json",
            artifact_store=store,
        )
        data = store.writes.get("output.json", [])
        self.assertEqual(len(data), 2)
        seg_ids = [entry["segment_id"] for entry in data]
        self.assertIn("seg1", seg_ids)
        self.assertIn("seg2", seg_ids)


# ─── RenderScene video_prompt priority ───────────────────────────────────────

class RenderSceneH3PriorityTests(unittest.TestCase):
    def test_video_prompt_returns_h3_when_present(self):
        scene = RenderScene.from_dict({
            "scene": 1,
            "h3": {"prompt": "H3 structured prompt here"},
            "ltx": {
                "original_style_i2v_prompt": "LTX prompt",
                "i2v_prompt_from_t2i": "LTX I2V",
                "base_prompt": "LTX base",
            },
        })
        self.assertEqual(scene.video_prompt, "H3 structured prompt here")

    def test_video_prompt_falls_back_to_ltx_when_no_h3(self):
        scene = RenderScene.from_dict({
            "scene": 1,
            "ltx": {
                "original_style_i2v_prompt": "LTX fallback prompt",
            },
        })
        self.assertEqual(scene.video_prompt, "LTX fallback prompt")

    def test_video_prompt_falls_back_to_ltx_when_h3_empty(self):
        scene = RenderScene.from_dict({
            "scene": 1,
            "h3": {"prompt": ""},
            "ltx": {
                "original_style_i2v_prompt": "LTX fallback",
            },
        })
        self.assertEqual(scene.video_prompt, "LTX fallback")

    def test_video_prompt_falls_back_chain_i2v(self):
        scene = RenderScene.from_dict({
            "scene": 1,
            "ltx": {
                "i2v_prompt_from_t2i": "I2V prompt",
                "base_prompt": "Base prompt",
            },
        })
        self.assertEqual(scene.video_prompt, "I2V prompt")

    def test_video_prompt_falls_back_chain_base(self):
        scene = RenderScene.from_dict({
            "scene": 1,
            "ltx": {
                "base_prompt": "Only base",
            },
        })
        self.assertEqual(scene.video_prompt, "Only base")

    def test_video_prompt_empty_when_no_data(self):
        scene = RenderScene.from_dict({"scene": 1})
        self.assertEqual(scene.video_prompt, "")

    def test_no_regression_ltx_only(self):
        scene = RenderScene.from_dict({
            "scene": 1,
            "ltx": {
                "original_style_i2v_prompt": "Original",
                "i2v_prompt_from_t2i": "From T2I",
                "base_prompt": "Base",
            },
        })
        self.assertEqual(scene.video_prompt, "Original")


# ─── Render plan builder with H3 ─────────────────────────────────────────────

class BuildRenderPlanH3Tests(unittest.TestCase):
    def test_render_plan_excludes_h3_when_not_provided(self):
        from feverslop.pipeline.render_plan_builder import build_render_plan
        store = FakeArtifactStore()
        store.reads["scene_prompts.json"] = [
            {
                "scene": 1,
                "segment_id": "seg1",
                "type": "vocals",
                "start": 0,
                "end": 5,
                "duration": 5,
                "zimage_prompt": "zimg",
                "i2v_prompt_from_t2i": "i2v",
            }
        ]
        store.reads["relay.json"] = [
            {
                "scene": 1,
                "prompt_relay": [
                    {"frame_start": 0, "frame_end": 120, "state": "singing"}
                ]
            }
        ]

        class FakeVideoSettings:
            fps = 24
            width = 1280
            height = 720
            def scene_frame_count_between(self, start, end):
                return int(self.fps * (end - start))

        build_render_plan(
            scene_prompts_json="scene_prompts.json",
            ltx_prompt_relay_json="relay.json",
            output_json_file="output.json",
            video_settings=FakeVideoSettings(),
            artifact_store=store,
        )

        output = store.writes["output.json"]
        self.assertEqual(len(output), 1)
        self.assertNotIn("h3", output[0])
        self.assertIn("ltx", output[0])

    def test_render_plan_includes_h3_when_provided(self):
        from feverslop.pipeline.render_plan_builder import build_render_plan
        store = FakeArtifactStore()
        store.reads["scene_prompts.json"] = [
            {
                "scene": 1,
                "segment_id": "seg1",
                "type": "vocals",
                "start": 0,
                "end": 5,
                "duration": 5,
                "zimage_prompt": "zimg",
                "i2v_prompt_from_t2i": "i2v",
            }
        ]
        store.reads["relay.json"] = [
            {
                "scene": 1,
                "prompt_relay": [
                    {"frame_start": 0, "frame_end": 120, "state": "singing"}
                ]
            }
        ]
        store.reads["h3.json"] = [
            {"segment_id": "seg1", "prompt": "H3 prompt content"}
        ]

        class FakeVideoSettings:
            fps = 24
            width = 1280
            height = 720
            def scene_frame_count_between(self, start, end):
                return int(self.fps * (end - start))

        build_render_plan(
            scene_prompts_json="scene_prompts.json",
            ltx_prompt_relay_json="relay.json",
            output_json_file="output.json",
            video_settings=FakeVideoSettings(),
            artifact_store=store,
            h3_prompts_json="h3.json",
        )

        output = store.writes["output.json"]
        self.assertEqual(len(output), 1)
        self.assertIn("h3", output[0])
        self.assertEqual(output[0]["h3"]["prompt"], "H3 prompt content")
        self.assertIn("ltx", output[0])



class BuildReferencesFromSegmentTests(unittest.TestCase):
    """Test _build_references_from_segment helper."""

    def setUp(self):
        from feverslop.prompting.h3_prompt_builder import (
            _build_references_from_segment,
        )
        self.build_refs = _build_references_from_segment

    def test_no_references_returns_none(self):
        segment = {"segment_id": "s1"}
        self.assertIsNone(self.build_refs(segment))

    def test_empty_references_returns_none(self):
        segment = {"segment_id": "s1", "references": {}}
        self.assertIsNone(self.build_refs(segment))

    def test_image_refs_only(self):
        segment = {
            "segment_id": "s1",
            "references": {
                "reference_image_paths": ["output/actor1.png", "output/loc1.png"],
            },
        }
        result = self.build_refs(segment)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "image")
        self.assertEqual(result[0]["label"], "actor1")
        self.assertEqual(result[1]["label"], "loc1")

    def test_ref_items_provide_labels(self):
        segment = {
            "segment_id": "s1",
            "references": {
                "reference_image_paths": ["output/actor1_msr.png", "output/loc1_msr.png"],
            },
            "ref_items": [
                {"type": "actor", "name": "Jane", "visual_description": "blonde"},
                {"type": "location", "name": "Studio", "visual_description": "dark"},
            ],
        }
        result = self.build_refs(segment)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["label"], "Jane")
        self.assertEqual(result[1]["label"], "Studio")

    def test_video_and_audio_refs(self):
        segment = {
            "segment_id": "s1",
            "references": {
                "reference_video_paths": ["output/scene_vid.mp4"],
                "reference_audio_paths": ["music/song.wav"],
            },
        }
        result = self.build_refs(segment)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "video")
        self.assertEqual(result[1]["type"], "audio")

    def test_mixed_refs(self):
        segment = {
            "segment_id": "s1",
            "references": {
                "reference_image_paths": ["output/actor1.png"],
                "reference_audio_paths": ["music/song.wav"],
            },
            "ref_items": [
                {"type": "actor", "name": "Hero"},
            ],
        }
        result = self.build_refs(segment)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"label": "Hero", "type": "image"})
        self.assertEqual(result[1], {"label": "song", "type": "audio"})


class RefModeSystemPromptAudioPreservationTests(unittest.TestCase):
    """Test that ref-mode system prompt includes audio preservation for music videos."""

    def test_music_video_has_audio_preservation_section(self):
        prompt = build_h3_video_system_prompt(
            mode="ref",
            video_type="music_video",
            silent_mode=False,
        )
        self.assertIn("## Audio Preservation (Music Video)", prompt)
        self.assertIn("audio=fully_preserved", prompt)
        self.assertIn("reference audio IS the soundtrack", prompt)

    def test_movie_type_has_no_audio_preservation_section(self):
        prompt = build_h3_video_system_prompt(
            mode="ref",
            video_type="movie",
            silent_mode=False,
        )
        self.assertNotIn("## Audio Preservation (Music Video)", prompt)
        self.assertNotIn("reference audio IS the soundtrack", prompt)

    def test_base_mode_has_no_audio_preservation(self):
        prompt = build_h3_video_system_prompt(
            mode="base",
            video_type="music_video",
            silent_mode=False,
        )
        self.assertNotIn("## Audio Preservation (Music Video)", prompt)


class RefModeWithReferencesLabelTests(unittest.TestCase):
    """Test that references produce correct <Picture N>/<Audio N> tags."""

    def test_audio_refs_produce_audio_tags(self):
        references = [
            {"label": "Song Track", "type": "audio"},
        ]
        prompt = build_h3_video_system_prompt(
            mode="ref",
            video_type="music_video",
            silent_mode=False,
            references=references,
        )
        self.assertIn("<Audio 1>", prompt)

    def test_image_refs_produce_picture_tags(self):
        references = [
            {"label": "Actor Jane", "type": "image"},
            {"label": "Studio Room", "type": "image"},
        ]
        prompt = build_h3_video_system_prompt(
            mode="ref",
            video_type="music_video",
            silent_mode=False,
            references=references,
        )
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 2>", prompt)
        self.assertIn("Actor Jane", prompt)
        self.assertIn("Studio Room", prompt)

    def test_video_refs_produce_video_tags(self):
        references = [
            {"label": "Intro Clip", "type": "video"},
        ]
        prompt = build_h3_video_system_prompt(
            mode="ref",
            video_type="music_video",
            silent_mode=False,
            references=references,
        )
        self.assertIn("<Video 1>", prompt)

    def test_mixed_refs_correct_numbering(self):
        references = [
            {"label": "Actor", "type": "image"},
            {"label": "Song", "type": "audio"},
        ]
        prompt = build_h3_video_system_prompt(
            mode="ref",
            video_type="music_video",
            silent_mode=False,
            references=references,
        )
        self.assertIn("<Picture 1>: Actor", prompt)
        self.assertIn("<Audio 2>: Song", prompt)  # sequential numbering: image=1, audio=2

    def test_builder_passes_references_for_ref_mode(self):
        """End-to-end: H3PromptBuilder passes references to system prompt for ref mode."""
        from feverslop.prompting.h3_prompt_builder import H3PromptBuilder

        segment_with_refs = {
            "segment_id": "s1",
            "type": "vocals",
            "references": {
                "reference_image_paths": ["output/actor.png"],
                "reference_audio_paths": ["music/song.wav"],
            },
            "ref_items": [
                {"type": "actor", "name": "Hero"},
            ],
        }
        llm = FakeLLMPort(json.dumps({"prompt": "test output"}))
        builder = H3PromptBuilder(llm)
        builder.build_h3_prompt(
            segment=segment_with_refs,
            concept="A hero sings",
            scene_details={},
            global_context={},
            mode="ref",
            video_type="music_video",
        )
        system_prompt = llm.calls[0][0]
        # References should be in the system prompt
        self.assertIn("<Picture 1>", system_prompt)
        self.assertIn("Hero", system_prompt)

    def test_builder_no_references_when_no_refs(self):
        """Builder should not inject reference labels when segment has no refs."""
        from feverslop.prompting.h3_prompt_builder import H3PromptBuilder

        segment_no_refs = {
            "segment_id": "s1",
            "type": "vocals",
        }
        llm = FakeLLMPort(json.dumps({"prompt": "test output"}))
        builder = H3PromptBuilder(llm)
        builder.build_h3_prompt(
            segment=segment_no_refs,
            concept="Test",
            scene_details={},
            global_context={},
            mode="ref",
            video_type="music_video",
        )
        system_prompt = llm.calls[0][0]
        # Should NOT contain ## Reference Labels Used (only the generic instruction)
        self.assertNotIn("## Reference Labels Used", system_prompt)
        # Generic instruction should still be present
        self.assertIn("## Reference Labels", system_prompt)


class VideoPipelineModeResolutionTests(unittest.TestCase):
    """Test that H3PromptPipeline derives mode from config.video_pipeline."""

    def test_r2v_derives_ref_mode(self):
        from feverslop.application.h3_prompt_pipeline import H3PromptPipeline
        pipeline = H3PromptPipeline(
            llm_factory=lambda c: None,
            h3_prompt_builder_factory=lambda llm: None,
        )
        # Verify required_keys includes config
        self.assertIn("config", pipeline.required_keys)

        class FakeConfig:
            video_pipeline = "minimax-h3-r2v"

        class FakeArtifactStore:
            def write_json(self, path, data): return path
            def read_json(self, path): return []
        build_count = {"mode": None}

        class FakeBuilder:
            def build_all_h3_prompts(self, **kwargs):
                build_count["mode"] = kwargs.get("mode")
                return "path"

        pipeline.h3_prompt_builder_factory = lambda llm: FakeBuilder()
        ctx = {
            "app_config": {},
            "config": FakeConfig(),
            "stage1_segments": [{"segment_id": "s1"}],
            "concept_prompts": {},
            "scene_details": {},
            "global_context": {},
            "h3_prompts_json": "test.json",
            "artifact_store": FakeArtifactStore(),
            "log_step": lambda x: None,
            "log_file": lambda a, b: None,
            "run_spinner": lambda msg, fn: fn(),
        }
        for key in pipeline.required_keys:
            if key not in ctx:
                ctx[key] = None

        pipeline.run(ctx)
        self.assertEqual(build_count["mode"], "ref")

    def test_base_pipeline_derives_base_mode(self):
        from feverslop.application.h3_prompt_pipeline import H3PromptPipeline

        class FakeConfig:
            video_pipeline = "ltx_i2v"

        class FakeArtifactStore:
            def write_json(self, path, data): return path
            def read_json(self, path): return []
        build_count = {"mode": None}

        class FakeBuilder:
            def build_all_h3_prompts(self, **kwargs):
                build_count["mode"] = kwargs.get("mode")
                return "path"

        pipeline = H3PromptPipeline(
            llm_factory=lambda c: None,
            h3_prompt_builder_factory=lambda llm: FakeBuilder(),
        )
        ctx = {
            "app_config": {},
            "config": FakeConfig(),
            "stage1_segments": [{"segment_id": "s1"}],
            "concept_prompts": {},
            "scene_details": {},
            "global_context": {},
            "h3_prompts_json": "test.json",
            "artifact_store": FakeArtifactStore(),
            "log_step": lambda x: None,
            "log_file": lambda a, b: None,
            "run_spinner": lambda msg, fn: fn(),
        }
        for key in pipeline.required_keys:
            if key not in ctx:
                ctx[key] = None

        pipeline.run(ctx)
        self.assertEqual(build_count["mode"], "base")


if __name__ == "__main__":
    unittest.main()

