import unittest
import tempfile
import json
from pathlib import Path

from ltx_video_renderer import AudioWindowSpec, LTXVideoRenderer
from autoprompter.adapters.workflow_patcher import WorkflowPatcher


class PromptRelayPayloadTests(unittest.TestCase):
    def _renderer(self) -> LTXVideoRenderer:
        return LTXVideoRenderer(
            client=None,
            ltx_workflow_path="workflow.json",
            output_dir="out",
        )

    def test_prompt_relay_segments_are_never_shorter_than_six_frames(self):
        scene = {
            "scene": 16,
            "frame_count": 162,
            "ltx": {
                "base_prompt": "base",
                "prompt_relay": [
                    {"frame_start": 0, "frame_end": 6, "prompt": "singing intro"},
                    {"frame_start": 35, "frame_end": 56, "prompt": "singing middle"},
                    {"frame_start": 155, "frame_end": 156, "prompt": "one frame glitch"},
                    {"frame_start": 156, "frame_end": 161, "prompt": "singing ending"},
                ],
            },
        }

        _, _, segment_lengths = self._renderer()._build_prompt_relay_payload(
            scene=scene,
            render_frame_count=162,
            trim_front_frames=0,
            tail_loss_frames=0,
        )

        lengths = [int(value) for value in segment_lengths.split(",")]
        self.assertEqual([6, 29, 21, 99, 6], lengths)
        self.assertEqual(161, sum(lengths))
        self.assertTrue(all(length >= 6 for length in lengths))

    def test_prompt_relay_preroll_and_tail_are_merged_when_short(self):
        scene = {
            "scene": 1,
            "frame_count": 25,
            "ltx": {
                "base_prompt": "base",
                "prompt_relay": [
                    {"frame_start": 0, "frame_end": 24, "prompt": "main prompt"},
                ],
            },
        }

        _, _, segment_lengths = self._renderer()._build_prompt_relay_payload(
            scene=scene,
            render_frame_count=32,
            trim_front_frames=3,
            tail_loss_frames=4,
        )

        lengths = [int(value) for value in segment_lengths.split(",")]
        self.assertEqual([31], lengths)
        self.assertEqual(31, sum(lengths))
        self.assertTrue(all(length >= 6 for length in lengths))


class LTXRenderModeTests(unittest.TestCase):
    def test_auto_mode_uses_scene_render_mode_hint(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="relay.json",
            output_dir="out",
            single_prompt_workflow_path="single.json",
            render_mode="auto",
        )

        self.assertEqual(
            "single_prompt",
            renderer._render_mode_for_scene({"ltx": {"render_mode_hint": "single_prompt"}}),
        )
        self.assertEqual(
            "relay",
            renderer._render_mode_for_scene({"ltx": {"render_mode_hint": "relay"}}),
        )

    def test_single_prompt_mode_patches_prompt_node_with_original_style_prompt(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="single.json",
            output_dir="out",
            render_mode="single_prompt",
            single_prompt_node_title="#PROMPT",
            single_prompt_input_name="text",
        )
        patcher = WorkflowPatcher({
            "1": {
                "inputs": {"text": ""},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "#PROMPT"},
            }
        })
        scene = {
            "ltx": {
                "base_prompt": "base prompt",
                "original_style_i2v_prompt": "original style prompt",
            }
        }

        renderer._patch_prompt_inputs(patcher, scene, mode="single_prompt", render_frame_count=24, trim_front_frames=0, tail_loss_frames=0)

        self.assertEqual("original style prompt", patcher.get()["1"]["inputs"]["text"])

    def test_single_prompt_mode_falls_back_to_prompt_positive_title(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="single.json",
            output_dir="out",
            render_mode="single_prompt",
            single_prompt_node_title="#PROMPT",
            single_prompt_input_name="text",
        )
        patcher = WorkflowPatcher({
            "1": {
                "inputs": {"text": ""},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "#PROMPT_POSITIVE"},
            }
        })
        scene = {
            "ltx": {
                "base_prompt": "base prompt",
                "original_style_i2v_prompt": "fallback prompt",
            }
        }

        renderer._patch_prompt_inputs(patcher, scene, mode="single_prompt", render_frame_count=24, trim_front_frames=0, tail_loss_frames=0)

        self.assertEqual("fallback prompt", patcher.get()["1"]["inputs"]["text"])


class LTXLoraTests(unittest.TestCase):
    def _workflow_with_titles(self, titles: list[str]) -> str:
        return "{" + ",".join(
            f'"{index}": {{"inputs": {{}}, "_meta": {{"title": "{title}"}}}}'
            for index, title in enumerate(titles, start=1)
        ) + "}"

    def _relay_titles(self) -> list[str]:
        return [
            "#WIDTH",
            "#HEIGHT",
            "#LOAD_AUDIO",
            "#TRIM_AUDIO",
            "#STARTFRAME",
            "#FRAMES",
            "#FRAMERATE",
            "#SEED",
            "#PROMPT_RELAY",
            "#SAVE_VIDEO",
        ]

    def _single_prompt_titles(self) -> list[str]:
        titles = self._relay_titles()
        titles.remove("#PROMPT_RELAY")
        titles.append("#PROMPT")
        return titles

    def test_workflow_patcher_patches_lora_1_name_and_strengths(self):
        patcher = WorkflowPatcher({
            "1": {
                "inputs": {
                    "lora_name": "",
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "#LORA_1"},
            }
        })

        patched = patcher.patch_lora_by_title(
            "#LORA_1",
            lora_name="characters/test.safetensors",
            strength_model=0.85,
            strength_clip=0.65,
        )

        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual(["lora_name", "strength_model", "strength_clip"], patched)
        self.assertEqual("characters/test.safetensors", inputs["lora_name"])
        self.assertEqual(0.85, inputs["strength_model"])
        self.assertEqual(0.65, inputs["strength_clip"])

    def test_lora_1_explicit_strengths_patch_workflow_default_without_name_patch(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="workflow.json",
            output_dir="out",
            lora_1_enabled=False,
            lora_1_name="",
            lora_1_strength_model=0.0,
            lora_1_strength_clip=0.0,
            lora_1_strengths_explicit=True,
        )
        patcher = WorkflowPatcher({
            "1": {
                "inputs": {
                    "lora_name": "workflow-default.safetensors",
                    "strength_model": 0.85,
                    "strength_clip": 0.85,
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "#LORA_1"},
            }
        })

        renderer._patch_lora_inputs(patcher)

        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual("workflow-default.safetensors", inputs["lora_name"])
        self.assertEqual(0.0, inputs["strength_model"])
        self.assertEqual(0.0, inputs["strength_clip"])

    def test_render_scene_queues_workflow_after_lora_1_explicit_strength_patch(self):
        class FakeClient:
            def __init__(self):
                self.queued_workflow = None

            def queue_prompt(self, workflow):
                self.queued_workflow = workflow
                return "prompt-id"

            def wait_for_completion(self, prompt_id):
                return {
                    "outputs": {
                        "save": {
                            "videos": [
                                {"filename": "scene_0001.mp4", "subfolder": "", "type": "output"}
                            ]
                        }
                    }
                }

            def download_view_file(self, filename, subfolder, file_type, output_path):
                return Path(output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(
                json.dumps({
                    "1": {"inputs": {"value": 1280}, "_meta": {"title": "#WIDTH"}},
                    "2": {"inputs": {"value": 704}, "_meta": {"title": "#HEIGHT"}},
                    "3": {"inputs": {"audio": "", "audioUI": ""}, "_meta": {"title": "#LOAD_AUDIO"}},
                    "4": {"inputs": {"start_index": 0, "duration": 1}, "_meta": {"title": "#TRIM_AUDIO"}},
                    "5": {"inputs": {"image": ""}, "_meta": {"title": "#STARTFRAME"}},
                    "6": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMES"}},
                    "7": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMERATE"}},
                    "8": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                    "9": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "10": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "11": {
                        "inputs": {
                            "lora_name": "workflow-default.safetensors",
                            "strength_model": 0.85,
                        },
                        "_meta": {"title": "#LORA_1"},
                    },
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            renderer = LTXVideoRenderer(
                client=client,
                ltx_workflow_path=workflow_path,
                output_dir=temp / "out",
                render_mode="single_prompt",
                lora_1_enabled=False,
                lora_1_strength_model=0.0,
                lora_1_strength_clip=0.0,
                lora_1_strengths_explicit=True,
                postprocess=False,
            )

            renderer.render_scene_video(
                scene={
                    "scene": 1,
                    "fps": 24,
                    "width": 1280,
                    "height": 704,
                    "ltx": {"original_style_i2v_prompt": "prompt"},
                },
                comfy_audio_name="audio.mp3",
                comfy_startframe_name="scene_0001.png",
                rolling={
                    "render_frame_count": 24,
                    "trim_front_frames": 0,
                    "tail_loss_frames": 0,
                    "audio_start_seconds": 0,
                    "audio_duration_seconds": 1,
                },
            )

            lora_node = next(
                node for node in client.queued_workflow.values()
                if node.get("_meta", {}).get("title") == "#LORA_1"
            )
            self.assertEqual("workflow-default.safetensors", lora_node["inputs"]["lora_name"])
            self.assertEqual(0.0, lora_node["inputs"]["strength_model"])

    def test_lora_disabled_does_not_require_lora_1_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = Path(temp_dir) / "workflow.json"
            workflow_path.write_text(self._workflow_with_titles(self._relay_titles()), encoding="utf-8")
            renderer = LTXVideoRenderer(
                client=None,
                ltx_workflow_path=workflow_path,
                output_dir="out",
                lora_1_enabled=False,
            )

            renderer.validate_workflow(mode="relay")

    def test_lora_enabled_requires_lora_1_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = Path(temp_dir) / "workflow.json"
            workflow_path.write_text(self._workflow_with_titles(self._relay_titles()), encoding="utf-8")
            renderer = LTXVideoRenderer(
                client=None,
                ltx_workflow_path=workflow_path,
                output_dir="out",
                lora_1_enabled=True,
                lora_1_name="characters/test.safetensors",
            )

            with self.assertRaisesRegex(ValueError, "#LORA_1.*workflow.json"):
                renderer.validate_workflow(mode="relay")

    def test_lora_enabled_validates_single_prompt_workflow_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            relay_path = Path(temp_dir) / "relay.json"
            single_path = Path(temp_dir) / "single.json"
            relay_path.write_text(
                self._workflow_with_titles([*self._relay_titles(), "#LORA_1"]),
                encoding="utf-8",
            )
            single_path.write_text(self._workflow_with_titles(self._single_prompt_titles()), encoding="utf-8")
            renderer = LTXVideoRenderer(
                client=None,
                ltx_workflow_path=relay_path,
                single_prompt_workflow_path=single_path,
                output_dir="out",
                lora_1_enabled=True,
                lora_1_name="characters/test.safetensors",
            )

            with self.assertRaisesRegex(ValueError, "#LORA_1.*single.json"):
                renderer.validate_workflow(mode="single_prompt")

    def test_single_prompt_validation_accepts_prompt_positive_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = Path(temp_dir) / "single.json"
            titles = self._single_prompt_titles()
            titles.remove("#PROMPT")
            titles.append("#PROMPT_POSITIVE")
            workflow_path.write_text(self._workflow_with_titles(titles), encoding="utf-8")
            renderer = LTXVideoRenderer(
                client=None,
                ltx_workflow_path=workflow_path,
                output_dir="out",
                render_mode="single_prompt",
                single_prompt_node_title="#PROMPT",
            )

            renderer.validate_workflow(mode="single_prompt")


class RollingFrameSpecTests(unittest.TestCase):
    def test_rolling_spec_is_typed_audio_window_with_mapping_compatibility(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="workflow.json",
            output_dir="out",
            preroll_frames=50,
            tail_loss_frames=25,
            round_render_frames_to_8n1=True,
        )
        scene = {
            "scene": 3,
            "fps": 25,
            "frame_count": 101,
            "abs_start_seconds": 12.0,
        }

        rolling = renderer._rolling_spec(scene)

        self.assertIsInstance(rolling, AudioWindowSpec)
        self.assertEqual(177, rolling.render_frame_count)
        self.assertEqual(177, rolling["render_frame_count"])
        self.assertEqual(50, rolling.trim_front_frames)
        self.assertEqual(26, rolling.tail_loss_frames)
        self.assertAlmostEqual(10.0, rolling.audio_start_seconds)
        self.assertAlmostEqual(176 / 25, rolling.audio_duration_seconds)
        self.assertAlmostEqual(101 / 25, rolling.output_duration_seconds)

    def test_preroll_and_tail_increase_render_frame_count_directly(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="workflow.json",
            output_dir="out",
            preroll_frames=50,
            tail_loss_frames=25,
        )
        scene = {
            "scene": 2,
            "fps": 25,
            "frame_count": 101,
            "abs_start_seconds": 10.0,
        }

        rolling = renderer._rolling_spec(scene)

        self.assertEqual(176, rolling["render_frame_count"])
        self.assertEqual(50, rolling["trim_front_frames"])
        self.assertEqual(25, rolling["tail_loss_frames"])

    def test_first_scene_audio_window_does_not_seek_before_song_start(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="workflow.json",
            output_dir="out",
            preroll_frames=50,
            tail_loss_frames=25,
        )
        scene = {
            "scene": 1,
            "fps": 25,
            "frame_count": 101,
            "abs_start_seconds": 0.0,
        }

        rolling = renderer._rolling_spec(scene)

        self.assertEqual(126, rolling.render_frame_count)
        self.assertEqual(0, rolling.trim_front_frames)
        self.assertAlmostEqual(0.0, rolling.audio_start_seconds)
        self.assertAlmostEqual(125 / 25, rolling.audio_duration_seconds)

    def test_original_rounding_adds_padding_to_effective_tail(self):
        renderer = LTXVideoRenderer(
            client=None,
            ltx_workflow_path="workflow.json",
            output_dir="out",
            preroll_frames=50,
            tail_loss_frames=25,
            round_render_frames_to_8n1=True,
        )
        scene = {
            "scene": 2,
            "fps": 25,
            "frame_count": 101,
            "abs_start_seconds": 10.0,
        }

        rolling = renderer._rolling_spec(scene)

        self.assertEqual(177, rolling["render_frame_count"])
        self.assertEqual(50, rolling["trim_front_frames"])
        self.assertEqual(26, rolling["tail_loss_frames"])


if __name__ == "__main__":
    unittest.main()
