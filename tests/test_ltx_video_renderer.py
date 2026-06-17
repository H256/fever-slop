import unittest

from ltx_video_renderer import LTXVideoRenderer
from workflow_patcher import WorkflowPatcher


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


class RollingFrameSpecTests(unittest.TestCase):
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
