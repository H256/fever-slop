import unittest

from feverslop.domain.ltx_rendering import (
    AudioWindowSpec,
    PromptRelayPayloadBuilder,
    build_audio_window_spec,
)


class LTXRenderingDomainTests(unittest.TestCase):
    def test_build_audio_window_spec_handles_original_rolling_profile(self):
        spec = build_audio_window_spec(
            scene_number=2,
            fps=25,
            scene_frame_count=101,
            scene_start_seconds=10.0,
            preroll_frames=50,
            tail_loss_frames=25,
            round_render_frames_to_8n1=True,
        )

        self.assertIsInstance(spec, AudioWindowSpec)
        self.assertEqual(177, spec.render_frame_count)
        self.assertEqual(50, spec.trim_front_frames)
        self.assertEqual(26, spec.tail_loss_frames)
        self.assertAlmostEqual(8.0, spec.audio_start_seconds)
        self.assertAlmostEqual(176 / 25, spec.audio_duration_seconds)

    def test_effective_preroll_is_integer_not_float(self):
        spec = build_audio_window_spec(
            scene_number=2,
            fps=24,
            scene_frame_count=101,
            scene_start_seconds=10.0,
            preroll_frames=50,
            tail_loss_frames=25,
            round_render_frames_to_8n1=True,
        )
        self.assertIsInstance(spec.trim_front_frames, int)

    def test_audio_duration_seconds_is_rounded(self):
        spec = build_audio_window_spec(
            scene_number=1,
            fps=24,
            scene_frame_count=101,
            scene_start_seconds=0.0,
            preroll_frames=0,
            tail_loss_frames=0,
            round_render_frames_to_8n1=False,
        )
        self.assertIsInstance(spec.audio_duration_seconds, float)
        self.assertEqual(round(spec.audio_duration_seconds, 6), spec.audio_duration_seconds)

    def test_first_scene_audio_window_does_not_seek_before_zero(self):
        spec = build_audio_window_spec(
            scene_number=1,
            fps=25,
            scene_frame_count=101,
            scene_start_seconds=0.0,
            preroll_frames=50,
            tail_loss_frames=25,
            round_render_frames_to_8n1=False,
        )

        self.assertEqual(126, spec.render_frame_count)
        self.assertEqual(0, spec.trim_front_frames)
        self.assertEqual(25, spec.tail_loss_frames)
        self.assertEqual(0.0, spec.audio_start_seconds)

    def test_prompt_relay_payload_merges_short_segments_and_preserves_frame_total(self):
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

        payload = PromptRelayPayloadBuilder().build(
            scene=scene,
            render_frame_count=162,
            trim_front_frames=0,
            tail_loss_frames=0,
        )

        self.assertEqual("base", payload.global_prompt)
        self.assertEqual("6,29,21,99,6", payload.segment_lengths)
        self.assertEqual(161, sum(int(value) for value in payload.segment_lengths.split(",")))

    def test_prompt_relay_payload_supports_frames_mode(self):
        scene = {
            "scene": 1,
            "frame_count": 24,
            "ltx": {
                "base_prompt": "base",
                "prompt_relay": [],
            },
        }

        payload = PromptRelayPayloadBuilder(segment_length_mode="frames").build(
            scene=scene,
            render_frame_count=27,
            trim_front_frames=2,
            tail_loss_frames=1,
        )

        self.assertEqual(27, sum(int(value) for value in payload.segment_lengths.split(",")))


if __name__ == "__main__":
    unittest.main()
