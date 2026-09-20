import unittest

from feverslop.domain.h3_two_pass import (
    H3TwoPassSchemaError,
    I2VFrameMode,
    validate_i2v_frames,
)


class H3I2VTwoPassTests(unittest.TestCase):
    def test_start_only_mode(self):
        self.assertEqual(I2VFrameMode.START_ONLY, validate_i2v_frames("start.png"))

    def test_start_end_mode(self):
        self.assertEqual(
            I2VFrameMode.START_END, validate_i2v_frames("start.png", "end.png")
        )

    def test_missing_start_frame_fails_clearly(self):
        with self.assertRaises(H3TwoPassSchemaError) as ctx:
            validate_i2v_frames(None)
        self.assertIn("start frame", str(ctx.exception))

    def test_end_frame_alone_is_not_a_valid_mode(self):
        # I2V is image-to-video: without a start frame there is no source
        # image, so an end frame alone is rejected (not a silent mode).
        with self.assertRaises(H3TwoPassSchemaError):
            validate_i2v_frames(None, "end.png")

    def test_profile_sidecar_declares_both_capabilities(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        profile = json.loads(
            (root / "workflows" / "video" / "minimax_h3" / "i2v_two_pass.profile.json").read_text(encoding="utf-8")
        )
        self.assertEqual("i2v", profile["mode"])
        self.assertEqual(
            [mode.value for mode in I2VFrameMode], profile["frame_capabilities"]
        )


if __name__ == "__main__":
    unittest.main()
