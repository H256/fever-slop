import unittest

from feverslop.adapters.comfyui_seedvr2_backend import (
    ComfyUISeedVR2Backend,
    SeedVR2RenderSettings,
)


class ComfyUISeedVR2BackendTests(unittest.TestCase):
    def test_build_workflow_uses_patch_anchors_and_seedvr2_defaults(self):
        backend = ComfyUISeedVR2Backend(client=object())

        workflow = backend.build_workflow(
            video_name="feverslop/input/scene_0001.mp4",
            output_prefix="feverslop/seedvr2/scene_0001/pass_01",
            output_size=(1920, 1080),
            settings=SeedVR2RenderSettings(),
        )

        self.assertEqual("feverslop/input/scene_0001.mp4", workflow["1"]["inputs"]["video"])
        self.assertEqual(1920, workflow["1"]["inputs"]["custom_width"])
        self.assertEqual(1080, workflow["1"]["inputs"]["custom_height"])
        self.assertEqual("seedvr2_3b_int8_convrot.safetensors", workflow["5"]["inputs"]["unet_name"])
        self.assertEqual(0.35, workflow["7"]["inputs"]["denoise"])
        self.assertEqual(4, workflow["4"]["inputs"]["temporal_overlap"])
        self.assertEqual("feverslop/seedvr2/scene_0001/pass_01", workflow["12"]["inputs"]["filename_prefix"])

    def test_build_workflow_rejects_unsupported_color_correction(self):
        backend = ComfyUISeedVR2Backend(client=object())

        with self.assertRaisesRegex(ValueError, "color_correction"):
            backend.build_workflow(
                video_name="input.mp4",
                output_prefix="output",
                output_size=(1920, 1080),
                settings=SeedVR2RenderSettings(color_correction="invalid"),
            )
