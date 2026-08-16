import unittest

from feverslop.adapters.comfyui_seedvr2_backend import (
    ComfyUISeedVR2Backend,
    SeedVR2RenderSettings,
)


class ComfyUISeedVR2BackendTests(unittest.TestCase):
    def test_build_workflow_uses_memory_safe_video_template(self):
        backend = ComfyUISeedVR2Backend(client=object())

        workflow = backend.build_workflow(
            video_name="feverslop/input/scene_0001.mp4",
            output_prefix="feverslop/seedvr2/scene_0001/pass_01",
            output_size=(1920, 1080),
            settings=SeedVR2RenderSettings(),
        )

        class_types = {node["class_type"] for node in workflow.values()}
        self.assertIn("VAEEncodeTiled", class_types)
        self.assertIn("VAEDecodeTiled", class_types)
        self.assertIn("ResizeImageMaskNode", class_types)
        self.assertIn("GetVideoComponents", class_types)
        self.assertIn("CreateVideo", class_types)
        self.assertIn("SaveVideo", class_types)
        self.assertNotIn("VHS_LoadVideo", class_types)
        self.assertNotIn("VHS_VideoCombine", class_types)

        encode = next(node for node in workflow.values() if node["class_type"] == "VAEEncodeTiled")
        decode = next(node for node in workflow.values() if node["class_type"] == "VAEDecodeTiled")
        self.assertEqual(512, encode["inputs"]["tile_size"])
        self.assertEqual(128, encode["inputs"]["overlap"])
        self.assertEqual(64, encode["inputs"]["temporal_size"])
        self.assertEqual(8, encode["inputs"]["temporal_overlap"])
        self.assertEqual(encode["inputs"]["tile_size"], decode["inputs"]["tile_size"])
        self.assertEqual(encode["inputs"]["overlap"], decode["inputs"]["overlap"])

        resize = next(node for node in workflow.values() if node["class_type"] == "ResizeImageMaskNode")
        self.assertEqual("scale dimensions", resize["inputs"]["resize_type"])
        self.assertEqual(1920, resize["inputs"]["resize_type.width"])
        self.assertEqual(1080, resize["inputs"]["resize_type.height"])

        self.assertEqual(["3", 0], workflow["4"]["inputs"]["resized_images"])
        self.assertEqual(["2", 0], workflow["3"]["inputs"]["input"])
        self.assertEqual(["2", 1], workflow["16"]["inputs"]["audio"])
        self.assertEqual(["2", 2], workflow["16"]["inputs"]["fps"])
        self.assertTrue(workflow["18"]["inputs"]["value"])

    def test_build_workflow_uses_patch_anchors_and_seedvr2_defaults(self):
        backend = ComfyUISeedVR2Backend(client=object())

        workflow = backend.build_workflow(
            video_name="feverslop/input/scene_0001.mp4",
            output_prefix="feverslop/seedvr2/scene_0001/pass_01",
            output_size=(1920, 1080),
            settings=SeedVR2RenderSettings(),
        )

        self.assertEqual("feverslop/input/scene_0001.mp4", workflow["1"]["inputs"]["file"])
        self.assertEqual(1920, workflow["3"]["inputs"]["resize_type.width"])
        self.assertEqual(1080, workflow["3"]["inputs"]["resize_type.height"])
        self.assertEqual("seedvr2_3b_int8_convrot.safetensors", workflow["8"]["inputs"]["unet_name"])
        self.assertEqual(0.35, workflow["11"]["inputs"]["denoise"])
        self.assertEqual(4, workflow["7"]["inputs"]["temporal_overlap"])
        self.assertEqual("feverslop/seedvr2/scene_0001/pass_01", workflow["17"]["inputs"]["filename_prefix"])

    def test_build_workflow_rejects_unsupported_color_correction(self):
        backend = ComfyUISeedVR2Backend(client=object())

        with self.assertRaisesRegex(ValueError, "color_correction"):
            backend.build_workflow(
                video_name="input.mp4",
                output_prefix="output",
                output_size=(1920, 1080),
                settings=SeedVR2RenderSettings(color_correction="invalid"),
            )
