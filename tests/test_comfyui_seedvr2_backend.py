import unittest

from feverslop.adapters.comfyui_seedvr2_backend import (
    ComfyUISeedVR2Backend,
    SeedVR2RenderSettings,
)


class ComfyUISeedVR2BackendTests(unittest.TestCase):
    @staticmethod
    def node(workflow, title):
        return next(node for node in workflow.values() if node.get("_meta", {}).get("title") == title)

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

        encode = self.node(workflow, "#VAE_ENCODE_TILED")
        decode = self.node(workflow, "#VAE_DECODE_TILED")
        self.assertEqual(512, encode["inputs"]["tile_size"])
        self.assertEqual(128, encode["inputs"]["overlap"])
        self.assertEqual(64, encode["inputs"]["temporal_size"])
        self.assertEqual(8, encode["inputs"]["temporal_overlap"])
        self.assertEqual(encode["inputs"]["tile_size"], decode["inputs"]["tile_size"])
        self.assertEqual(encode["inputs"]["overlap"], decode["inputs"]["overlap"])

        resize = self.node(workflow, "#RESIZE_VIDEO")
        self.assertEqual("scale by multiplier", resize["inputs"]["resize_type"])
        self.assertEqual(2.0, resize["inputs"]["resize_type.multiplier"])

        self.assertEqual(["66:57", 0], self.node(workflow, "#SEEDVR_PREPROCESS")["inputs"]["resized_images"])
        self.assertEqual(["66:74", 0], resize["inputs"]["input"])
        self.assertEqual(["66:74", 1], self.node(workflow, "#CREATE_VIDEO")["inputs"]["audio"])
        self.assertEqual(["66:74", 2], self.node(workflow, "#CREATE_VIDEO")["inputs"]["fps"])
        self.assertTrue(self.node(workflow, "#SPLIT_LATENT_BOOLEAN")["inputs"]["value"])

    def test_build_workflow_uses_patch_anchors_and_seedvr2_defaults(self):
        backend = ComfyUISeedVR2Backend(client=object())

        workflow = backend.build_workflow(
            video_name="feverslop/input/scene_0001.mp4",
            output_prefix="feverslop/seedvr2/scene_0001/pass_01",
            output_size=(1920, 1080),
            settings=SeedVR2RenderSettings(),
        )

        self.assertEqual("feverslop/input/scene_0001.mp4", self.node(workflow, "#LOAD_VIDEO")["inputs"]["file"])
        self.assertEqual(2.0, self.node(workflow, "#RESIZE_VIDEO")["inputs"]["resize_type.multiplier"])
        self.assertEqual("seedvr2_3b_int8_convrot.safetensors", self.node(workflow, "#SEEDVR_MODEL")["inputs"]["unet_name"])
        self.assertEqual(0.35, self.node(workflow, "#SEEDVR_SAMPLER")["inputs"]["denoise"])
        self.assertEqual(4, self.node(workflow, "#TEMPORAL_CHUNK")["inputs"]["temporal_overlap"])
        self.assertEqual("feverslop/seedvr2/scene_0001/pass_01", self.node(workflow, "#SAVE_VIDEO")["inputs"]["filename_prefix"])

    def test_build_workflow_rejects_unsupported_color_correction(self):
        backend = ComfyUISeedVR2Backend(client=object())

        with self.assertRaisesRegex(ValueError, "color_correction"):
            backend.build_workflow(
                video_name="input.mp4",
                output_prefix="output",
                output_size=(1920, 1080),
                settings=SeedVR2RenderSettings(color_correction="invalid"),
            )

    def test_build_workflow_activates_trim_for_segment(self):
        backend = ComfyUISeedVR2Backend(client=object())

        workflow = backend.build_workflow(
            video_name="input.mp4",
            output_prefix="output/segment",
            output_size=(1920, 1088),
            settings=SeedVR2RenderSettings(
                trim_start_seconds=4.0,
                trim_duration_seconds=3.08,
            ),
        )

        source_switch = self.node(workflow, "#VIDEO_SOURCE")
        trim = self.node(workflow, "#VIDEO_SLICE")
        self.assertTrue(source_switch["inputs"]["switch"])
        self.assertEqual(4.0, trim["inputs"]["start_time"])
        self.assertEqual(3.08, trim["inputs"]["duration"])
