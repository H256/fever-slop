"""Tests for the pure latent upscaler device detection helpers."""

import unittest

from feverslop.adapters.comfyui_upscaler_device import (
    LATENT_UPSCALER_NODE_CLASS,
    detect_upscaler_device,
    device_input_candidates,
    format_vram_gib,
    primary_gpu_device,
)


def _stats(*devices) -> dict:
    return {"devices": list(devices)}


def _gpu(name="NVIDIA GeForce RTX 4090", device_type="cuda") -> dict:
    return {"name": name, "type": device_type}


class DetectUpscalerDeviceTests(unittest.TestCase):
    def test_amd_radeon_name_returns_rocm(self):
        self.assertEqual("rocm", detect_upscaler_device(_stats(_gpu("AMD Radeon RX 7900 XTX"))))

    def test_amd_instinct_name_returns_rocm(self):
        self.assertEqual("rocm", detect_upscaler_device(_stats(_gpu("AMD Instinct MI300X"))))

    def test_radeon_marker_is_case_insensitive(self):
        self.assertEqual("rocm", detect_upscaler_device(_stats(_gpu("RADEON VII"))))

    def test_nvidia_name_returns_none(self):
        self.assertIsNone(detect_upscaler_device(_stats(_gpu("NVIDIA GeForce RTX 4090"))))

    def test_cpu_only_devices_return_none(self):
        self.assertIsNone(detect_upscaler_device(_stats({"name": "AMD Ryzen 9 7950X", "type": "cpu"})))

    def test_empty_devices_list_returns_none(self):
        self.assertIsNone(detect_upscaler_device(_stats()))

    def test_missing_devices_key_returns_none(self):
        self.assertIsNone(detect_upscaler_device({}))

    def test_non_dict_stats_returns_none(self):
        self.assertIsNone(detect_upscaler_device(["not a dict"]))

    def test_non_list_devices_returns_none(self):
        self.assertIsNone(detect_upscaler_device({"devices": "gpu0"}))

    def test_non_dict_entry_returns_none(self):
        self.assertIsNone(detect_upscaler_device(_stats("AMD Radeon RX 7900 XTX")))

    def test_missing_name_returns_none(self):
        self.assertIsNone(detect_upscaler_device(_stats({"type": "cuda"})))

    def test_non_string_name_returns_none(self):
        self.assertIsNone(detect_upscaler_device(_stats({"name": 42, "type": "cuda"})))

    def test_multi_gpu_first_non_amd_wins(self):
        stats = _stats(_gpu("NVIDIA GeForce RTX 4090"), _gpu("AMD Radeon RX 7900 XTX"))
        self.assertIsNone(detect_upscaler_device(stats))

    def test_multi_gpu_first_amd_wins(self):
        stats = _stats(_gpu("AMD Radeon RX 7900 XTX"), _gpu("NVIDIA GeForce RTX 4090"))
        self.assertEqual("rocm", detect_upscaler_device(stats))

    def test_rocm_type_is_recognized(self):
        self.assertEqual("rocm", detect_upscaler_device(_stats(_gpu("AMD Radeon RX 7900 XTX", device_type="rocm"))))


class PrimaryGpuDeviceTests(unittest.TestCase):
    def test_returns_first_cuda_entry(self):
        stats = _stats({"name": "cpu", "type": "cpu"}, _gpu("NVIDIA GeForce RTX 4090"))
        self.assertEqual(_gpu("NVIDIA GeForce RTX 4090"), primary_gpu_device(stats))

    def test_skips_cpu_entries_even_when_named_amd(self):
        stats = _stats({"name": "AMD Ryzen 9 7950X", "type": "cpu"}, _gpu("NVIDIA GeForce RTX 4090"))
        self.assertEqual("NVIDIA GeForce RTX 4090", primary_gpu_device(stats)["name"])

    def test_type_matching_is_case_insensitive(self):
        self.assertEqual("NVIDIA GeForce RTX 4090", primary_gpu_device(_stats(_gpu(device_type="CUDA")))["name"])

    def test_malformed_payloads_return_none(self):
        for stats in (None, [], "x", {"devices": "x"}, {"devices": [1, "x"]}):
            with self.subTest(stats=stats):
                self.assertIsNone(primary_gpu_device(stats))


class DeviceInputCandidatesTests(unittest.TestCase):
    def _payload(self, device_descriptor, section="required") -> dict:
        return {
            LATENT_UPSCALER_NODE_CLASS: {
                "input": {section: {"device": device_descriptor}}
            }
        }

    def test_required_wrapped_list_shape(self):
        # Real ComfyUI /object_info wraps a string-list in a list: [[...]].
        self.assertEqual(
            ["cuda", "rocm", "cpu"],
            device_input_candidates(self._payload([["cuda", "rocm", "cpu"]])),
        )

    def test_optional_section(self):
        payload = self._payload([["cuda", "cpu"]], section="optional")
        self.assertEqual(["cuda", "cpu"], device_input_candidates(payload))

    def test_required_section_wins_over_optional(self):
        payload = {
            LATENT_UPSCALER_NODE_CLASS: {
                "input": {
                    "required": {"device": [["cuda"]]},
                    "optional": {"device": [["cpu"]]},
                }
            }
        }
        self.assertEqual(["cuda"], device_input_candidates(payload))

    def test_combo_shape(self):
        payload = self._payload(["COMBO", {"options": ["cuda", "rocm"]}])
        self.assertEqual(["cuda", "rocm"], device_input_candidates(payload))

    def test_empty_wrapped_option_list_returns_empty_list(self):
        self.assertEqual([], device_input_candidates(self._payload([[]])))

    def test_node_class_missing_returns_none(self):
        self.assertIsNone(device_input_candidates({"OtherNode": {}}))

    def test_node_class_entry_not_a_dict_returns_none(self):
        self.assertIsNone(device_input_candidates({LATENT_UPSCALER_NODE_CLASS: "x"}))

    def test_input_spec_missing_returns_none(self):
        self.assertIsNone(device_input_candidates({LATENT_UPSCALER_NODE_CLASS: {}}))

    def test_unknown_shape_returns_none(self):
        for descriptor in (
            "cuda",
            {"device": ["cuda"]},
            [],
            ["cuda", "rocm"],       # bare flat list is not a ComfyUI shape
            ["COMBO", "x"],
            ["COMBO", {"options": "x"}],
            [[1, 2]],
        ):
            with self.subTest(descriptor=descriptor):
                self.assertIsNone(device_input_candidates(self._payload(descriptor)))

    def test_payload_none_or_malformed_returns_none(self):
        for payload in (None, [], "x"):
            with self.subTest(payload=payload):
                self.assertIsNone(device_input_candidates(payload))


class FormatVramGibTests(unittest.TestCase):
    def test_bytes_to_gib(self):
        self.assertEqual("24.0 GiB", format_vram_gib(25769803776))

    def test_rounding_to_one_decimal(self):
        self.assertEqual("21.3 GiB", format_vram_gib(22886172672))

    def test_zero(self):
        self.assertEqual("0.0 GiB", format_vram_gib(0))

    def test_int_value(self):
        self.assertEqual("24.0 GiB", format_vram_gib(25769803776))

    def test_non_numeric_returns_none(self):
        for value in (None, "24", ["24"], {"gb": 24}, object()):
            with self.subTest(value=value):
                self.assertIsNone(format_vram_gib(value))

    def test_negative_returns_none(self):
        self.assertIsNone(format_vram_gib(-1))

    def test_non_finite_returns_none(self):
        self.assertIsNone(format_vram_gib(float("inf")))
        self.assertIsNone(format_vram_gib(float("-inf")))
        self.assertIsNone(format_vram_gib(float("nan")))


if __name__ == "__main__":
    unittest.main()
