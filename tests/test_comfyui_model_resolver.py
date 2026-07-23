import json
import tempfile
import unittest
from pathlib import Path


class FakeObjectInfoClient:
    def __init__(self, object_info):
        self.object_info = object_info

    def get_object_info(self):
        return self.object_info


def object_info_for(*, values, class_type="LoraLoader", input_name="lora_name"):
    return {
        class_type: {
            "input": {
                "required": {
                    input_name: [list(values)],
                }
            }
        }
    }


def combo_object_info_for(*, values, class_type="LatentUpscaleModelLoader", input_name="model_name"):
    return {
        class_type: {
            "input": {
                "required": {
                    input_name: [
                        "COMBO",
                        {
                            "options": list(values),
                            "multiselect": False,
                        },
                    ],
                }
            }
        }
    }


def object_info_with_fields(class_type: str, fields: dict[str, list[str]]):
    return {
        class_type: {
            "input": {
                "required": {
                    input_name: [list(values)]
                    for input_name, values in fields.items()
                }
            }
        }
    }


def workflow_with(value, *, class_type="LoraLoader", node_id="12", title="#LORA_1", input_name="lora_name"):
    return {
        node_id: {
            "class_type": class_type,
            "inputs": {
                input_name: value,
                "model": ["11", 0],
            },
            "_meta": {"title": title},
        }
    }


class ComfyUIModelResolverTests(unittest.TestCase):
    def test_exact_dropdown_value_remains_unchanged(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["models/foo.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("models/foo.safetensors"),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("models/foo.safetensors", resolved["12"]["inputs"]["lora_name"])

    def test_windows_path_separator_resolves_to_server_dropdown_value(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["zimage/own/klw251209-v1_000001250.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("zimage\\own\\klw251209-v1_000001250.safetensors"),
            workflow_path=Path("workflows/image_t2i_startframe_v1.json"),
        )

        self.assertEqual(
            "zimage/own/klw251209-v1_000001250.safetensors",
            resolved["12"]["inputs"]["lora_name"],
        )

    def test_basename_resolves_when_unique(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["characters/foo.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("foo.safetensors"),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("characters/foo.safetensors", resolved["12"]["inputs"]["lora_name"])

    def test_combo_options_resolve_as_dropdown_values(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(combo_object_info_for(values=["upscale/model.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with(
                "model.safetensors",
                class_type="LatentUpscaleModelLoader",
                input_name="model_name",
            ),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("upscale/model.safetensors", resolved["12"]["inputs"]["model_name"])

    def test_non_model_dropdowns_are_ignored(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["server.png"], class_type="LoadImage", input_name="image"))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with(
                "scene_0038.png",
                class_type="LoadImage",
                input_name="image",
                title="#STARTFRAME",
            ),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("scene_0038.png", resolved["12"]["inputs"]["image"])
        self.assertEqual(0, resolver.last_report["patched_count"])

    def test_class_name_marker_does_not_make_non_model_input_a_model(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["ltxv"], class_type="DualCLIPLoaderGGUF", input_name="type"))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with(
                "custom_type_value",
                class_type="DualCLIPLoaderGGUF",
                input_name="type",
                title="DualCLIPLoader (GGUF)",
            ),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("custom_type_value", resolved["12"]["inputs"]["type"])
        self.assertEqual(0, resolver.last_report["patched_count"])

    def test_multiple_model_errors_are_reported_together(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(
                object_info_with_fields(
                    "DualCLIPLoaderGGUF",
                    {
                        "clip_name1": ["known_clip_1.safetensors"],
                        "clip_name2": ["known_clip_2.safetensors"],
                    },
                )
            )
        )

        with self.assertRaisesRegex(
            ComfyUIModelResolutionError,
            r"missing_clip_1\.safetensors.*missing_clip_2\.safetensors",
        ):
            resolver.resolve_workflow_models(
                {
                    "3": {
                        "class_type": "DualCLIPLoaderGGUF",
                        "inputs": {
                            "clip_name1": "missing_clip_1.safetensors",
                            "clip_name2": "missing_clip_2.safetensors",
                        },
                        "_meta": {"title": "DualCLIPLoader (GGUF)"},
                    }
                },
                workflow_path=Path("workflows/test.json"),
            )

    def test_basename_ambiguity_raises_clear_error(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["a/foo.safetensors", "b/foo.safetensors"]))
        )

        with self.assertRaisesRegex(
            ComfyUIModelResolutionError,
            r"Ambiguous ComfyUI model reference 'foo\.safetensors'.*Matches: a/foo\.safetensors, b/foo\.safetensors",
        ):
            resolver.resolve_workflow_models(
                workflow_with("foo.safetensors"),
                workflow_path=Path("workflows/test.json"),
            )

    def test_missing_model_raises_clear_error(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["known.safetensors"]))
        )

        with self.assertRaisesRegex(
            ComfyUIModelResolutionError,
            r"ComfyUI model reference 'missing\.safetensors'.*was not found in server dropdown values",
        ):
            resolver.resolve_workflow_models(
                workflow_with("missing.safetensors"),
                workflow_path=Path("workflows/test.json"),
            )

    def test_empty_model_dropdown_raises_missing_model(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=[], class_type="UnetLoaderGGUF", input_name="unet_name"))
        )

        with self.assertRaisesRegex(
            ComfyUIModelResolutionError,
            r"ComfyUI model reference 'LTX-2\.3-22B-distilled-1\.1-Q6_K\.gguf'.*was not found",
        ):
            resolver.resolve_workflow_models(
                workflow_with(
                    "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
                    class_type="UnetLoaderGGUF",
                    input_name="unet_name",
                    title="Unet Loader \\(GGUF\\)",
                ),
                workflow_path=Path("workflows/test.json"),
            )

    def test_missing_node_class_raises_clear_error(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["known.safetensors"]))
        )

        with self.assertRaisesRegex(
            ComfyUIModelResolutionError,
            r"Missing node types: node 12: MissingCustomNode.*node 13: AnotherMissingNode",
        ):
            resolver.resolve_workflow_models(
                {
                    **workflow_with("known.safetensors", class_type="MissingCustomNode"),
                    **workflow_with(
                        "known.safetensors",
                        class_type="AnotherMissingNode",
                        node_id="13",
                        title="#MISSING_2",
                    ),
                },
                workflow_path=Path("workflows/test.json"),
            )

    def test_missing_ltx_msr_nodes_include_setup_hint(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["known.safetensors"]))
        )

        with self.assertRaisesRegex(
            ComfyUIModelResolutionError,
            r"LTXAddVideoICLoRAGuide.*LTXICLoRALoaderModelOnly.*ComfyUI-Licon-MSR.*LTX IC-LoRA",
        ):
            resolver.resolve_workflow_models(
                {
                    **workflow_with("known.safetensors", class_type="LTXAddVideoICLoRAGuide", title="#MSR_GUIDE"),
                    **workflow_with(
                        "known.safetensors",
                        class_type="LTXICLoRALoaderModelOnly",
                        node_id="59",
                        title="#MSR_LORA",
                    ),
                },
                workflow_path=Path("workflows/video_ltxv_msr_1actor_1background_v4.json"),
            )

    def test_strict_override_applies_when_expected_value_matches(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolver,
        )
        from feverslop.config.comfyui import ComfyUIModelOverride

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["server/foo.safetensors"])),
            overrides=[
                ComfyUIModelOverride(
                    workflow="workflows/test.json",
                    node_id="12",
                    node_title="#LORA_1",
                    input="lora_name",
                    expected_value="old\\foo.safetensors",
                    replacement="server/foo.safetensors",
                )
            ],
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("old\\foo.safetensors"),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("server/foo.safetensors", resolved["12"]["inputs"]["lora_name"])

    def test_stale_override_fails_when_workflow_value_changed(self):
        from feverslop.adapters.comfyui_model_resolver import (
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )
        from feverslop.config.comfyui import ComfyUIModelOverride

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["server/foo.safetensors"])),
            overrides=[
                ComfyUIModelOverride(
                    workflow="workflows/test.json",
                    node_id="12",
                    node_title="#LORA_1",
                    input="lora_name",
                    expected_value="old\\foo.safetensors",
                    replacement="server/foo.safetensors",
                )
            ],
        )

        with self.assertRaisesRegex(ComfyUIModelResolutionError, "Stale ComfyUI model override"):
            resolver.resolve_workflow_models(
                workflow_with("new\\foo.safetensors"),
                workflow_path=Path("workflows/test.json"),
            )

    def test_validate_workflow_directory_reports_resolved_files(self):
        from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["folder/foo.safetensors"]))
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            workflows = Path(temp_dir) / "workflows"
            workflows.mkdir()
            (workflows / "sample.json").write_text(
                json.dumps(workflow_with("foo.safetensors")),
                encoding="utf-8",
            )

            reports = resolver.validate_workflow_directory(workflows)

        self.assertEqual(1, len(reports))
        self.assertEqual("sample.json", reports[0]["workflow"])
        self.assertEqual(1, reports[0]["patched_count"])


class ComfyUIConfigModelOverrideTests(unittest.TestCase):
    def test_app_config_loads_model_overrides(self):
        from feverslop.config.app_config import AppConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "comfyui": {
                            "base_url": "http://comfy.example",
                            "model_overrides": [
                                {
                                    "workflow": "workflows/test.json",
                                    "node_id": "12",
                                    "node_title": "#LORA_1",
                                    "input": "lora_name",
                                    "expected_value": "old\\foo.safetensors",
                                    "replacement": "new/foo.safetensors",
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = AppConfig.load(config_path)

        self.assertEqual("http://comfy.example", config.comfyui.base_url)
        self.assertEqual(1, len(config.comfyui.model_overrides))
        self.assertEqual("workflows/test.json", config.comfyui.model_overrides[0].workflow)

    def test_missing_app_config_defaults_to_empty_model_overrides(self):
        from feverslop.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertEqual([], config.comfyui.model_overrides)


class ComfyUIWorkflowValidationCliTests(unittest.TestCase):
    def test_validate_cli_parser_defaults_to_workflows_directory(self):
        from feverslop.tools.validate_comfyui_workflows import build_arg_parser

        args = build_arg_parser().parse_args([])

        self.assertEqual("app_config.json", args.app_config)
        self.assertEqual("workflows", args.workflows_dir)

    def test_validate_cli_run_returns_reports(self):
        from feverslop.tools.validate_comfyui_workflows import validate_comfyui_workflows

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflows = temp / "workflows"
            workflows.mkdir()
            (workflows / "sample.json").write_text(
                json.dumps(workflow_with("foo.safetensors")),
                encoding="utf-8",
            )

            reports = validate_comfyui_workflows(
                client=FakeObjectInfoClient(object_info_for(values=["folder/foo.safetensors"])),
                workflows_dir=workflows,
                overrides=[],
            )

        self.assertEqual(1, len(reports))
        self.assertEqual("sample.json", reports[0]["workflow"])
        self.assertEqual(1, reports[0]["patched_count"])

    def test_validate_cli_reports_errors_without_stopping_at_first_workflow(self):
        from feverslop.tools.validate_comfyui_workflows import validate_comfyui_workflows

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflows = temp / "workflows"
            workflows.mkdir()
            (workflows / "a.json").write_text(
                json.dumps(workflow_with("missing-a.safetensors")),
                encoding="utf-8",
            )
            (workflows / "b.json").write_text(
                json.dumps(workflow_with("missing-b.safetensors")),
                encoding="utf-8",
            )

            reports = validate_comfyui_workflows(
                client=FakeObjectInfoClient(object_info_for(values=["known.safetensors"])),
                workflows_dir=workflows,
                overrides=[],
            )

        self.assertEqual(["a.json", "b.json"], [report["workflow"] for report in reports])
        self.assertIn("missing-a.safetensors", reports[0]["errors"][0])
        self.assertIn("missing-b.safetensors", reports[1]["errors"][0])


if __name__ == "__main__":
    unittest.main()
