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
        from autoprompter.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["models/foo.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("models/foo.safetensors"),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("models/foo.safetensors", resolved["12"]["inputs"]["lora_name"])

    def test_windows_path_separator_resolves_to_server_dropdown_value(self):
        from autoprompter.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["zimage/own/klw251209-v1_000001250.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("zimage\\own\\klw251209-v1_000001250.safetensors"),
            workflow_path=Path("workflows/autoprompt_image_z_image_turbo.json"),
        )

        self.assertEqual(
            "zimage/own/klw251209-v1_000001250.safetensors",
            resolved["12"]["inputs"]["lora_name"],
        )

    def test_basename_resolves_when_unique(self):
        from autoprompter.adapters.comfyui_model_resolver import ComfyUIModelResolver

        resolver = ComfyUIModelResolver(
            FakeObjectInfoClient(object_info_for(values=["characters/foo.safetensors"]))
        )

        resolved = resolver.resolve_workflow_models(
            workflow_with("foo.safetensors"),
            workflow_path=Path("workflows/test.json"),
        )

        self.assertEqual("characters/foo.safetensors", resolved["12"]["inputs"]["lora_name"])

    def test_basename_ambiguity_raises_clear_error(self):
        from autoprompter.adapters.comfyui_model_resolver import (
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
        from autoprompter.adapters.comfyui_model_resolver import (
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

    def test_strict_override_applies_when_expected_value_matches(self):
        from autoprompter.adapters.comfyui_model_resolver import (
            ComfyUIModelOverride,
            ComfyUIModelResolver,
        )

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
        from autoprompter.adapters.comfyui_model_resolver import (
            ComfyUIModelOverride,
            ComfyUIModelResolutionError,
            ComfyUIModelResolver,
        )

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
        from autoprompter.adapters.comfyui_model_resolver import ComfyUIModelResolver

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
        from autoprompter.config.app_config import AppConfig

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
        from autoprompter.config.app_config import AppConfig

        config = AppConfig.load(Path("does-not-exist.json"))

        self.assertEqual([], config.comfyui.model_overrides)


class ComfyUIWorkflowValidationCliTests(unittest.TestCase):
    def test_validate_cli_parser_defaults_to_workflows_directory(self):
        from autoprompter.tools.validate_comfyui_workflows import build_arg_parser

        args = build_arg_parser().parse_args([])

        self.assertEqual("app_config.json", args.app_config)
        self.assertEqual("workflows", args.workflows_dir)

    def test_validate_cli_run_returns_reports(self):
        from autoprompter.tools.validate_comfyui_workflows import validate_comfyui_workflows

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


if __name__ == "__main__":
    unittest.main()
