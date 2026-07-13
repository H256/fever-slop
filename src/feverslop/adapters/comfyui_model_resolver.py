from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
import json
from typing import Any

from feverslop.config.comfyui import ComfyUIModelOverride
from feverslop.errors import FeverSlopAdaptationError


class ComfyUIModelResolutionError(FeverSlopAdaptationError):
    pass


class NoOpComfyUIModelResolver:
    def resolve_workflow_models(self, workflow: dict, workflow_path: str | PurePosixPath | None = None) -> dict:
        return workflow

    def validate_workflow_directory(self, workflows_dir: str | PurePosixPath) -> list[dict[str, Any]]:
        return []


class ComfyUIModelResolver:
    missing_node_hints = {
        "LiconMSR": "Install ComfyUI-Licon-MSR and restart ComfyUI.",
        "LTXAddVideoICLoRAGuide": "Install ComfyUI-Licon-MSR with LTX IC-LoRA support and restart ComfyUI.",
        "LTXICLoRALoaderModelOnly": (
            "Install ComfyUI-Licon-MSR with the LTX IC-LoRA nodes, make the LTX-2.3 Licon MSR LoRA "
            "available to ComfyUI, and restart ComfyUI."
        ),
    }

    model_input_markers = (
        "model",
        "ckpt",
        "checkpoint",
        "lora",
        "vae",
        "clip",
        "unet",
        "control",
        "upscale",
        "embedding",
        "hypernetwork",
        "gligen",
        "ipadapter",
        "diffusion",
        "diffusers",
    )

    def __init__(
        self,
        client,
        overrides: list[ComfyUIModelOverride] | None = None,
    ):
        self.client = client
        self.overrides = list(overrides or [])
        self._object_info: dict[str, Any] | None = None
        self.last_report: dict[str, Any] = {"patched": []}

    def resolve_workflow_models(self, workflow: dict, workflow_path: str | PurePosixPath | None = None) -> dict:
        resolved = deepcopy(workflow)
        path_label = self._workflow_label(workflow_path)
        patched: list[dict[str, str]] = []

        self._validate_node_class_types(resolved, path_label)

        for override in self._matching_overrides(path_label):
            self._apply_override(resolved, override, path_label, patched)

        dropdowns = self._model_dropdowns()
        errors: list[str] = []
        for node_id, node in resolved.items():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})
            if not isinstance(class_type, str) or not isinstance(inputs, dict):
                continue

            for input_name, candidates in dropdowns.get(class_type, {}).items():
                value = inputs.get(input_name)
                if not isinstance(value, str):
                    continue
                try:
                    replacement = self._resolve_value(
                        value=value,
                        candidates=candidates,
                        workflow_path=path_label,
                        node_id=str(node_id),
                        class_type=class_type,
                        input_name=input_name,
                    )
                except ComfyUIModelResolutionError as exc:
                    errors.append(str(exc))
                    continue
                if replacement != value:
                    inputs[input_name] = replacement
                    patched.append({
                        "node_id": str(node_id),
                        "class_type": class_type,
                        "input": input_name,
                        "from": value,
                        "to": replacement,
                    })

        if errors:
            raise ComfyUIModelResolutionError("; ".join(errors))

        self.last_report = {"workflow": path_label, "patched": patched, "patched_count": len(patched)}
        return resolved

    def validate_workflow_directory(self, workflows_dir: str | PurePosixPath) -> list[dict[str, Any]]:
        directory = PurePosixPath(str(workflows_dir))
        reports = []
        from pathlib import Path

        for workflow_path in sorted(Path(str(workflows_dir)).glob("*.json")):
            workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
            try:
                self.resolve_workflow_models(workflow, workflow_path=workflow_path)
                reports.append({
                    "workflow": workflow_path.name,
                    "patched_count": self.last_report["patched_count"],
                    "patched": list(self.last_report["patched"]),
                    "errors": [],
                    "directory": directory.as_posix(),
                })
            except ComfyUIModelResolutionError as exc:
                reports.append({
                    "workflow": workflow_path.name,
                    "patched_count": 0,
                    "patched": [],
                    "errors": self._split_error_messages(str(exc)),
                    "directory": directory.as_posix(),
                })
        return reports

    def _object_info_payload(self) -> dict[str, Any]:
        if self._object_info is None:
            self._object_info = self.client.get_object_info()
        return self._object_info

    def _validate_node_class_types(self, workflow: dict, workflow_path: str) -> None:
        server_class_types = set(self._object_info_payload())
        missing: list[str] = []
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            if not isinstance(class_type, str) or class_type in server_class_types:
                continue
            node_title = str(node.get("_meta", {}).get("title", ""))
            title_suffix = f" titled '{node_title}'" if node_title else ""
            missing.append(f"node {node_id}: {class_type}{title_suffix}")

        if missing:
            hints = self._missing_node_hints(missing)
            hint_suffix = f" Required setup: {' '.join(hints)}" if hints else ""
            raise ComfyUIModelResolutionError(
                f"ComfyUI workflow {workflow_path} uses node types that are not available on the "
                f"configured server. Install the required custom nodes or use a ComfyUI server/workflow "
                f"with matching custom-node support. Missing node types: {'; '.join(missing)}.{hint_suffix}"
            )

    def _model_dropdowns(self) -> dict[str, dict[str, list[str]]]:
        dropdowns: dict[str, dict[str, list[str]]] = {}
        for class_type, spec in self._object_info_payload().items():
            if not isinstance(spec, dict):
                continue
            class_dropdowns: dict[str, list[str]] = {}
            input_spec = spec.get("input", {})
            if not isinstance(input_spec, dict):
                continue
            for section in ("required", "optional"):
                fields = input_spec.get(section, {})
                if not isinstance(fields, dict):
                    continue
                for input_name, descriptor in fields.items():
                    candidates = self._dropdown_values(descriptor)
                    if candidates is not None and self._is_model_dropdown(str(class_type), str(input_name)):
                        class_dropdowns[str(input_name)] = candidates
            if class_dropdowns:
                dropdowns[str(class_type)] = class_dropdowns
        return dropdowns

    @classmethod
    def _is_model_dropdown(cls, class_type: str, input_name: str) -> bool:
        haystack = input_name.replace("-", "_").casefold()
        return any(marker in haystack for marker in cls.model_input_markers)

    @staticmethod
    def _dropdown_values(descriptor: Any) -> list[str] | None:
        if (
            isinstance(descriptor, list)
            and descriptor
            and isinstance(descriptor[0], list)
            and all(isinstance(item, str) for item in descriptor[0])
        ):
            return list(descriptor[0])
        if (
            isinstance(descriptor, list)
            and len(descriptor) >= 2
            and descriptor[0] == "COMBO"
            and isinstance(descriptor[1], dict)
            and isinstance(descriptor[1].get("options"), list)
            and all(isinstance(item, str) for item in descriptor[1]["options"])
        ):
            return list(descriptor[1]["options"])
        return None

    def _resolve_value(
        self,
        *,
        value: str,
        candidates: list[str],
        workflow_path: str,
        node_id: str,
        class_type: str,
        input_name: str,
    ) -> str:
        if value in candidates:
            return value

        normalized = self._normalize_model_ref(value)
        normalized_matches = [
            candidate
            for candidate in candidates
            if self._normalize_model_ref(candidate).casefold() == normalized.casefold()
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0]
        if len(normalized_matches) > 1:
            raise self._ambiguous_error(value, workflow_path, node_id, class_type, input_name, normalized_matches)

        basename = self._basename(normalized)
        basename_matches = [
            candidate
            for candidate in candidates
            if self._basename(self._normalize_model_ref(candidate)).casefold() == basename.casefold()
        ]
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise self._ambiguous_error(value, workflow_path, node_id, class_type, input_name, basename_matches)

        raise ComfyUIModelResolutionError(
            f"ComfyUI model reference '{value}' for {class_type}.{input_name} in "
            f"{workflow_path} node {node_id} was not found in server dropdown values."
        )

    def _apply_override(
        self,
        workflow: dict,
        override: ComfyUIModelOverride,
        workflow_path: str,
        patched: list[dict[str, str]],
    ) -> None:
        node = workflow.get(str(override.node_id))
        if not isinstance(node, dict):
            raise ComfyUIModelResolutionError(
                f"Stale ComfyUI model override for {override.workflow}: node {override.node_id} was not found."
            )
        node_title = str(node.get("_meta", {}).get("title", ""))
        if node_title != override.node_title:
            raise ComfyUIModelResolutionError(
                f"Stale ComfyUI model override for {override.workflow}: node {override.node_id} title is "
                f"'{node_title}', expected '{override.node_title}'."
            )
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict) or override.input not in inputs:
            raise ComfyUIModelResolutionError(
                f"Stale ComfyUI model override for {override.workflow}: input '{override.input}' was not found."
            )
        current = inputs[override.input]
        if current != override.expected_value:
            raise ComfyUIModelResolutionError(
                f"Stale ComfyUI model override for {override.workflow}: node {override.node_id} input "
                f"'{override.input}' is '{current}', expected '{override.expected_value}'."
            )
        inputs[override.input] = override.replacement
        patched.append({
            "node_id": override.node_id,
            "class_type": str(node.get("class_type", "")),
            "input": override.input,
            "from": override.expected_value,
            "to": override.replacement,
            "source": "override",
            "workflow": workflow_path,
        })

    def _matching_overrides(self, workflow_path: str) -> list[ComfyUIModelOverride]:
        normalized_path = self._normalize_workflow_path(workflow_path)
        return [
            override
            for override in self.overrides
            if self._workflow_matches(normalized_path, self._normalize_workflow_path(override.workflow))
        ]

    @staticmethod
    def _workflow_matches(workflow_path: str, override_path: str) -> bool:
        return workflow_path == override_path or workflow_path.endswith(f"/{override_path}")

    @staticmethod
    def _workflow_label(workflow_path: str | PurePosixPath | None) -> str:
        return ComfyUIModelResolver._normalize_workflow_path(str(workflow_path or "<workflow>"))

    @staticmethod
    def _normalize_workflow_path(value: str) -> str:
        return value.replace("\\", "/").casefold()

    @staticmethod
    def _normalize_model_ref(value: str) -> str:
        return value.replace("\\", "/")

    @staticmethod
    def _basename(value: str) -> str:
        return value.rstrip("/").split("/")[-1]

    @classmethod
    def _missing_node_hints(cls, missing: list[str]) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()
        for item in missing:
            class_type = item.split(":", 1)[-1].strip().split(" ", 1)[0]
            hint = cls.missing_node_hints.get(class_type)
            if hint and hint not in seen:
                hints.append(hint)
                seen.add(hint)
        return hints

    @staticmethod
    def _split_error_messages(message: str) -> list[str]:
        return [part.strip() for part in message.split("; ") if part.strip()]

    @staticmethod
    def _ambiguous_error(
        value: str,
        workflow_path: str,
        node_id: str,
        class_type: str,
        input_name: str,
        matches: list[str],
    ) -> ComfyUIModelResolutionError:
        return ComfyUIModelResolutionError(
            f"Ambiguous ComfyUI model reference '{value}' for {class_type}.{input_name} in "
            f"{workflow_path} node {node_id}. Matches: {', '.join(matches)}"
        )
