from __future__ import annotations

from typing import Any

from feverslop.adapters.workflow_patcher import (
    WorkflowPatcher,
    _workflow_dependency_ids,
)


class MovieWorkflowPatcher:
    AUDIO_SOURCE_CLASSES = {
        "LoadAudio",
        "VHS_LoadAudio",
        "AudioLoader",
        "TrimAudio",
        "TrimAudioDuration",
    }
    AUDIO_LATENT_CLASSES = {
        "LTXVAudioVAEEncode",
        "SetLatentNoiseMask",
        "SolidMask",
    }
    AUDIO_VIDEO_CONCAT_CLASSES = {"LTXVConcatAVLatent"}
    AUDIO_TITLES = {"#LOAD_AUDIO", "#TRIM_AUDIO"}

    def strip_audio_inputs(self, workflow: dict[str, Any]) -> dict[str, Any]:
        patcher = WorkflowPatcher(workflow)
        patched = patcher.get()
        removed: set[str] = set()
        for node_id, node in list(patched.items()):
            if node.get("class_type") not in self.AUDIO_VIDEO_CONCAT_CLASSES:
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            audio_latent = inputs.get("audio_latent")
            if not _is_link(audio_latent):
                continue
            if self._is_empty_audio_latent(patched, audio_latent):
                self._repair_empty_audio_latent(patched, audio_latent)
                continue
            replacement = self._build_empty_audio_latent(
                patched,
                audio_latent,
                frames_link=self._find_input_by_title(patched, "#FRAMES"),
                framerate_link=self._find_input_by_title(patched, "#FRAMERATE"),
            )
            if replacement is None:
                continue
            replacement_id, upstream_removed = replacement
            inputs["audio_latent"] = [replacement_id, 0]
            removed.update(upstream_removed)
        removed.update(
            node_id
            for node_id, node in patched.items()
            if node.get("class_type") in self.AUDIO_SOURCE_CLASSES
            or node.get("_meta", {}).get("title") in self.AUDIO_TITLES
        )
        for node_id in removed:
            patched.pop(node_id, None)
        for node in patched.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for key, value in list(inputs.items()):
                cleaned = _without_removed_links(value, removed)
                if cleaned is None:
                    del inputs[key]
                elif cleaned is not value:
                    inputs[key] = cleaned
        return patched

    def _is_empty_audio_latent(self, workflow: dict[str, Any], value: Any) -> bool:
        node = workflow.get(str(value[0])) if _is_link(value) else None
        return isinstance(node, dict) and node.get("class_type") == "LTXVEmptyLatentAudio"

    def _repair_empty_audio_latent(self, workflow: dict[str, Any], value: Any) -> None:
        node = workflow.get(str(value[0])) if _is_link(value) else None
        if not isinstance(node, dict):
            return
        audio_vae = self._find_audio_vae_link(workflow)
        if audio_vae is not None:
            node.setdefault("inputs", {})["audio_vae"] = audio_vae

    def _build_empty_audio_latent(
        self,
        workflow: dict[str, Any],
        audio_latent: Any,
        *,
        frames_link: list[Any] | None,
        framerate_link: list[Any] | None,
    ) -> tuple[str, set[str]] | None:
        if frames_link is None or framerate_link is None:
            return None
        trace = self._trace_audio_latent_source(workflow, audio_latent)
        if trace is None:
            return None
        audio_vae, removed = trace
        node_id = _next_node_id(workflow)
        workflow[node_id] = {
            "inputs": {
                "frames_number": frames_link,
                "frame_rate": framerate_link,
                "batch_size": 1,
                "audio_vae": audio_vae,
            },
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "LTXV Empty Latent Audio"},
        }
        return node_id, removed

    def _trace_audio_latent_source(self, workflow: dict[str, Any], value: Any) -> tuple[Any, set[str]] | None:
        if not _is_link(value):
            return None
        node_id = str(value[0])
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            return None
        class_type = node.get("class_type")
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        if class_type == "LTXVAudioVAEEncode":
            audio_vae = inputs.get("audio_vae")
            if not _is_link(audio_vae):
                return None
            removed = {node_id}
            removed.update(self._collect_audio_source_nodes(workflow, inputs.get("audio")))
            return audio_vae, removed
        if class_type == "SetLatentNoiseMask":
            trace = self._trace_audio_latent_source(workflow, inputs.get("samples"))
            if trace is None:
                return None
            audio_vae, removed = trace
            removed.add(node_id)
            if _is_link(inputs.get("mask")):
                removed.add(str(inputs["mask"][0]))
            return audio_vae, removed
        return None

    def _collect_audio_source_nodes(self, workflow: dict[str, Any], value: Any) -> set[str]:
        if not _is_link(value):
            return set()
        node_id = str(value[0])
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            return set()
        class_type = node.get("class_type")
        title = node.get("_meta", {}).get("title")
        if class_type not in self.AUDIO_SOURCE_CLASSES and title not in self.AUDIO_TITLES:
            return set()
        removed = {node_id}
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        for input_value in inputs.values():
            removed.update(self._collect_audio_source_nodes(workflow, input_value))
        return removed

    def _find_input_by_title(self, workflow: dict[str, Any], title: str) -> list[Any] | None:
        for node_id, node in workflow.items():
            if node.get("_meta", {}).get("title") == title:
                return [node_id, 0]
        return None

    def _find_audio_vae_link(self, workflow: dict[str, Any]) -> list[Any] | None:
        for node in workflow.values():
            if node.get("class_type") == "LTXVAudioVAEDecode":
                audio_vae = (node.get("inputs") or {}).get("audio_vae")
                if _is_link(audio_vae):
                    return list(audio_vae)
        for node_id, node in workflow.items():
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            vae_name = str(inputs.get("vae_name") or "").lower()
            if node.get("class_type") == "VAELoaderKJ" and "audio" in vae_name:
                return [node_id, 0]
        return None

    def patch_msr_i2v_startframe(
        self,
        workflow: dict[str, Any],
        *,
        startframe_image_name: str,
        msr_lora_name: str | None = None,
        msr_frame_count: int | None = None,
    ) -> dict[str, Any]:
        patcher = WorkflowPatcher(workflow)
        missing = [title for title in ("#MSR_ACTOR_1", "#MSR_BACKGROUND", "#MSR_LORA", "#MSR_FRAME_COUNT", "#PROMPT_RELAY", "#STARTFRAME") if not _has_title(patcher, title)]
        if missing:
            raise ValueError(f"MSR-I2V workflow is missing required anchor(s): {', '.join(missing)}")
        patcher.set_input_by_title("#STARTFRAME", "image", startframe_image_name)
        if msr_lora_name:
            patcher.try_set_existing_input_by_title("#MSR_LORA", "lora_name", msr_lora_name)
        if msr_frame_count is not None:
            patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "frame_count", int(msr_frame_count))
            patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "value", int(msr_frame_count))
        return patcher.get()


def _without_removed_links(value: Any, removed: set[str]) -> Any | None:
    """Return value without wires to removed nodes; None means delete the input."""
    if not isinstance(value, list):
        return value
    if bool(value) and str(value[0]) in removed:
        return None
    kept: list[Any] = []
    for item in value:
        if _workflow_dependency_ids(item) & removed:
            continue
        kept.append(item)
    if kept == value:
        return value
    return kept or None


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2


def _next_node_id(workflow: dict[str, Any]) -> str:
    numeric_ids = [int(node_id) for node_id in workflow if str(node_id).isdigit()]
    return str((max(numeric_ids) if numeric_ids else 0) + 1)


def _has_title(patcher: WorkflowPatcher, title: str) -> bool:
    try:
        patcher.find_node_by_meta_title(title)
        return True
    except KeyError:
        return False
