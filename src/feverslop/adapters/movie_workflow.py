from __future__ import annotations

from typing import Any

from feverslop.adapters.workflow_patcher import WorkflowPatcher


class MovieWorkflowPatcher:
    AUDIO_CLASSES = {"LoadAudio", "VHS_LoadAudio", "AudioLoader", "TrimAudio", "TrimAudioDuration"}
    AUDIO_TITLES = {"#LOAD_AUDIO", "#TRIM_AUDIO"}

    def strip_audio_inputs(self, workflow: dict[str, Any]) -> dict[str, Any]:
        patcher = WorkflowPatcher(workflow)
        patched = patcher.get()
        removed = {
            node_id
            for node_id, node in patched.items()
            if node.get("class_type") in self.AUDIO_CLASSES or node.get("_meta", {}).get("title") in self.AUDIO_TITLES
        }
        for node_id in removed:
            del patched[node_id]
        for node in patched.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for key, value in list(inputs.items()):
                if _links_removed_node(value, removed):
                    del inputs[key]
        return patched


def _links_removed_node(value: Any, removed: set[str]) -> bool:
    return isinstance(value, list) and bool(value) and str(value[0]) in removed
