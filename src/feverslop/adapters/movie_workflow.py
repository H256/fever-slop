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


def _links_removed_node(value: Any, removed: set[str]) -> bool:
    return isinstance(value, list) and bool(value) and str(value[0]) in removed


def _has_title(patcher: WorkflowPatcher, title: str) -> bool:
    try:
        patcher.find_node_by_meta_title(title)
        return True
    except KeyError:
        return False
