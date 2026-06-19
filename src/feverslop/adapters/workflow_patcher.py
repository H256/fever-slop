from __future__ import annotations

from copy import deepcopy
from typing import Any


class WorkflowPatcher:
    """
    Patches ComfyUI API workflow JSON.

    Supports addressing nodes by:
    - node id
    - _meta.title
    - class_type

    Recommended workflow convention:
    Give dynamic nodes explicit _meta.title values, e.g.
    #POSITIVE_PROMPT
    #NEGATIVE_PROMPT
    #SAVE_IMAGE
    #LORA_1
    """

    def __init__(self, workflow: dict):
        self.workflow = deepcopy(workflow)

    def get(self) -> dict:
        return self.workflow

    def find_node_by_id(self, node_id: str | int) -> dict:
        key = str(node_id)

        if key not in self.workflow:
            raise KeyError(f"Node id not found: {key}")

        return self.workflow[key]

    def find_nodes_by_class_type(self, class_type: str) -> list[tuple[str, dict]]:
        return [
            (node_id, node)
            for node_id, node in self.workflow.items()
            if node.get("class_type") == class_type
        ]

    def find_node_by_meta_title(self, title: str) -> tuple[str, dict]:
        for node_id, node in self.workflow.items():
            meta = node.get("_meta", {})
            if meta.get("title") == title:
                return node_id, node

        raise KeyError(f"Node with _meta.title not found: {title}")

    def set_input_by_id(
        self,
        node_id: str | int,
        input_name: str,
        value: Any,
    ) -> "WorkflowPatcher":
        node = self.find_node_by_id(node_id)
        node.setdefault("inputs", {})[input_name] = value
        return self

    def set_input_by_title(
        self,
        title: str,
        input_name: str,
        value: Any,
    ) -> "WorkflowPatcher":
        _, node = self.find_node_by_meta_title(title)
        node.setdefault("inputs", {})[input_name] = value
        return self

    def set_existing_input_by_title(
        self,
        title: str,
        input_name: str,
        value: Any,
    ) -> "WorkflowPatcher":
        _, node = self.find_node_by_meta_title(title)
        inputs = node.setdefault("inputs", {})

        if input_name not in inputs:
            raise KeyError(f"Input '{input_name}' not found on node '{title}'")

        inputs[input_name] = value
        return self

    def try_set_existing_input_by_title(
        self,
        title: str,
        input_name: str,
        value: Any,
    ) -> bool:
        try:
            self.set_existing_input_by_title(title, input_name, value)
            return True
        except KeyError:
            return False

    def set_input_by_class_type(
        self,
        class_type: str,
        input_name: str,
        value: Any,
        index: int = 0,
    ) -> "WorkflowPatcher":
        nodes = self.find_nodes_by_class_type(class_type)

        if not nodes:
            raise KeyError(f"No node with class_type found: {class_type}")

        _, node = nodes[index]
        node.setdefault("inputs", {})[input_name] = value
        return self

    def patch_lora_strength_by_title(
        self,
        title: str,
        strength: float,
        input_names: tuple[str, ...] = (
            "strength_model",
            "strength_clip",
            "strength",
            "model_strength",
            "clip_strength",
        ),
    ) -> list[str]:
        patched = []

        for input_name in input_names:
            if self.try_set_existing_input_by_title(title, input_name, strength):
                patched.append(input_name)

        if not patched:
            raise KeyError(
                f"No known LoRA strength input found on node '{title}'. "
                f"Tried: {', '.join(input_names)}"
            )

        return patched

    def patch_lora_by_title(
        self,
        title: str,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
    ) -> list[str]:
        patched = []

        self.set_existing_input_by_title(title, "lora_name", lora_name)
        patched.append("lora_name")

        model_strength_patched = False
        for input_name in ("strength_model", "model_strength", "strength"):
            if self.try_set_existing_input_by_title(title, input_name, strength_model):
                patched.append(input_name)
                model_strength_patched = True
                break

        if not model_strength_patched:
            raise KeyError(
                f"No known LoRA model strength input found on node '{title}'. "
                "Tried: strength_model, model_strength, strength"
            )

        for input_name in ("strength_clip", "clip_strength"):
            if self.try_set_existing_input_by_title(title, input_name, strength_clip):
                patched.append(input_name)
                break

        return patched

    def patch_lora_strengths_by_title(
        self,
        title: str,
        strength_model: float,
        strength_clip: float,
    ) -> list[str]:
        patched = []

        for input_name in ("strength_model", "model_strength", "strength"):
            if self.try_set_existing_input_by_title(title, input_name, strength_model):
                patched.append(input_name)
                break

        for input_name in ("strength_clip", "clip_strength"):
            if self.try_set_existing_input_by_title(title, input_name, strength_clip):
                patched.append(input_name)
                break

        if not patched:
            raise KeyError(
                f"No known LoRA strength input found on node '{title}'. "
                "Tried model and clip strength inputs."
            )

        return patched
