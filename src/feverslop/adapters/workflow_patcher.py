from __future__ import annotations

from copy import deepcopy
from typing import Any
import re


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

    def find_nodes_by_meta_title(self, title: str) -> list[tuple[str, dict]]:
        return [
            (node_id, node)
            for node_id, node in self.workflow.items()
            if node.get("_meta", {}).get("title") == title
        ]

    def apply_patch_spec(self, operations: list[dict], context: dict | None = None) -> "WorkflowPatcher":
        context = context or {}
        for operation in operations:
            op = operation.get("op")
            if op == "set_input":
                self._apply_set_input(operation, context)
            elif op == "remove_node":
                self._apply_remove_node(operation)
            elif op == "insert_node_between":
                self._apply_insert_node_between(operation)
            else:
                raise ValueError(f"Unsupported workflow patch op: {op}")
        return self

    def _apply_set_input(self, operation: dict, context: dict) -> None:
        _, node = self._resolve_target(operation["target"])
        if "value_from" in operation:
            value = resolve_dotted_path(context, str(operation["value_from"]))
        else:
            value = operation.get("value")
        node.setdefault("inputs", {})[str(operation["input"])] = value

    def _apply_remove_node(self, operation: dict) -> None:
        node_id, node = self._resolve_target(operation["target"])
        inputs = node.get("inputs", {})
        for bridge in operation.get("bridge", []):
            from_input = str(bridge["from_input"])
            if from_input not in inputs:
                raise KeyError(f"Bridge input '{from_input}' not found on removed node {node_id}")
            _, target_node = self._resolve_target(bridge["to"])
            target_node.setdefault("inputs", {})[str(bridge["to"]["input"])] = deepcopy(inputs[from_input])

        # Detect dangling wire references
        dangling = self._find_dangling_references(node_id)
        if dangling:
            lines = [f"Removing node '{node_id}' leaves dangling wire reference(s):"]
            for ref_id, inp_name, inp_val in dangling:
                lines.append(f"  - node {ref_id}, input '{inp_name}' -> {inp_val}")
            lines.append("Specify bridge entries in the remove_node operation to resolve these references.")
            raise ValueError("\n".join(lines))

        del self.workflow[node_id]

    def _find_dangling_references(self, removed_node_id: str) -> list[tuple[str, str, Any]]:
        """Find inputs in remaining nodes that reference the removed node."""
        dangling: list[tuple[str, str, Any]] = []
        for other_id, other_node in self.workflow.items():
            if other_id == removed_node_id:
                continue
            for input_name, input_value in other_node.get("inputs", {}).items():
                for dep_id in _workflow_dependency_ids(input_value):
                    if dep_id == removed_node_id:
                        dangling.append((other_id, input_name, input_value))
        return dangling

    def _apply_insert_node_between(self, operation: dict) -> None:
        new_node_id = str(operation["new_node_id"])
        if new_node_id in self.workflow:
            raise ValueError(f"Node id already exists: {new_node_id}")

        source_id, _ = self._resolve_target(operation["source"])
        _, target_node = self._resolve_target(operation["target"])
        new_node = deepcopy(operation["node"])
        new_node.setdefault("inputs", {})[str(operation["new_node_input"])] = [
            source_id,
            int(operation.get("source", {}).get("output", 0)),
        ]
        self.workflow[new_node_id] = new_node
        target_node.setdefault("inputs", {})[str(operation["target"]["input"])] = [
            new_node_id,
            int(operation.get("new_node_output", 0)),
        ]

    def _resolve_target(self, target: dict) -> tuple[str, dict]:
        node_id_key = target.get("id") or target.get("node_id")
        if node_id_key is not None:
            node_id = str(node_id_key)
            return node_id, self.find_node_by_id(node_id)
        if "title" in target:
            return self.find_node_by_meta_title(str(target["title"]))
        if "class_type" in target:
            nodes = self.find_nodes_by_class_type(str(target["class_type"]))
            if not nodes:
                raise KeyError(f"No node with class_type found: {target['class_type']}")
            return nodes[int(target.get("index", 0))]
        raise ValueError(f"Patch target must include id, title, or class_type: {target}")

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

    def set_existing_input_by_title_any(
        self,
        title: str,
        input_name: str,
        value: Any,
    ) -> "WorkflowPatcher":
        nodes = self.find_nodes_by_meta_title(title)
        for _, node in nodes:
            inputs = node.setdefault("inputs", {})
            if input_name in inputs:
                inputs[input_name] = value
                return self

        if not nodes:
            raise KeyError(f"Node with _meta.title not found: {title}")
        raise KeyError(f"Input '{input_name}' not found on any node titled '{title}'")

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

    def patch_lora_fields_by_title(
        self,
        title: str,
        *,
        lora_name: str | None = None,
        strength_model: float | None = None,
        strength_clip: float | None = None,
    ) -> list[str]:
        patched = []

        if lora_name is not None:
            self.set_existing_input_by_title(title, "lora_name", lora_name)
            patched.append("lora_name")

        if strength_model is not None:
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

        if strength_clip is not None:
            for input_name in ("strength_clip", "clip_strength"):
                if self.try_set_existing_input_by_title(title, input_name, strength_clip):
                    patched.append(input_name)
                    break

        if not patched:
            raise KeyError(f"No LoRA fields were patched on node '{title}'")

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

    def prune_unreachable_nodes(
        self,
        *,
        root_titles: tuple[str, ...] = (),
        root_class_types: tuple[str, ...] = (),
    ) -> set[str]:
        """Remove nodes that cannot contribute to any selected output node.

        ComfyUI API workflows may contain template anchors and entire branches
        that are no longer connected after dynamic inputs are patched. Walking
        backwards from the output nodes keeps only nodes referenced by an
        actual input link and avoids submitting stale paths to ComfyUI.
        """
        roots = {
            node_id
            for node_id, node in self.workflow.items()
            if node.get("_meta", {}).get("title") in root_titles
            or node.get("class_type") in root_class_types
        }
        if not roots:
            return set()
        if not any(
            dependency
            for node_id in roots
            for input_value in self.workflow[node_id].get("inputs", {}).values()
            for dependency in _workflow_dependency_ids(input_value)
        ):
            return set()

        reachable: set[str] = set()
        pending = list(roots)
        while pending:
            node_id = str(pending.pop())
            if node_id in reachable or node_id not in self.workflow:
                continue
            reachable.add(node_id)
            for input_value in self.workflow[node_id].get("inputs", {}).values():
                for dependency_id in _workflow_dependency_ids(input_value):
                    if dependency_id not in reachable:
                        pending.append(dependency_id)

        removed = set(self.workflow) - reachable
        for node_id in removed:
            del self.workflow[node_id]
        return removed

    def remove_node_by_title(self, title: str) -> str:
        """Remove a node by title, return its node id."""
        node_id, _ = self.find_node_by_meta_title(title)
        del self.workflow[node_id]
        return node_id

    def add_node(self, node_id: str | int, node: dict) -> str:
        """Add a node with the given id. Returns the string id."""
        key = str(node_id)
        if key in self.workflow:
            raise ValueError(f"Node id already exists: {key}")
        self.workflow[key] = deepcopy(node)
        return key

    def find_free_node_id(self, start: int = 10000) -> int:
        """Find a free numeric node id starting from start."""
        candidate = start
        while str(candidate) in self.workflow:
            candidate += 1
        return candidate


_PATH_PART_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)])?")


def _workflow_dependency_ids(value: Any) -> set[str]:
    """Return node ids from nested ComfyUI input-link values."""
    dependencies: set[str] = set()
    if isinstance(value, list):
        if len(value) >= 2 and str(value[0]).strip() and isinstance(value[1], int):
            dependencies.add(str(value[0]))
        else:
            for item in value:
                dependencies.update(_workflow_dependency_ids(item))
    return dependencies


def resolve_dotted_path(context: dict, path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        match = _PATH_PART_RE.fullmatch(part)
        if not match:
            raise ValueError(f"Invalid dotted path part: {part}")
        key, index = match.groups()
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
        if index is not None:
            current = current[int(index)]
    return current
