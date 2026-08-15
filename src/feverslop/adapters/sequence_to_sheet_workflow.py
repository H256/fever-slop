"""Patcher-facing contracts for sequence-to-sheet ComfyUI workflows."""

from __future__ import annotations

from dataclasses import dataclass

from feverslop.adapters.workflow_patcher import WorkflowPatcher


@dataclass(frozen=True, slots=True)
class SequenceToSheetWorkflowProfile:
    backend: str
    workflow_filename: str
    required_titles: tuple[str, ...]
    prompt_titles: tuple[str, ...]
    required_class_types: tuple[str, ...] = ()

    def validate(self, workflow: dict) -> tuple[str, ...]:
        patcher = WorkflowPatcher(workflow)
        missing: list[str] = []
        for title in self.required_titles:
            try:
                patcher.find_node_by_meta_title(title)
            except KeyError:
                missing.append(title)

        if self.prompt_titles and not any(self._has_title(patcher, title) for title in self.prompt_titles):
            missing.append(" or ".join(self.prompt_titles))
        for class_type in self.required_class_types:
            if not patcher.find_nodes_by_class_type(class_type):
                missing.append(f"class_type:{class_type}")
        return tuple(missing)

    @staticmethod
    def _has_title(patcher: WorkflowPatcher, title: str) -> bool:
        try:
            patcher.find_node_by_meta_title(title)
            return True
        except KeyError:
            return False


MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE = SequenceToSheetWorkflowProfile(
    backend="minimax-h3",
    workflow_filename="sequence_to_sheet_minimax_h3_v1.json",
    required_titles=(
        "#MEGAPIXELS",
        "#R2V_COMBINE",
        "#PROMPT",
        "#SEED",
        "#FRAMECOUNT",
        *(f"#REF_{index}" for index in range(1, 10)),
        "#SAVE_VIDEO",
    ),
    prompt_titles=(),
    required_class_types=("VRAMCleanup",),
)
