from typing import Any


def build_reference_sheet_signature(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    from feverslop.domain.reference_sheet import ReferenceSheetPlan

    class ReferenceSheetFromSequence(dspy_module.Signature):
        """Create semantic instructions for a backend-neutral multi-view reference take.

        For a character, anchor_description contains only stable identity and appearance:
        face, hair, body, wardrobe, materials, and colors. It must describe one neutral
        character and must not include actions, performance, instruments, props, or a
        scene location. Motion and location belong to the later sequence instructions.
        """

        kind: str = dspy_module.InputField()
        description: str = dspy_module.InputField()
        asset_context: dict[str, Any] = dspy_module.InputField()
        plan: ReferenceSheetPlan = dspy_module.OutputField()

    return ReferenceSheetFromSequence
