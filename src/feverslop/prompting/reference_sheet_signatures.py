from typing import Any


def build_reference_sheet_signature(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    from feverslop.domain.reference_sheet import ReferenceSheetPlanContract

    class ReferenceSheetFromSequence(dspy_module.Signature):
        """Create semantic instructions for a backend-neutral multi-view reference take.

        For a character, anchor_description contains only stable identity and appearance:
        face, hair, body, wardrobe, materials, and colors. It must describe one neutral
        character and must not include actions, performance, instruments, props, or a
        scene location. Motion and location belong to the later sequence instructions.
        """

        kind: str = dspy_module.InputField(desc="Asset kind: character or location.")
        description: str = dspy_module.InputField(desc="The stable appearance or environment description to preserve.")
        asset_context: dict[str, Any] = dspy_module.InputField(desc="Structured asset facts; do not invent unrelated actors, props, or locations.")
        plan: ReferenceSheetPlanContract = dspy_module.OutputField(desc="A complete plan with one view label for every requested view and bounded text fields.")

    return ReferenceSheetFromSequence
