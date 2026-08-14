from typing import Any

from feverslop.prompting.dspy_runtime import H3SignatureBundle


def build_h3_signature_bundle(dspy_module: Any | None = None) -> H3SignatureBundle:
    if dspy_module is None:
        import dspy as dspy_module

    from feverslop.prompting.dspy_h3_models import (
        BaseVideoPrompt,
        ImageAnalysis,
        PromptPlan,
        RetentionAnalysis,
    )

    class AnalyzeImage(dspy_module.Signature):
        """Analyze only observable information in a reference image for video generation."""
        image: dspy_module.Image = dspy_module.InputField()
        intended_role: str = dspy_module.InputField()
        user_hint: str = dspy_module.InputField()
        analysis: ImageAnalysis = dspy_module.OutputField()

    class BuildPromptPlan(dspy_module.Signature):
        """Create a strict plan using only supplied references.

        `music_intent=none` means no audience-only score and requires
        `non_diegetic_music` to be omitted or N/A. For `generate` or
        `reference`, provide a concrete non-diegetic music description.
        Scene vocals, instruments, and referenced soundtrack audio belong in
        the detailed description and audio references, not in this field.
        """
        mode: str = dspy_module.InputField()
        user_prompt: str = dspy_module.InputField()
        duration_seconds: float | None = dspy_module.InputField()
        references_json: str = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        strict_fidelity: bool = dspy_module.InputField()
        requested_music_intent: str = dspy_module.InputField()
        relay_segments_json: str = dspy_module.InputField(default="[]")
        plan: PromptPlan = dspy_module.OutputField()

    class RenderBasePrompt(dspy_module.Signature):
        """Render a production-ready MiniMax base prompt; the guide is authoritative."""
        guide: str = dspy_module.InputField()
        mode: str = dspy_module.InputField()
        user_prompt: str = dspy_module.InputField()
        plan_json: str = dspy_module.InputField()
        references_json: str = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        strict_fidelity: bool = dspy_module.InputField()
        music_intent: str = dspy_module.InputField()
        relay_segments_json: str = dspy_module.InputField(default="[]")
        result: BaseVideoPrompt = dspy_module.OutputField()

    class RenderReferencePrompt(dspy_module.Signature):
        """Render all generated portions of a MiniMax full-reference prompt."""
        guide: str = dspy_module.InputField()
        user_prompt: str = dspy_module.InputField()
        plan_json: str = dspy_module.InputField()
        references_json: str = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        strict_fidelity: bool = dspy_module.InputField()
        music_intent: str = dspy_module.InputField()
        relay_segments_json: str = dspy_module.InputField(default="[]")
        summary: str = dspy_module.OutputField()
        retention_analysis: list[RetentionAnalysis] = dspy_module.OutputField()
        detailed_description: str = dspy_module.OutputField()
        overall_soundscape: str = dspy_module.OutputField()
        non_diegetic_music: str | None = dspy_module.OutputField(
            desc=(
                "Audience-only background music prose, or null/N/A. "
                "Never include <Audio N> labels or audio-reference definitions here; "
                "put those in detailed_description and retention_analysis."
            )
        )

    return H3SignatureBundle(AnalyzeImage, BuildPromptPlan, RenderBasePrompt, RenderReferencePrompt)


def build_dspy_signatures():
    bundle = build_h3_signature_bundle()
    return (
        bundle.analyze_image,
        bundle.build_prompt_plan,
        bundle.render_base_prompt,
        bundle.render_reference_prompt,
    )
