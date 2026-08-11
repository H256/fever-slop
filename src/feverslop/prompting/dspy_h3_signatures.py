def build_dspy_signatures():
    import dspy

    from feverslop.prompting.dspy_h3_models import (
        BaseVideoPrompt,
        ImageAnalysis,
        PromptPlan,
        RetentionAnalysis,
    )

    class AnalyzeImage(dspy.Signature):
        """Analyze only observable information in a reference image for video generation."""
        image: dspy.Image = dspy.InputField()
        intended_role: str = dspy.InputField()
        user_hint: str = dspy.InputField()
        analysis: ImageAnalysis = dspy.OutputField()

    class BuildPromptPlan(dspy.Signature):
        """Create a strict plan using only supplied references.

        `music_intent=none` means no audience-only score and requires
        `non_diegetic_music` to be omitted or N/A. For `generate` or
        `reference`, provide a concrete non-diegetic music description.
        Scene vocals, instruments, and referenced soundtrack audio belong in
        the detailed description and audio references, not in this field.
        """
        mode: str = dspy.InputField()
        user_prompt: str = dspy.InputField()
        duration_seconds: float | None = dspy.InputField()
        references_json: str = dspy.InputField()
        notes: str = dspy.InputField()
        strict_fidelity: bool = dspy.InputField()
        requested_music_intent: str = dspy.InputField()
        relay_segments_json: str = dspy.InputField()
        plan: PromptPlan = dspy.OutputField()

    class RenderBasePrompt(dspy.Signature):
        """Render a production-ready MiniMax base prompt; the guide is authoritative."""
        guide: str = dspy.InputField()
        mode: str = dspy.InputField()
        user_prompt: str = dspy.InputField()
        plan_json: str = dspy.InputField()
        references_json: str = dspy.InputField()
        strict_fidelity: bool = dspy.InputField()
        music_intent: str = dspy.InputField()
        relay_segments_json: str = dspy.InputField()
        result: BaseVideoPrompt = dspy.OutputField()

    class RenderReferencePrompt(dspy.Signature):
        """Render all generated portions of a MiniMax full-reference prompt."""
        guide: str = dspy.InputField()
        user_prompt: str = dspy.InputField()
        plan_json: str = dspy.InputField()
        references_json: str = dspy.InputField()
        strict_fidelity: bool = dspy.InputField()
        music_intent: str = dspy.InputField()
        relay_segments_json: str = dspy.InputField()
        summary: str = dspy.OutputField()
        retention_analysis: list[RetentionAnalysis] = dspy.OutputField()
        detailed_description: str = dspy.OutputField()
        overall_soundscape: str = dspy.OutputField()
        non_diegetic_music: str | None = dspy.OutputField(
            desc=(
                "Audience-only background music prose, or null/N/A. "
                "Never include <Audio N> labels or audio-reference definitions here; "
                "put those in detailed_description and retention_analysis."
            )
        )

    return AnalyzeImage, BuildPromptPlan, RenderBasePrompt, RenderReferencePrompt