"""Golden vocabulary test for the pipeline-stage owner (domain/stages.py).

Pins the exact stage vocabulary so that adding a stage forces a conscious
decision plus a test update, instead of a silent drift between the enum,
the resume order, and the resource classification.
"""

from __future__ import annotations

import unittest

from feverslop.domain.stages import (
    COMFYUI_RENDERING_STAGES,
    COMFYUI_STAGES,
    LLM_STAGES,
    NEUTRAL_STAGES,
    PipelineStage,
    RESUME_STAGE_ORDER,
)
from feverslop.domain.resource_phase import stage_resource


class StageVocabularyTests(unittest.TestCase):
    def test_exact_enum_values_in_order(self):
        self.assertEqual(
            [
                "tests",
                "main_pipeline",
                "h3_prompts",
                "render_plan",
                "relay_compact",
                "anchor_fix",
                "set_resolution",
                "sync_project_settings",
                "storyboard_frames",
                "storyboard_page",
                "reference_render",
                "reference_sheets",
                "msr_references",
                "msr_reference_sheets",
                "msr_prompt_enrich",
                "ingredients_sheets",
                "ltx_prepare_workflows",
                "ltx_render_scenes",
                "prepare_workflows",
                "render_scenes",
                "upscale",
                "concat_video_only",
                "mux_original_audio",
                "diagnostic_scene_audio_concat",
                "facefix",
                "facefix_concat",
                "export_timeline",
                "openshot_export",
            ],
            [stage.value for stage in PipelineStage],
        )

    def test_exact_resume_order(self):
        self.assertEqual(
            [
                "tests",
                "main_pipeline",
                "sync_project_settings",
                "relay_compact",
                "anchor_fix",
                "msr_references",
                "msr_reference_sheets",
                "h3_prompts",
                "render_plan",
                "msr_prompt_enrich",
                "ingredients_sheets",
                "ltx_prepare_workflows",
                "ltx_render_scenes",
                "facefix",
                "upscale",
                "concat_video_only",
                "mux_original_audio",
                "diagnostic_scene_audio_concat",
                "export_timeline",
            ],
            list(RESUME_STAGE_ORDER),
        )

    def test_exact_set_memberships(self):
        self.assertEqual(
            {
                "main_pipeline",
                "relay_compact",
                "h3_prompts",
                "msr_prompt_enrich",
                "ingredients_sheets",
            },
            LLM_STAGES,
        )
        self.assertEqual(
            {
                "storyboard_frames",
                "msr_references",
                "ltx_prepare_workflows",
                "ltx_render_scenes",
                "facefix",
                "upscale",
            },
            COMFYUI_STAGES,
        )
        self.assertEqual(
            {
                "tests",
                "sync_project_settings",
                "anchor_fix",
                "msr_reference_sheets",
                "render_plan",
                "storyboard_page",
                "concat_video_only",
                "mux_original_audio",
                "diagnostic_scene_audio_concat",
                "export_timeline",
            },
            NEUTRAL_STAGES,
        )
        self.assertEqual(
            {
                PipelineStage.STORYBOARD_FRAMES,
                PipelineStage.MSR_REFERENCES,
                PipelineStage.LTX_RENDER_SCENES,
                PipelineStage.RENDER_SCENES,
                PipelineStage.FACEFIX,
                PipelineStage.UPSCALE,
            },
            COMFYUI_RENDERING_STAGES,
        )

    def test_stage_resource_classifies_exactly_and_raises_for_unclassified(self):
        from feverslop.domain.resource_phase import StageResource

        for stage in PipelineStage:
            if stage.value in LLM_STAGES:
                self.assertIs(StageResource.LLM, stage_resource(stage.value))
            elif stage.value in COMFYUI_STAGES:
                self.assertIs(StageResource.COMFYUI, stage_resource(stage.value))
            elif stage.value in NEUTRAL_STAGES:
                self.assertIsNone(stage_resource(stage.value))
            else:
                with self.assertRaises(ValueError):
                    stage_resource(stage.value)

    def test_membership_ledger(self):
        """Each stage is exactly one of: resumable, classified-not-resumable, or unclassified."""
        all_stages = {stage.value for stage in PipelineStage}
        resumable = set(RESUME_STAGE_ORDER)
        classified = LLM_STAGES | COMFYUI_STAGES | NEUTRAL_STAGES
        # 19 resumable + 2 classified-but-not-resumable + 7 unclassified = 28
        self.assertEqual(19, len(resumable))
        self.assertEqual(2, len(classified - resumable))
        self.assertEqual(7, len(all_stages - classified))
        # No overlap between resumable and unclassified.
        self.assertEqual(set(), resumable & (all_stages - classified))


if __name__ == "__main__":
    unittest.main()
