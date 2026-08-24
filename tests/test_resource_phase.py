from __future__ import annotations

import unittest


class ResourcePhaseTests(unittest.TestCase):
    def test_selects_comfyui_phase_with_leading_and_trailing_neutral_stages(self):
        from feverslop.domain.resource_phase import StageResource, select_first_resource_phase

        phase = select_first_resource_phase((
            "sync_project_settings",
            "msr_references",
            "msr_reference_sheets",
            "h3_prompts",
            "render_plan",
            "ltx_render_scenes",
        ))

        self.assertEqual(
            ("sync_project_settings", "msr_references", "msr_reference_sheets"),
            phase.stages,
        )
        self.assertIs(StageResource.COMFYUI, phase.resource)
        self.assertIs(StageResource.LLM, phase.next_resource)

    def test_repeated_selection_partitions_comfyui_llm_comfyui_sequence(self):
        from feverslop.domain.resource_phase import StageResource, select_first_resource_phase

        stages = (
            "msr_references",
            "msr_reference_sheets",
            "h3_prompts",
            "render_plan",
            "ltx_render_scenes",
            "concat_video_only",
        )

        first = select_first_resource_phase(stages)
        second = select_first_resource_phase(stages[len(first.stages):])
        third = select_first_resource_phase(stages[len(first.stages) + len(second.stages):])

        self.assertEqual(("msr_references", "msr_reference_sheets"), first.stages)
        self.assertEqual(("h3_prompts", "render_plan"), second.stages)
        self.assertEqual(("ltx_render_scenes", "concat_video_only"), third.stages)
        self.assertEqual(
            (StageResource.COMFYUI, StageResource.LLM, StageResource.COMFYUI),
            (first.resource, second.resource, third.resource),
        )
        self.assertIsNone(third.next_resource)

    def test_all_neutral_stages_form_one_phase_without_owner(self):
        from feverslop.domain.resource_phase import select_first_resource_phase

        phase = select_first_resource_phase((
            "sync_project_settings",
            "render_plan",
            "ltx_prepare_workflows",
            "concat_video_only",
        ))

        self.assertEqual(
            (
                "sync_project_settings",
                "render_plan",
                "ltx_prepare_workflows",
                "concat_video_only",
            ),
            phase.stages,
        )
        self.assertIsNone(phase.resource)
        self.assertIsNone(phase.next_resource)

    def test_ingredients_sheets_is_llm_owned(self):
        from feverslop.domain.resource_phase import StageResource, stage_resource

        self.assertIs(StageResource.LLM, stage_resource("ingredients_sheets"))

    def test_empty_stage_sequence_is_supported(self):
        from feverslop.domain.resource_phase import select_first_resource_phase

        phase = select_first_resource_phase(())

        self.assertEqual((), phase.stages)
        self.assertIsNone(phase.resource)
        self.assertIsNone(phase.next_resource)

    def test_unknown_stage_fails_closed(self):
        from feverslop.domain.resource_phase import select_first_resource_phase

        with self.assertRaisesRegex(ValueError, "unclassified.*future_gpu_stage"):
            select_first_resource_phase(("future_gpu_stage",))


if __name__ == "__main__":
    unittest.main()
