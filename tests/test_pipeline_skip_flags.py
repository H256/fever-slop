import unittest

from feverslop.composition.arg_parser import PipelineStage, build_arg_parser
from feverslop.composition.stage_runners import resolve_pipeline_stages


class PipelineSkipFlagTests(unittest.TestCase):
    def test_audio_skip_flags_are_exposed(self):
        args = build_arg_parser().parse_args([
            "project",
            "--skip-stem-separation",
            "--skip-whisper",
            "--skip-beat-analysis",
        ])
        self.assertTrue(args.skip_stem_separation)
        self.assertTrue(args.skip_whisper)
        self.assertTrue(args.skip_beat_analysis)

    def test_generic_stage_names_accept_legacy_aliases(self):
        args = build_arg_parser().parse_args([
            "project",
            "--stage", "prepare_workflows",
            "--stage", "render_scenes",
        ])
        self.assertEqual(
            [PipelineStage.LTX_PREPARE_WORKFLOWS, PipelineStage.LTX_RENDER_SCENES],
            resolve_pipeline_stages(args),
        )

    def test_neutral_stage_names_and_render_skip_flag_are_supported(self):
        args = build_arg_parser().parse_args([
            "project",
            "--stage", "reference_render",
            "--stage", "reference_sheets",
            "--stage", "prepare_workflows",
            "--stage", "render_scenes",
            "--skip-render",
            "--skip-reference-render",
        ])

        self.assertTrue(args.skip_ltx)
        self.assertTrue(args.skip_msr_reference_render)
        self.assertEqual(
            [
                PipelineStage.MSR_REFERENCES,
                PipelineStage.MSR_REFERENCE_SHEETS,
                PipelineStage.LTX_PREPARE_WORKFLOWS,
                PipelineStage.LTX_RENDER_SCENES,
            ],
            resolve_pipeline_stages(args),
        )

    def test_minimax_resume_rebuilds_h3_prompts_before_rendering(self):
        args = build_arg_parser().parse_args([
            "project",
            "--skip-main-pipeline",
            "--video-pipeline", "minimax-h3-r2v",
            "--reference-generation", "sequence_sheet",
            "--skip-tests",
        ])

        stages = resolve_pipeline_stages(args)

        self.assertLess(stages.index(PipelineStage.MSR_REFERENCE_SHEETS), stages.index(PipelineStage.H3_PROMPTS))
        self.assertLess(stages.index(PipelineStage.H3_PROMPTS), stages.index(PipelineStage.RENDER_PLAN))
        self.assertLess(stages.index(PipelineStage.RENDER_PLAN), stages.index(PipelineStage.LTX_RENDER_SCENES))


if __name__ == "__main__":
    unittest.main()
