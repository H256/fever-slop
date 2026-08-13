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


if __name__ == "__main__":
    unittest.main()
