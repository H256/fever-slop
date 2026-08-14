import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from feverslop.composition.arg_parser import PipelineStage


class PipelineComfyUICacheTests(unittest.TestCase):
    def test_frees_comfyui_before_each_rendering_stage(self):
        from feverslop.composition import pipeline_runner

        events = []
        client = SimpleNamespace(free_cache_and_vram=lambda: events.append("free"))
        state = SimpleNamespace(
            comfyui_client=client,
            plan_for_next_step="plan.json",
            final_video_path=None,
            video_only_path=None,
            openshot_project_path=None,
        )
        stages = [PipelineStage.INGREDIENTS_SHEETS, PipelineStage.LTX_RENDER_SCENES]

        with (
            patch.object(pipeline_runner, "resolve_pipeline_stages", return_value=stages),
            patch.object(pipeline_runner, "build_run_state", return_value=state),
            patch.dict(
                pipeline_runner.STAGE_RUNNERS,
                {
                    PipelineStage.INGREDIENTS_SHEETS: lambda _state: events.append("sheets"),
                    PipelineStage.LTX_RENDER_SCENES: lambda _state: events.append("ltx"),
                },
            ),
        ):
            pipeline_runner.run(argparse.Namespace())

        self.assertEqual(["sheets", "free", "ltx"], events)


if __name__ == "__main__":
    unittest.main()
