from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class PipelineActionAvailabilityTests(unittest.TestCase):
    def test_missing_selected_scene_workflow_recommends_preparation_and_blocks_render(self):
        from feverslop.studio.pipeline_actions import pipeline_action_availability

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / ".studio"
            state_dir.mkdir()
            (state_dir / "pipeline_state.json").write_text(
                '{"completed_stages": ["main-pipeline", "msr-references", "msr_reference_sheets", "msr_prompt_enrich"]}',
                encoding="utf-8",
            )
            prepared = root / "output" / "render" / "scenes" / "scene_0001"
            prepared.mkdir(parents=True)
            (prepared / "workflow.json").write_text("{}", encoding="utf-8")
            (prepared / "manifest.json").write_text("{}", encoding="utf-8")

            actions = {item["value"]: item for item in pipeline_action_availability(root, [1, 3])}

        self.assertTrue(actions["ltx-prepare-workflows"]["enabled"])
        self.assertTrue(actions["ltx-prepare-workflows"]["recommended"])
        self.assertFalse(actions["ltx-render-scenes"]["enabled"])
        self.assertEqual("Prepare LTX workflows for scenes 3 first.", actions["ltx-render-scenes"]["reason"])

    def test_unfinished_enrichment_blocks_workflow_preparation(self):
        from feverslop.studio.pipeline_actions import pipeline_action_availability

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / ".studio"
            state_dir.mkdir()
            (state_dir / "pipeline_state.json").write_text(
                '{"completed_stages": ["main-pipeline", "msr-references"]}',
                encoding="utf-8",
            )

            actions = {item["value"]: item for item in pipeline_action_availability(root, [3])}

        self.assertTrue(actions["msr-enrich"]["recommended"])
        self.assertFalse(actions["ltx-prepare-workflows"]["enabled"])
        self.assertEqual("Run MSR enrichment first.", actions["ltx-prepare-workflows"]["reason"])

    def test_all_selected_workflows_enable_render(self):
        from feverslop.studio.pipeline_actions import pipeline_action_availability

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for scene in [1, 3]:
                prepared = root / "output" / "render" / "scenes" / f"scene_{scene:04d}"
                prepared.mkdir(parents=True)
                (prepared / "workflow.json").write_text("{}", encoding="utf-8")
                (prepared / "manifest.json").write_text("{}", encoding="utf-8")

            actions = {item["value"]: item for item in pipeline_action_availability(root, [1, 3])}

        self.assertTrue(actions["ltx-render-scenes"]["enabled"])
        self.assertTrue(actions["ltx-render-scenes"]["recommended"])
        self.assertEqual("", actions["ltx-render-scenes"]["reason"])

    def test_final_concat_is_blocked_until_scene_clips_exist(self):
        from feverslop.studio.pipeline_actions import pipeline_action_availability

        with tempfile.TemporaryDirectory() as temp_dir:
            actions = {item["value"]: item for item in pipeline_action_availability(Path(temp_dir), [1])}

        self.assertFalse(actions["final-concat"]["enabled"])
        self.assertEqual("Render scene clips first.", actions["final-concat"]["reason"])

    def test_final_concat_stays_blocked_after_partial_scene_render(self):
        from feverslop.studio.pipeline_actions import pipeline_action_availability

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plans = root / "output" / "render" / "plans"
            plans.mkdir(parents=True)
            (plans / "references.json").write_text('[{"scene": 1}, {"scene": 3}]', encoding="utf-8")
            rendered = root / "output" / "render" / "scenes" / "scene_0001"
            rendered.mkdir(parents=True)
            (rendered / "final.mp4").touch()

            actions = {item["value"]: item for item in pipeline_action_availability(root, [1])}

        self.assertFalse(actions["final-concat"]["enabled"])
        self.assertEqual("Render scene clips first.", actions["final-concat"]["reason"])
