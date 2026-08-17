"""Tests for composition defensive guards (Issue #279)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestValidateRenderPlanEntries(unittest.TestCase):
    """COMP-105: render plan schema validation."""

    def _write_plan(self, plan, filename="plan.json"):
        """Write a render plan to a temp file and return path."""
        if isinstance(plan, str):
            content = plan
        else:
            import json
            content = json.dumps(plan)
        tmp = Path(tempfile.mkdtemp()) / filename
        self.addCleanup(shutil.rmtree, tmp.parent, ignore_errors=True)
        tmp.write_text(content)
        return tmp

    def test_collect_rejects_non_list_render_plan(self):
        from feverslop.composition.config_loader import collect_render_plan_scene_clips
        render_plan = '{"scenario": "movie", "scenes": []}'
        tmp = self._write_plan(render_plan)
        with self.assertRaises(ValueError) as ctx:
            collect_render_plan_scene_clips(tmp, tmp.parent)
        self.assertIn("JSON array", str(ctx.exception))

    def test_collect_rejects_missing_scene_key(self):
        from feverslop.composition.config_loader import collect_render_plan_scene_clips
        render_plan = [{"prompt": "something"}]
        tmp = self._write_plan(render_plan)
        with self.assertRaises(ValueError) as ctx:
            collect_render_plan_scene_clips(tmp, tmp.parent)
        self.assertIn("missing required 'scene'", str(ctx.exception))

    def test_collect_rejects_non_numeric_scene(self):
        from feverslop.composition.config_loader import collect_render_plan_scene_clips
        render_plan = [{"scene": "not_a_number"}]
        tmp = self._write_plan(render_plan)
        with self.assertRaises(ValueError) as ctx:
            collect_render_plan_scene_clips(tmp, tmp.parent)
        self.assertIn("not numeric", str(ctx.exception))

    def test_count_rejects_malformed_plan(self):
        from feverslop.composition.config_loader import count_render_plan_items
        render_plan = [{"scene": "abc", "prompt": "x"}]
        tmp = self._write_plan(render_plan)
        with self.assertRaises(ValueError):
            count_render_plan_items(tmp)

    def test_count_works_on_valid_plan(self):
        from feverslop.composition.config_loader import count_render_plan_items
        render_plan = [
            {"scene": 1, "prompt": "a"},
            {"scene": 2, "prompt": "b"},
            {"scene": 3, "prompt": "c"},
        ]
        tmp = self._write_plan(render_plan)
        self.assertEqual(3, count_render_plan_items(tmp))

    def test_count_filters_by_scene_numbers(self):
        from feverslop.composition.config_loader import count_render_plan_items
        render_plan = [
            {"scene": 1, "prompt": "a"},
            {"scene": 2, "prompt": "b"},
            {"scene": 3, "prompt": "c"},
        ]
        tmp = self._write_plan(render_plan)
        self.assertEqual(2, count_render_plan_items(tmp, scene_numbers={1, 3}))


class TestSelectedVideoWorkflows(unittest.TestCase):
    """COMP-107: empty workflow guard."""

    def test_empty_workflows_raises_value_error(self):
        from feverslop.composition.stage_runners import _selected_video_workflows

        args = MagicMock()
        args.video_pipeline = "ltx_msr"
        args.render_mode = "auto"

        state = MagicMock()
        state.args = args
        state.msr_workflow = Path("")
        state.relay_workflow = Path(".")
        state.single_prompt_workflow = Path("")

        with self.assertRaises(ValueError) as ctx:
            _selected_video_workflows(state)
        self.assertIn("pipeline", str(ctx.exception).lower())

    def test_valid_workflows_returned(self):
        from feverslop.composition.stage_runners import _selected_video_workflows

        args = MagicMock()
        args.video_pipeline = "ltx_msr"
        args.render_mode = "auto"

        state = MagicMock()
        state.args = args
        state.msr_workflow = Path("/workflows/msr.json")
        state.relay_workflow = Path("/workflows/relay.json")
        state.single_prompt_workflow = Path("/workflows/sp.json")

        result = _selected_video_workflows(state)
        self.assertEqual(1, len(result))
        self.assertEqual(Path("/workflows/msr.json"), result[0])

    def test_minimax_r2v_uses_bundled_r2v_workflow_when_cli_default_is_ltx(self):
        from feverslop.composition.stage_runners import _selected_video_workflows

        args = MagicMock()
        args.video_pipeline = "minimax-h3-r2v"
        args.render_mode = "single_prompt"

        state = MagicMock()
        state.args = args
        state.single_prompt_workflow = Path("workflows/video_ltxv_i2v_v2.json")
        state.relay_workflow = Path("")

        self.assertEqual(
            Path("workflows/video_minimax_h3_r2v_v1.json"),
            _selected_video_workflows(state)[0],
        )


class TestRunUnitTestSuite(unittest.TestCase):
    """COMP-108: subprocess cwd."""

    @patch("feverslop.composition.stage_runners.subprocess.run")
    @patch("feverslop.composition.stage_runners.runner_root", return_value=Path("/repo/root"))
    def test_uses_runner_root_cwd(self, mock_root, mock_run):
        from feverslop.composition.stage_runners import run_unittest_suite
        mock_run.return_value = MagicMock(returncode=0)
        run_unittest_suite()
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        self.assertEqual(Path("/repo/root"), call_kwargs.kwargs.get("cwd") or call_kwargs[1].get("cwd"))


class TestContinuityDownstream(unittest.TestCase):
    """COMP-109: cycle detection."""

    def test_continuity_downstream_detects_cycle(self):
        from feverslop.composition.stage_runners import _continuity_downstream
        # Cycle: 2 -> 3 -> 4 -> 2
        predecessors_cycle = {3: 2, 4: 3, 2: 4}
        result = _continuity_downstream(2, predecessors_cycle)
        # Starting from 2: 2->3 (dependent of 2), 3->4 (dependent of 3), 4->2 (cycle, 2 not in downstream, but 4 is dependent of 3)
        # Actually: next(x for x, pred in preds if pred == 2) = 3, next(x for x,pred in preds if pred==3) = 4, next(x for x,pred in preds if pred==4) = 2
        # 2 is NOT in downstream {3, 4}, so it would add 2. Then next(x for x,pred if pred==2) = 3, which IS in downstream. So result = {3, 4, 2}
        # Wait, let me trace through again: start from current=2
        # dependent = next(x for x,p in predecessors.items() if p == 2) = 3
        # 3 not in downstream={} -> add 3 -> downstream={3}, current=3
        # dependent = next(x for x,p in predecessors.items() if p == 3) = 4
        # 4 not in downstream={3} -> add 4 -> downstream={3,4}, current=4
        # dependent = next(x for x,p in predecessors.items() if p == 4) = 2
        # 2 not in downstream={3,4} -> add 2 -> downstream={3,4,2}, current=2
        # dependent = next(x for x,p in predecessors.items() if p == 2) = 3
        # 3 IS in downstream={3,4,2} -> return {3,4,2}
        self.assertIn(3, result)
        self.assertIn(4, result)
        self.assertIn(2, result)

    def test_continuity_downstream_normal(self):
        from feverslop.composition.stage_runners import _continuity_downstream
        predecessors = {2: 1, 3: 2}
        result = _continuity_downstream(1, predecessors)
        self.assertEqual({2, 3}, result)

    def test_continuity_downstream_no_dependents(self):
        from feverslop.composition.stage_runners import _continuity_downstream
        predecessors = {2: 1}  # 2 depends on 1
        result = _continuity_downstream(2, predecessors)
        self.assertEqual(set(), result)


if __name__ == "__main__":
    unittest.main()
