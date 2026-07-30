from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class RebuildPlanTimelineRegistrationTests(unittest.TestCase):
    """Verify that rebuild-plan-timeline is registered as a pipeline action."""

    def test_rebuild_plan_timeline_in_pipeline_actions(self):
        from feverslop.studio.jobs import PIPELINE_ACTIONS

        self.assertIn("rebuild-plan-timeline", PIPELINE_ACTIONS)

    def test_add_rebuild_plan_timeline_method_exists_on_registry(self):
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        self.assertTrue(hasattr(registry, "add_rebuild_plan_timeline"))

    def test_add_rebuild_plan_timeline_accepts_affected_artifacts(self):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts(beat_json=True)
        result = registry.add_rebuild_plan_timeline("/tmp/project", affected)
        self.assertIsInstance(result, dict)

    def test_payload_contains_required_keys(self):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        affected = AffectedArtifacts(beat_json=True, scene_srt=False, render_plan=True)
        registry = JobRegistry()
        result = registry.add_rebuild_plan_timeline("/tmp/my-project", affected, rebuild_id="abc-123")

        self.assertIn("project_dir", result)
        self.assertIn("affected", result)
        self.assertIn("rebuild_id", result)
        self.assertIn("timestamp", result)

    def test_payload_stores_affected_flags_correctly(self):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        affected = AffectedArtifacts(
            beat_json=True,
            scene_srt=False,
            stage1_segments=True,
            ltx_prompt=False,
            render_plan=True,
        )
        registry = JobRegistry()
        result = registry.add_rebuild_plan_timeline("/tmp/project", affected)

        flags = result["affected"]
        self.assertIsInstance(flags, dict)
        self.assertTrue(flags["beat_json"])
        self.assertFalse(flags["scene_srt"])
        self.assertTrue(flags["stage1_segments"])
        self.assertFalse(flags["ltx_prompt"])
        self.assertTrue(flags["render_plan"])

    def test_payload_uses_provided_rebuild_id(self):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts()
        result = registry.add_rebuild_plan_timeline("/tmp/project", affected, rebuild_id="custom-id")
        self.assertEqual("custom-id", result["rebuild_id"])

    def test_payload_generates_rebuild_id_when_none(self):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts()
        result = registry.add_rebuild_plan_timeline("/tmp/project", affected)
        self.assertIsNotNone(result["rebuild_id"])
        self.assertIsInstance(result["rebuild_id"], str)

    def test_timestamp_is_numeric(self):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts()
        result = registry.add_rebuild_plan_timeline("/tmp/project", affected)

        ts = result["timestamp"]
        self.assertIsInstance(ts, (int, float))
        self.assertGreater(ts, time.time() - 2)


class RebuildPlanTimelineHandlerTests(unittest.TestCase):
    """Verify that the handler calls rebuild functions for affected artifacts."""

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_handler_calls_rebuild_beat_json_when_affected(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts(beat_json=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)

        # Wait briefly for the async job to complete
        time.sleep(0.3)
        mock_beat.assert_called_once()
        mock_scene_srt.assert_not_called()
        mock_stage1.assert_not_called()
        mock_ltx_prompt.assert_not_called()
        mock_render_plan.assert_not_called()

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_handler_calls_all_affected_rebuilds(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts(
            beat_json=True,
            scene_srt=True,
            stage1_segments=True,
            ltx_prompt=True,
            render_plan=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)
        time.sleep(0.3)

        mock_beat.assert_called_once()
        mock_scene_srt.assert_called_once()
        mock_stage1.assert_called_once()
        mock_ltx_prompt.assert_called_once()
        mock_render_plan.assert_called_once()

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_handler_passes_project_dir_to_rebuilds(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts(beat_json=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)
        time.sleep(0.3)

        mock_beat.assert_called_once()
        call_args = mock_beat.call_args
        project_dir_arg = call_args[0][0]  # first element of the args tuple
        self.assertIsInstance(project_dir_arg, str)
        self.assertEqual(project_dir_arg, str(Path(tmpdir)))


class RebuildPlanTimelineDependencyOrderTests(unittest.TestCase):
    """Verify dependency ordering: beat -> scene -> stage1 -> prompt -> render."""

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_dependency_order_when_all_affected(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        """Functions must be called in dependency order: beat < scene < stage1 < prompt < render."""
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        call_order = []

        def track(name):
            def wrapper(*args, **kwargs):
                call_order.append(name)
                return {"status": "ok"}
            return wrapper

        mock_beat.side_effect = track("beat_json")
        mock_scene_srt.side_effect = track("scene_srt")
        mock_stage1.side_effect = track("stage1_segments")
        mock_ltx_prompt.side_effect = track("ltx_prompt")
        mock_render_plan.side_effect = track("render_plan")

        registry = JobRegistry()
        affected = AffectedArtifacts(
            beat_json=True,
            scene_srt=True,
            stage1_segments=True,
            ltx_prompt=True,
            render_plan=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)
        time.sleep(0.3)

        self.assertEqual(
            ["beat_json", "scene_srt", "stage1_segments", "ltx_prompt", "render_plan"],
            call_order,
        )

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_dependency_order_partial_subset(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        """When only beat and render_plan are affected, beat must still run before render_plan."""
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        call_order = []

        def track(name):
            def wrapper(*args, **kwargs):
                call_order.append(name)
                return {"status": "ok"}
            return wrapper

        mock_beat.side_effect = track("beat_json")
        mock_scene_srt.side_effect = track("scene_srt")
        mock_stage1.side_effect = track("stage1_segments")
        mock_ltx_prompt.side_effect = track("ltx_prompt")
        mock_render_plan.side_effect = track("render_plan")

        registry = JobRegistry()
        affected = AffectedArtifacts(beat_json=True, render_plan=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)
        time.sleep(0.3)

        self.assertEqual(["beat_json", "render_plan"], call_order)


class RebuildPlanTimelineErrorHandlingTests(unittest.TestCase):
    """Verify that one failure doesn't stop other rebuilds."""

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_partial_failure_continues_other_rebuilds(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        mock_beat.side_effect = RuntimeError("beat rebuild failed")
        mock_scene_srt.return_value = {"status": "ok"}
        mock_render_plan.return_value = {"status": "ok"}

        registry = JobRegistry()
        affected = AffectedArtifacts(
            beat_json=True,
            scene_srt=True,
            render_plan=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)
        time.sleep(0.3)

        # scene_srt and render_plan should still run even though beat_json failed
        mock_scene_srt.assert_called_once()
        mock_render_plan.assert_called_once()

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_partial_failure_reports_errors(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        mock_beat.side_effect = RuntimeError("beat rebuild failed")
        mock_render_plan.side_effect = RuntimeError("render plan rebuild failed")
        mock_scene_srt.return_value = {"status": "ok"}

        registry = JobRegistry()
        affected = AffectedArtifacts(
            beat_json=True,
            scene_srt=True,
            render_plan=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)

        time.sleep(0.3)

        # The job status should reflect partial success
        job = registry.get(registry.list()[0]["id"])
        logs = "\n".join(job.get("logs", []))
        self.assertIn("beat_json", logs)
        self.assertIn("scene_srt", logs)
        self.assertIn("render_plan", logs)

    @patch("feverslop.studio.services.rebuild_beat_json")
    @patch("feverslop.studio.services.rebuild_scene_srt")
    @patch("feverslop.studio.services.rebuild_stage1_segments")
    @patch("feverslop.studio.services.rebuild_ltx_prompt")
    @patch("feverslop.studio.services.rebuild_render_plan")
    def test_no_affected_artifacts_produces_empty_success(
        self, mock_render_plan, mock_ltx_prompt, mock_stage1, mock_scene_srt, mock_beat
    ):
        from feverslop.ports.timeline_documents import AffectedArtifacts
        from feverslop.studio.jobs import JobRegistry

        registry = JobRegistry()
        affected = AffectedArtifacts()
        with tempfile.TemporaryDirectory() as tmpdir:
            registry.add_rebuild_plan_timeline(tmpdir, affected)
        time.sleep(0.3)

        mock_beat.assert_not_called()
        mock_scene_srt.assert_not_called()
        mock_stage1.assert_not_called()
        mock_ltx_prompt.assert_not_called()
        mock_render_plan.assert_not_called()


class RebuildServicesExistTests(unittest.TestCase):
    """Verify that the rebuild service functions exist and are importable."""

    def test_rebuild_beat_json_exists(self):
        from feverslop.studio import services
        self.assertTrue(hasattr(services, "rebuild_beat_json"))
        self.assertTrue(callable(services.rebuild_beat_json))

    def test_rebuild_scene_srt_exists(self):
        from feverslop.studio import services
        self.assertTrue(hasattr(services, "rebuild_scene_srt"))
        self.assertTrue(callable(services.rebuild_scene_srt))

    def test_rebuild_stage1_segments_exists(self):
        from feverslop.studio import services
        self.assertTrue(hasattr(services, "rebuild_stage1_segments"))
        self.assertTrue(callable(services.rebuild_stage1_segments))

    def test_rebuild_ltx_prompt_exists(self):
        from feverslop.studio import services
        self.assertTrue(hasattr(services, "rebuild_ltx_prompt"))
        self.assertTrue(callable(services.rebuild_ltx_prompt))

    def test_rebuild_render_plan_exists(self):
        from feverslop.studio import services
        self.assertTrue(hasattr(services, "rebuild_render_plan"))
        self.assertTrue(callable(services.rebuild_render_plan))


if __name__ == "__main__":
    unittest.main()
