import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feverslop.domain.visual_consistency import PreflightMode
from feverslop.studio.job_service import (
    StudioJobRequest,
    StudioJobService,
    VisualConsistencyPreflightAction,
)
from feverslop.studio.jobs import JobRegistry


class _Store:
    def __init__(self, root: Path):
        self.root = root

    def project_metadata(self, project_id):
        return {"project_type": "standard_music_video"}

    def resolve_project_path(self, project_id, relative_path):
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("outside project")
        return path


class StudioVisualConsistencyJobTests(unittest.TestCase):
    def test_read_only_action_is_registered_with_typed_options(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "config.json").write_text(
                json.dumps({"input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            plan = project / "plan.json"
            plan.write_text(json.dumps([{"scene": 1, "prompt": "legacy"}]), encoding="utf-8")
            service = StudioJobService(store=_Store(project), jobs=JobRegistry())
            request = StudioJobRequest(
                action="visual-consistency-preflight",
                plan="plan.json",
                visual_consistency_mode="ingredients",
                preflight_mode="warn",
            )

            action = service.registry.resolve(request.action)
            before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
            with patch(
                "feverslop.studio.jobs.subprocess.run"
            ) as subprocess_run, patch(
                "feverslop.studio.jobs.pipeline_runner.run"
            ) as gpu_pipeline:
                result = action.build("demo", request, {})(lambda _message: None)
            after = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}

        self.assertIsInstance(action, VisualConsistencyPreflightAction)
        self.assertTrue(result["renderable"])
        self.assertEqual(before, after)
        self.assertEqual("legacy_contract_unknown", result["issues"][0]["code"])
        subprocess_run.assert_not_called()
        gpu_pipeline.assert_not_called()

    def test_request_normalizes_and_validates_typed_preflight_mode(self):
        request = StudioJobRequest(
            action="visual-consistency-preflight",
            preflight_mode="off",
        )

        self.assertIs(PreflightMode.OFF, request.preflight_mode)
        with self.assertRaisesRegex(ValueError, "strict, warn, or off"):
            StudioJobRequest(
                action="visual-consistency-preflight",
                preflight_mode="invalid",
            )

    def test_action_rejects_invalid_consistency_mode_and_outside_plan(self):
        with TemporaryDirectory() as tmp:
            service = StudioJobService(
                store=_Store(Path(tmp)),
                jobs=JobRegistry(),
            )
            with self.assertRaisesRegex(ValueError, "ingredients, msr, or i2v"):
                service.registry.resolve("visual-consistency-preflight").build(
                    "demo",
                    StudioJobRequest(
                        action="visual-consistency-preflight",
                        visual_consistency_mode="invalid",
                    ),
                    {},
                )
            with self.assertRaisesRegex(ValueError, "outside project"):
                service.registry.resolve("visual-consistency-preflight").build(
                    "demo",
                    StudioJobRequest(
                        action="visual-consistency-preflight",
                        plan="../outside.json",
                    ),
                    {},
                )

    def test_off_action_bypasses_config_and_manifest_and_is_read_only(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "config.json").write_text("{}", encoding="utf-8")
            plan = project / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            manifest = project / "output" / "references" / "actors" / "bad" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{bad", encoding="utf-8")
            service = StudioJobService(store=_Store(project), jobs=JobRegistry())
            request = StudioJobRequest(
                action="visual-consistency-preflight",
                plan="plan.json",
                preflight_mode=PreflightMode.OFF,
            )
            action = service.registry.resolve(request.action)
            before = {
                path.relative_to(project).as_posix(): path.read_bytes()
                for path in project.rglob("*")
                if path.is_file()
            }

            with patch(
                "feverslop.studio.jobs.ProjectConfig.load"
            ) as config_load, patch(
                "feverslop.studio.jobs.ProjectReferenceManifestAdapter"
            ) as manifest_adapter, patch(
                "feverslop.studio.jobs.subprocess.run"
            ) as subprocess_run:
                result = action.build("demo", request, {})(lambda _message: None)
            after = {
                path.relative_to(project).as_posix(): path.read_bytes()
                for path in project.rglob("*")
                if path.is_file()
            }

        self.assertTrue(result["renderable"])
        self.assertEqual([], result["contracts"])
        self.assertEqual([], result["issues"])
        self.assertEqual(before, after)
        config_load.assert_not_called()
        manifest_adapter.assert_not_called()
        subprocess_run.assert_not_called()

    def test_strict_validation_job_is_failed_with_structured_payload(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "config.json").write_text(
                json.dumps({"input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            plan = project / "plan.json"
            plan.write_text(json.dumps([{
                "scene": 1,
                "references": {"actor_ids": ["missing"]},
            }]), encoding="utf-8")
            service = StudioJobService(store=_Store(project), jobs=JobRegistry())

            started = service.start_job(
                "demo",
                StudioJobRequest(
                    action="visual-consistency-preflight",
                    plan="plan.json",
                    visual_consistency_mode="ingredients",
                    preflight_mode=PreflightMode.STRICT,
                ),
            )
            for _ in range(100):
                job = service.jobs.get(started["id"])
                if job["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.005)

        self.assertEqual("failed", job["status"])
        self.assertIsInstance(job["result"], dict)
        self.assertEqual(
            {"renderable", "contracts", "issues"},
            set(job["result"]),
        )
        self.assertFalse(job["result"]["renderable"])
        self.assertEqual(
            "missing_actor_reference",
            job["result"]["issues"][0]["code"],
        )
        structured_logs = []
        for line in job["logs"]:
            try:
                structured_logs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.assertTrue(
            any(
                payload.get("renderable") is False
                for payload in structured_logs
                if isinstance(payload, dict)
            )
        )


if __name__ == "__main__":
    unittest.main()
