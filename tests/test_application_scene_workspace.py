from __future__ import annotations

import unittest

from feverslop.application.scene_workspace import (
    SceneWorkspaceService,
    normalize_scene_numbers,
)
from feverslop.application.job_contracts import JobRequest


class _RecordingJobs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, JobRequest]] = []

    def start_job(self, project_id: str, request: JobRequest) -> object:
        self.calls.append((project_id, request))
        return {"id": "job-1", "action": request.action}


class ApplicationSceneWorkspaceTests(unittest.TestCase):
    def test_scene_action_uses_application_job_request(self):
        jobs = _RecordingJobs()
        service = SceneWorkspaceService(
            load_workspace=object(),
            patch_scene=object(),
            jobs=jobs,
        )

        result = service.start_action(
            project_id="demo",
            action="render",
            scene_numbers=(4, 2, 4),
        )

        self.assertEqual({"id": "job-1", "action": "ltx-render-scenes"}, result)
        self.assertEqual(
            ("demo", JobRequest(action="ltx-render-scenes", scenes=[2, 4])),
            jobs.calls[-1],
        )

    def test_scene_number_normalization_remains_canonical(self):
        self.assertEqual((2, 4, 7), normalize_scene_numbers((4, 2, 7, 4, 2)))


if __name__ == "__main__":
    unittest.main()
