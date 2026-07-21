from __future__ import annotations

import unittest

from feverslop.ports.scene_documents import SceneLtxPromptField
from feverslop.studio.job_service import StudioJobRequest
from feverslop.studio.scene_workspace_service import (
    SceneWorkspaceService,
    normalize_scene_numbers,
)


class RecordingLoadWorkspace:
    def __init__(self) -> None:
        self.project_ids: list[str] = []

    def execute(self, project_id: str) -> object:
        self.project_ids.append(project_id)
        return {"project_id": project_id}


class RecordingPatchScene:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return {"revision": "revision-2"}


class RecordingJobService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, StudioJobRequest]] = []

    def start_job(self, project_id: str, request: StudioJobRequest) -> object:
        self.calls.append((project_id, request))
        return {"id": "job-1", "action": request.action}


class SceneWorkspaceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = RecordingLoadWorkspace()
        self.patcher = RecordingPatchScene()
        self.jobs = RecordingJobService()
        self.service = SceneWorkspaceService(
            load_workspace=self.loader,
            patch_scene=self.patcher,
            jobs=self.jobs,
        )

    def test_selection_is_deduplicated_and_sorted(self):
        self.assertEqual((2, 4, 7), normalize_scene_numbers((4, 2, 7, 4, 2)))

    def test_selection_rejects_bool_nonpositive_and_malformed_values(self):
        invalid_selections = ((True,), (0,), (-1,), ("2",), (2.5,), (None,))

        for selection in invalid_selections:
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    normalize_scene_numbers(selection)  # type: ignore[arg-type]

    def test_load_forwards_to_application_use_case(self):
        result = self.service.load("demo")

        self.assertEqual({"project_id": "demo"}, result)
        self.assertEqual(["demo"], self.loader.project_ids)

    def test_patch_forwards_canonical_fields_revision_and_selected_enum(self):
        changes = {
            "shot_description": "Tracking shot",
            "z_image.prompt": "Blue stage",
            "ltx.i2v_prompt_from_t2i": "Fast dolly",
        }

        result = self.service.patch_scene(
            project_id="demo",
            scene_number=4,
            changes=changes,
            expected_revision="revision-1",
            selected_ltx_prompt_field=SceneLtxPromptField.I2V_PROMPT_FROM_T2I,
        )

        self.assertEqual({"revision": "revision-2"}, result)
        self.assertEqual(
            [
                {
                    "project_id": "demo",
                    "scene_number": 4,
                    "changes": changes,
                    "expected_revision": "revision-1",
                    "selected_ltx_prompt_field": SceneLtxPromptField.I2V_PROMPT_FROM_T2I,
                }
            ],
            self.patcher.calls,
        )

    def test_render_intentions_start_registered_action_for_selected_scenes(self):
        for intention in ("render", "rerender", "retake", "ltx-render"):
            with self.subTest(intention=intention):
                result = self.service.start_action(
                    project_id="demo",
                    action=intention,
                    scene_numbers=(4, 2, 4),
                )

                self.assertEqual("ltx-render-scenes", result["action"])
                project_id, request = self.jobs.calls[-1]
                self.assertEqual("demo", project_id)
                self.assertEqual(
                    StudioJobRequest(action="ltx-render-scenes", scenes=[2, 4]),
                    request,
                )

    def test_selection_is_transient_between_action_requests(self):
        self.service.start_action(
            project_id="demo",
            action="render",
            scene_numbers=(2, 4),
        )

        with self.assertRaisesRegex(ValueError, "at least one scene"):
            self.service.start_action(
                project_id="demo",
                action="render",
                scene_numbers=(),
            )

        self.assertEqual(1, len(self.jobs.calls))

    def test_unknown_ui_action_is_rejected_before_job_service(self):
        with self.assertRaisesRegex(ValueError, "Unknown scene action"):
            self.service.start_action(
                project_id="demo",
                action="shell-command",
                scene_numbers=(2,),
            )

        self.assertEqual([], self.jobs.calls)

    def test_stage_one_preview_is_clearly_rejected_until_registered(self):
        requests = (
            {"action": "stage-1-preview", "preview_stage": 1},
            {"action": "ltx-render", "preview_stage": 1},
        )

        for request in requests:
            with self.subTest(request=request):
                with self.assertRaisesRegex(ValueError, "Stage-1 preview is unavailable"):
                    self.service.start_action(
                        project_id="demo",
                        scene_numbers=(2, 4),
                        **request,
                    )

        self.assertEqual([], self.jobs.calls)


if __name__ == "__main__":
    unittest.main()
