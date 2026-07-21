from __future__ import annotations

import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class StudioDesktopCompositionTests(unittest.TestCase):
    def test_web_server_adapter_is_removed(self):
        self.assertIsNone(importlib.util.find_spec("feverslop.studio.server"))

    def test_parse_args_uses_requested_projects_root(self):
        from feverslop.studio.desktop.app import parse_args

        args = parse_args(["--projects-root", "/tmp/feverslop-projects"])

        self.assertEqual(args.projects_root, Path("/tmp/feverslop-projects"))

    def test_create_context_wires_existing_studio_services(self):
        from feverslop.studio.desktop.composition import create_studio_context
        from feverslop.studio.job_service import StudioJobService
        from feverslop.studio.jobs import JobRegistry
        from feverslop.studio.projects import ProjectStore
        from feverslop.studio.scene_workspace_service import SceneWorkspaceService

        with tempfile.TemporaryDirectory() as temp_dir:
            context = create_studio_context(Path(temp_dir))

        self.assertIsInstance(context.store, ProjectStore)
        self.assertIsInstance(context.jobs, JobRegistry)
        self.assertIsInstance(context.job_service, StudioJobService)
        self.assertIsInstance(context.scene_service, SceneWorkspaceService)


class SceneWorkspaceViewModelTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtGui import QGuiApplication

        self.app = QGuiApplication.instance() or QGuiApplication([])

    @staticmethod
    def _snapshot(*scenes, revision="revision-1"):
        from feverslop.application.scene_workspace import SceneWorkspaceSnapshot
        from feverslop.domain.scene_workspace import SceneMedia, SceneWorkspace, SceneWorkspaceItem

        items = []
        for scene in scenes:
            values = dict(scene)
            thumbnail_path = values.pop("thumbnail_path", None)
            items.append(
                SceneWorkspaceItem(
                    media=SceneMedia(thumbnail_path=thumbnail_path),
                    **values,
                )
            )
        return SceneWorkspaceSnapshot(
            workspace=SceneWorkspace(tuple(items)),
            revision=revision,
        )

    def test_scene_list_model_exposes_explicit_card_and_inspector_roles(self):
        from PySide6.QtCore import Qt

        from feverslop.studio.desktop.viewmodels.scenes import SceneListModel

        model = SceneListModel(
            thumbnail_url=lambda path: f"file:///project/{path}",
        )
        model.replace(
            self._snapshot(
                {
                    "scene_number": 7,
                    "start_seconds": 1.25,
                    "end_seconds": 4.75,
                    "performance_state": "performance",
                    "shot_description": "Tracking shot",
                    "image_prompt": "Blue stage",
                    "video_prompt": "Fast dolly",
                    "reference_ids": ("hero",),
                    "thumbnail_path": "output/render/storyboard/scene_0007.png",
                }
            ).workspace.items
        )

        roles = {bytes(name).decode(): role for role, name in model.roleNames().items()}
        index = model.index(0, 0)

        self.assertEqual(
            {
                "sceneNumber",
                "startSeconds",
                "endSeconds",
                "performanceState",
                "status",
                "thumbnailUrl",
                "shotDescription",
                "imagePrompt",
                "videoPrompt",
                "videoPromptField",
                "referenceIds",
                "selected",
            },
            set(roles),
        )
        self.assertEqual(7, model.data(index, roles["sceneNumber"]))
        self.assertEqual("file:///project/output/render/storyboard/scene_0007.png", model.data(index, roles["thumbnailUrl"]))
        self.assertIsNone(model.data(index, Qt.ItemDataRole.DisplayRole))

    def test_project_signal_reloads_and_switch_resets_transient_state(self):
        from PySide6.QtCore import QObject, Signal

        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Studio(QObject):
            currentProjectChanged = Signal()

            def __init__(self):
                super().__init__()
                self.current_project_id = "alpha"

        class Service:
            def load(self, project_id):
                number = 1 if project_id == "alpha" else 2
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": number, "shot_description": project_id}
                )

        studio = Studio()
        view_model = SceneWorkspaceViewModel(service=Service(), studio_view_model=studio)
        view_model.reload()
        view_model.toggleSelection(1)

        studio.current_project_id = "beta"
        studio.currentProjectChanged.emit()

        self.assertEqual("beta", view_model.current_project_id)
        self.assertEqual(2, view_model.scenes.data(view_model.scenes.index(0, 0), view_model.scenes.SceneNumberRole))
        self.assertEqual([], view_model.selected_scene_numbers)
        self.assertFalse(view_model.dirty)
        self.assertFalse(view_model.conflict)

        studio.current_project_id = ""
        studio.currentProjectChanged.emit()
        self.assertEqual(0, view_model.scenes.rowCount())

    def test_selection_is_toggled_and_published_in_sorted_order(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        service = SimpleNamespace(load=lambda _project_id: self._snapshot(
            {"scene_number": 5}, {"scene_number": 2}
        ))
        studio = SimpleNamespace(current_project_id="demo")
        view_model = SceneWorkspaceViewModel(service=service, studio_view_model=studio)
        view_model.reload()

        self.assertTrue(view_model.toggleSelection(5))
        self.assertTrue(view_model.toggleSelection(2))
        self.assertEqual([2, 5], view_model.selected_scene_numbers)
        self.assertTrue(view_model.scenes.data(view_model.scenes.index(0, 0), view_model.scenes.SelectedRole))
        self.assertTrue(view_model.toggleSelection(5))
        self.assertEqual([2], view_model.selected_scene_numbers)
        self.assertFalse(view_model.toggleSelection(99))

    def test_successful_prompt_save_forwards_revision_and_clears_dirty_state(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        calls = []

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 3, "shot_description": "Old"}
                )

            def patch_scene(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(revision="revision-2")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()

        saved = view_model.savePromptFields(
            3,
            {"shotDescription": "New", "imagePrompt": "Still", "videoPrompt": "Move"},
            "i2v_prompt_from_t2i",
        )

        self.assertTrue(saved)
        self.assertEqual("revision-1", calls[0]["expected_revision"])
        self.assertEqual(
            {
                "shot_description": "New",
                "z_image.prompt": "Still",
                "ltx.i2v_prompt_from_t2i": "Move",
            },
            calls[0]["changes"],
        )
        self.assertEqual("revision-2", view_model.revision)
        self.assertFalse(view_model.dirty)
        self.assertFalse(view_model.conflict)
        self.assertEqual("", view_model.error)

    def test_save_conflict_preserves_local_edits_and_exposes_state(self):
        from feverslop.ports.scene_documents import SceneDocumentConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 3, "shot_description": "Old"}
                )

            def patch_scene(self, **_kwargs):
                raise SceneDocumentConflict("demo", "revision-1", "revision-2")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()

        self.assertFalse(view_model.savePromptFields(3, {"shotDescription": "Unsaved"}, ""))

        self.assertTrue(view_model.dirty)
        self.assertTrue(view_model.conflict)
        self.assertIn("changed", view_model.error)
        self.assertEqual(
            "Unsaved",
            view_model.scenes.data(view_model.scenes.index(0, 0), view_model.scenes.ShotDescriptionRole),
        )

    def test_discard_local_edits_restores_baseline_without_loading(self):
        from feverslop.ports.scene_documents import SceneDocumentConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        load_calls = []

        class Service:
            def load(self, project_id):
                load_calls.append(project_id)
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 3, "shot_description": "Confirmed"}
                )

            def patch_scene(self, **kwargs):
                raise SceneDocumentConflict("demo", kwargs["expected_revision"], "external")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(3)
        view_model.savePromptFields(3, {"shotDescription": "Local"}, "")

        self.assertTrue(hasattr(view_model, "discardLocalEdits"))
        view_model.discardLocalEdits()

        self.assertEqual(["demo"], load_calls)
        self.assertEqual("Confirmed", view_model.inspectedScene["shotDescription"])
        self.assertEqual([3], view_model.selected_scene_numbers)
        self.assertEqual("revision-1", view_model.revision)
        self.assertFalse(view_model.dirty)
        self.assertFalse(view_model.conflict)
        self.assertEqual("", view_model.error)

    def test_reload_after_conflict_reads_new_external_values(self):
        from feverslop.ports.scene_documents import SceneDocumentConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        load_calls = []

        class Service:
            def load(self, project_id):
                load_calls.append(project_id)
                description = "Original" if len(load_calls) == 1 else "External"
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 2, "shot_description": description},
                    revision=f"revision-{len(load_calls)}",
                )

            def patch_scene(self, **kwargs):
                raise SceneDocumentConflict("demo", kwargs["expected_revision"], "external")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(2)
        view_model.savePromptFields(2, {"shotDescription": "Local"}, "")

        self.assertTrue(view_model.reload())

        self.assertEqual(["demo", "demo"], load_calls)
        self.assertEqual("External", view_model.inspectedScene["shotDescription"])
        self.assertEqual("revision-2", view_model.revision)
        self.assertFalse(view_model.dirty)

    def test_discard_preserves_an_earlier_successful_save(self):
        from feverslop.ports.scene_documents import SceneDocumentConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        attempts = []

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 5, "shot_description": "Original"}
                )

            def patch_scene(self, **kwargs):
                attempts.append(kwargs)
                if len(attempts) == 1:
                    return SimpleNamespace(revision="revision-2")
                raise SceneDocumentConflict("demo", kwargs["expected_revision"], "external")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(5)
        self.assertTrue(view_model.savePromptFields(5, {"shotDescription": "Saved"}, ""))
        self.assertFalse(view_model.savePromptFields(5, {"shotDescription": "Failed"}, ""))

        self.assertTrue(hasattr(view_model, "discardLocalEdits"))
        view_model.discardLocalEdits()

        self.assertEqual("Saved", view_model.inspectedScene["shotDescription"])
        self.assertEqual("revision-2", view_model.revision)
        self.assertFalse(view_model.dirty)
        self.assertFalse(view_model.conflict)

    def test_selected_action_forwards_sorted_scenes_and_preview_stage(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        calls = []

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 8}, {"scene_number": 3}
                )

            def start_action(self, **kwargs):
                calls.append(kwargs)
                return {}

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(8)
        view_model.toggleSelection(3)

        self.assertTrue(view_model.startSelectedAction("ltx-render", 1))
        self.assertEqual(
            [{
                "project_id": "demo",
                "action": "ltx-render",
                "scene_numbers": (3, 8),
                "preview_stage": 1,
            }],
            calls,
        )

    def test_invalid_prompt_patch_does_not_mutate_model_or_dirty_state(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 1, "shot_description": "Original", "video_prompt": "Old"}
                )

            def patch_scene(self, **_kwargs):
                raise AssertionError("invalid patch must not reach service")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(1)

        invalid_requests = (
            ({}, ""),
            ({"unknown": "value"}, ""),
            ({"videoPrompt": "New"}, "not-a-prompt-field"),
        )
        for fields, prompt_field in invalid_requests:
            with self.subTest(fields=fields, prompt_field=prompt_field):
                self.assertFalse(view_model.savePromptFields(1, fields, prompt_field))
                self.assertFalse(view_model.dirty)
                self.assertFalse(view_model.conflict)
                self.assertEqual("Original", view_model.inspectedScene["shotDescription"])
                self.assertEqual(
                    "Old",
                    view_model.scenes.data(
                        view_model.scenes.index(0, 0),
                        view_model.scenes.VideoPromptRole,
                    ),
                )

    def test_pending_and_conflicts_are_granular_across_scene_saves(self):
        from feverslop.ports.scene_documents import SceneDocumentConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        attempts = []

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 1, "shot_description": "One"},
                    {"scene_number": 2, "shot_description": "Two"},
                )

            def patch_scene(self, **kwargs):
                attempts.append(kwargs)
                if len(attempts) in {1, 3}:
                    raise SceneDocumentConflict("demo", kwargs["expected_revision"], "external")
                return SimpleNamespace(revision=f"revision-{len(attempts) + 1}")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()

        self.assertFalse(view_model.savePromptFields(1, {"shotDescription": "One local"}, ""))
        self.assertTrue(view_model.dirty)
        self.assertTrue(view_model.conflict)

        self.assertTrue(view_model.savePromptFields(2, {"shotDescription": "Two saved"}, ""))
        self.assertTrue(view_model.dirty)
        self.assertTrue(view_model.conflict)

        self.assertFalse(view_model.savePromptFields(1, {"shotDescription": "One local"}, ""))
        self.assertTrue(view_model.conflict)
        self.assertTrue(view_model.savePromptFields(1, {"shotDescription": "One local"}, ""))
        self.assertFalse(view_model.dirty)
        self.assertFalse(view_model.conflict)

    def test_same_project_reload_failure_preserves_workspace_and_transient_state(self):
        from feverslop.ports.scene_documents import SceneDocumentConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Service:
            fail_reload = False

            def load(self, _project_id):
                if self.fail_reload:
                    raise OSError("temporarily unavailable")
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 4})

            def patch_scene(self, **kwargs):
                raise SceneDocumentConflict("demo", kwargs["expected_revision"], "external")

        service = Service()
        view_model = SceneWorkspaceViewModel(
            service=service,
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(4)
        view_model.savePromptFields(4, {"shotDescription": "Local"}, "")
        service.fail_reload = True

        self.assertFalse(view_model.reload())
        self.assertEqual(1, view_model.scenes.rowCount())
        self.assertEqual([4], view_model.selected_scene_numbers)
        self.assertEqual("revision-1", view_model.revision)
        self.assertTrue(view_model.dirty)
        self.assertTrue(view_model.conflict)

    def test_project_switch_load_failure_clears_old_project_state(self):
        from PySide6.QtCore import QObject, Signal

        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Studio(QObject):
            currentProjectChanged = Signal()

            def __init__(self):
                super().__init__()
                self.current_project_id = "alpha"

        class Service:
            def load(self, project_id):
                if project_id == "beta":
                    raise FileNotFoundError("no render plan")
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 1})

        studio = Studio()
        view_model = SceneWorkspaceViewModel(service=Service(), studio_view_model=studio)
        view_model.reload()
        view_model.toggleSelection(1)
        studio.current_project_id = "beta"

        studio.currentProjectChanged.emit()

        self.assertEqual("beta", view_model.current_project_id)
        self.assertEqual(0, view_model.scenes.rowCount())
        self.assertEqual([], view_model.selected_scene_numbers)
        self.assertEqual("", view_model.revision)
        self.assertFalse(view_model.dirty)
        self.assertFalse(view_model.conflict)
        self.assertIn("render plan", view_model.error)

    def test_inspected_scene_tracks_last_toggle_and_notifies_for_edits(self):
        from PySide6.QtTest import QSignalSpy

        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {
                        "scene_number": 5,
                        "shot_description": "Five",
                        "thumbnail_path": "output/render/storyboard/scene_0005.png",
                    },
                    {"scene_number": 2, "shot_description": "Two"},
                )

            def patch_scene(self, **_kwargs):
                return SimpleNamespace(revision="revision-2")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
            thumbnail_url=lambda project_id, path: f"file:///{project_id}/{path}",
        )
        spy = QSignalSpy(view_model.inspectedSceneChanged)
        view_model.reload()
        self.assertEqual({}, view_model.inspectedScene)

        view_model.toggleSelection(5)
        self.assertEqual(5, view_model.inspectedScene["sceneNumber"])
        self.assertEqual("file:///demo/output/render/storyboard/scene_0005.png", view_model.inspectedScene["thumbnailUrl"])
        self.assertTrue(view_model.inspectedScene["selected"])

        view_model.toggleSelection(2)
        self.assertEqual(2, view_model.currentScene["sceneNumber"])
        view_model.toggleSelection(2)
        self.assertEqual(5, view_model.inspectedScene["sceneNumber"])

        before_edit = spy.count()
        view_model.savePromptFields(5, {"shotDescription": "Edited"}, "")
        self.assertGreater(spy.count(), before_edit)
        self.assertEqual("Edited", view_model.inspectedScene["shotDescription"])
        view_model.toggleSelection(5)
        self.assertEqual({}, view_model.inspectedScene)

    def test_video_prompt_field_reports_priority_provenance(self):
        from feverslop.application.scene_workspace import SceneWorkspaceSnapshot
        from feverslop.domain.scene_workspace import SceneWorkspace
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        raw_scenes = (
            {"scene": 1, "ltx": {"base_prompt": "Base"}},
            {"scene": 2, "ltx": {"base_prompt": "Base", "i2v_prompt_from_t2i": "I2V"}},
            {
                "scene": 3,
                "ltx": {
                    "base_prompt": "Base",
                    "i2v_prompt_from_t2i": "I2V",
                    "original_style_i2v_prompt": "Original",
                },
            },
            {"scene": 4},
        )
        snapshot = SceneWorkspaceSnapshot(
            workspace=SceneWorkspace.from_scenes(raw_scenes),
            revision="revision-1",
        )
        view_model = SceneWorkspaceViewModel(
            service=SimpleNamespace(
                load=lambda _project_id: snapshot,
                patch_scene=lambda **_kwargs: SimpleNamespace(revision="revision-2"),
            ),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()

        fields = [
            view_model.scenes.data(view_model.scenes.index(row, 0), view_model.scenes.VideoPromptFieldRole)
            for row in range(4)
        ]
        self.assertEqual(
            ["base_prompt", "i2v_prompt_from_t2i", "original_style_i2v_prompt", ""],
            fields,
        )
        view_model.toggleSelection(3)
        self.assertEqual("original_style_i2v_prompt", view_model.inspectedScene["videoPromptField"])

        view_model.toggleSelection(1)
        self.assertTrue(
            view_model.savePromptFields(
                1,
                {"videoPrompt": "New I2V"},
                "i2v_prompt_from_t2i",
            )
        )
        self.assertEqual("i2v_prompt_from_t2i", view_model.inspectedScene["videoPromptField"])

    def test_selected_video_prompt_field_stays_coherent_across_save_and_reload(self):
        from PySide6.QtTest import QSignalSpy

        from feverslop.application.scene_workspace import SceneWorkspaceSnapshot
        from feverslop.domain.scene_workspace import SceneWorkspace
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        current_scene = {
            "scene": 1,
            "ltx": {
                "original_style_i2v_prompt": "Original priority prompt",
                "i2v_prompt_from_t2i": "Old lower prompt",
            },
        }

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceSnapshot(
                    workspace=SceneWorkspace.from_scenes([current_scene]),
                    revision="revision-2",
                )

            def patch_scene(self, **kwargs):
                current_scene["ltx"]["i2v_prompt_from_t2i"] = kwargs["changes"][
                    "ltx.i2v_prompt_from_t2i"
                ]
                return SimpleNamespace(revision="revision-2")

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(1)
        changed = QSignalSpy(view_model.scenes.dataChanged)

        self.assertTrue(
            view_model.savePromptFields(
                1,
                {"videoPrompt": "New lower prompt"},
                "i2v_prompt_from_t2i",
            )
        )

        index = view_model.scenes.index(0, 0)
        self.assertEqual(
            "New lower prompt",
            view_model.scenes.data(index, view_model.scenes.VideoPromptRole),
        )
        self.assertEqual(
            "i2v_prompt_from_t2i",
            view_model.scenes.data(index, view_model.scenes.VideoPromptFieldRole),
        )
        self.assertEqual("New lower prompt", view_model.inspectedScene["videoPrompt"])
        self.assertEqual("i2v_prompt_from_t2i", view_model.inspectedScene["videoPromptField"])
        changed_roles = list(changed.at(changed.count() - 1)[2])
        self.assertIn(view_model.scenes.VideoPromptRole, changed_roles)
        self.assertIn(view_model.scenes.VideoPromptFieldRole, changed_roles)

        self.assertTrue(view_model.reload())
        self.assertEqual("New lower prompt", view_model.inspectedScene["videoPrompt"])
        self.assertEqual("i2v_prompt_from_t2i", view_model.inspectedScene["videoPromptField"])


class StudioViewModelTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtGui import QGuiApplication

        self.app = QGuiApplication.instance() or QGuiApplication([])

    def test_refresh_and_select_project_exposes_native_state(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def list_projects(self):
                return [{"id": "scholoraid", "name": "Scholoraid", "status": {}, "artifacts": {}}]

            def describe_project(self, project_id):
                return {"id": project_id, "name": "Scholoraid", "status": {}, "artifacts": {}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.refresh_projects()
        view_model.select_project("scholoraid")

        self.assertEqual(view_model.projects[0]["id"], "scholoraid")
        self.assertEqual(view_model.current_project["name"], "Scholoraid")
        self.assertEqual(view_model.current_project_id, "scholoraid")

    def test_save_json_artifact_uses_structured_parser(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        calls = []

        class Store:
            def write_artifact(self, project_id, request):
                calls.append((project_id, request.path, request.data))
                return {"path": request.path, "data": request.data}

            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        self.assertTrue(view_model.save_json_artifact("config.json", '{"fps": 24}'))
        self.assertEqual(calls, [("scholoraid", "config.json", {"fps": 24})])

    def test_save_json_artifact_reports_invalid_json_without_writing(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def write_artifact(self, project_id, request):
                raise AssertionError("must not write invalid JSON")

            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        self.assertFalse(view_model.save_json_artifact("config.json", "{"))
        self.assertIn("JSON", view_model.error)

    def test_load_json_artifact_formats_content_for_editor(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def read_artifact(self, project_id, path):
                return {"path": path, "data": {"scene": 5, "prompt": "gate"}}

            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")
        view_model.load_json_artifact("render_plan.json")

        self.assertEqual(json.loads(view_model.editor_text), {"scene": 5, "prompt": "gate"})
        self.assertEqual(view_model.editor_path, "render_plan.json")

    def test_loaded_render_plan_exposes_scene_records(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def read_artifact(self, project_id, path):
                return {"path": path, "data": [{"scene": 1, "prompt": "Gate"}, {"scene": 3, "prompt": "Hall"}]}

            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "name": project_id,
                    "status": {},
                    "artifacts": {"render_plans": ["output/render/plans/msr.json"]},
                }

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")
        view_model.load_json_artifact("output/render/plans/msr.json")

        self.assertEqual([scene["scene"] for scene in view_model.editor_scenes], [1, 3])
        self.assertEqual(view_model.preferred_artifact("render_plans"), "output/render/plans/msr.json")

    def test_create_project_maps_qml_payload_to_domain_request(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        requests = []

        class Store:
            def create_project(self, request):
                requests.append(request)
                return {"id": "new-film", "name": request.name, "status": {}, "artifacts": {}}

            def list_projects(self):
                return [{"id": "new-film", "name": "New Film", "status": {}, "artifacts": {}}]

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())

        project_id = view_model.create_project(
            {"project_type": "movie", "name": "New Film", "story_text": "A locked gate.", "desired_length": 45}
        )

        self.assertEqual(project_id, "new-film")
        self.assertEqual(requests[0].project_type, "movie")
        self.assertEqual(requests[0].story_text, "A locked gate.")
        self.assertEqual(requests[0].desired_length, 45)

    def test_start_job_and_refresh_jobs_exposes_logs(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        requests = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        class Jobs:
            def list(self, project_id):
                return [{"id": "job-1", "project_id": project_id, "status": "running", "logs": ["Planning", "Rendering"]}]

        class Service:
            def start_job(self, project_id, request):
                requests.append((project_id, request))
                return {"id": "job-1", "status": "queued", "logs": []}

        view_model = StudioViewModel(store=Store(), jobs=Jobs(), job_service=Service())
        view_model.select_project("scholoraid")

        self.assertTrue(view_model.start_job("movie-render", [1, 3, 5]))
        view_model.refresh_jobs()

        self.assertEqual(requests[0][1].scenes, [1, 3, 5])
        self.assertEqual(view_model.jobs[0]["status"], "running")
        self.assertEqual(view_model.job_logs, "Planning\nRendering")

    def test_patch_render_scene_uses_structured_patch(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        patches = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

            def patch_render_plan(self, project_id, patch):
                patches.append((project_id, patch))
                return {"scene": {"scene": patch.scene, **patch.updates}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        self.assertTrue(view_model.patch_render_scene("render_plan_msr.json", 5, {"prompt": "The party reaches the gate."}))
        self.assertEqual(patches[0][1].scene, 5)
        self.assertEqual(patches[0][1].updates["prompt"], "The party reaches the gate.")

    def test_media_url_only_resolves_project_media(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

            def resolve_media_path(self, project_id, path):
                self.seen = (project_id, path)
                return Path("/tmp/final.mp4")

        store = Store()
        view_model = StudioViewModel(store=store, jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        url = view_model.media_url("output/final.mp4")

        self.assertEqual(store.seen, ("scholoraid", "output/final.mp4"))
        self.assertTrue(url.startswith("file:"))

    def test_artifact_entries_flatten_catalog_with_media_kind(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "name": project_id,
                    "status": {},
                    "artifacts": {
                        "generated_json": ["movie/render_plan.json"],
                        "images": ["references/hero.png"],
                        "videos": ["output/final.mp4"],
                    },
                }

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        entries = view_model.artifact_entries

        self.assertEqual([entry["path"] for entry in entries], [
            "movie/render_plan.json",
            "references/hero.png",
            "output/final.mp4",
        ])
        self.assertEqual(entries[1]["kind"], "image")
        self.assertEqual(entries[2]["kind"], "video")

    def test_start_recut_builds_specific_job_request(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        requests = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        class Jobs:
            def list(self, project_id):
                return []

        class Service:
            def start_job(self, project_id, request):
                requests.append(request)
                return {}

        view_model = StudioViewModel(store=Store(), jobs=Jobs(), job_service=Service())
        view_model.select_project("scholoraid")

        self.assertTrue(view_model.start_recut("raw/scene_0005.mp4", "scenes/scene_0005.mp4", 1.25, 4.75, True))
        self.assertEqual(requests[0].action, "recut-scene")
        self.assertEqual(requests[0].raw_in_seconds, 1.25)
        self.assertEqual(requests[0].raw_out_seconds, 4.75)
        self.assertTrue(requests[0].exact)

    def test_review_timeline_load_move_undo_and_save(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        writes = []

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "name": project_id,
                    "status": {},
                    "artifacts": {
                        "render_plans": ["output/render/plan.json"],
                        "videos": ["output/final/scene_0001.mp4", "output/final/scene_0002.mp4"],
                    },
                }

            def read_artifact(self, project_id, path):
                return {"path": path, "data": [{"scene": 1, "duration_seconds": 2}, {"scene": 2, "duration_seconds": 3}]}

            def write_artifact(self, project_id, request):
                writes.append((project_id, request.path, request.data))
                return {"path": request.path, "data": request.data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        self.assertTrue(view_model.load_review_timeline())
        self.assertEqual([item["scene"] for item in view_model.review_items], [1, 2])
        self.assertTrue(view_model.move_review_scene(1, 0))
        self.assertEqual([item["scene"] for item in view_model.review_items], [2, 1])
        self.assertTrue(view_model.undo_review_timeline())
        self.assertTrue(view_model.save_review_timeline())
        self.assertEqual([scene["scene"] for scene in writes[0][2]], [1, 2])

    def test_import_image_encodes_file_for_existing_media_port(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        writes = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

            def write_media_data_url(self, project_id, path, data_url):
                writes.append((project_id, path, data_url))
                return {"path": path}

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "reference.png"
            image_path.write_bytes(b"png-data")
            view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
            view_model.select_project("scholoraid")

            self.assertTrue(view_model.import_image(image_path.as_uri(), "output/references/actor/hero/sheet.png"))

        self.assertEqual(writes[0][0:2], ("scholoraid", "output/references/actor/hero/sheet.png"))
        self.assertTrue(writes[0][2].startswith("data:image/png;base64,"))

    def test_start_reference_rerender_uses_actor_identity(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        requests = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        class Jobs:
            def list(self, project_id):
                return []

        class Service:
            def start_job(self, project_id, request):
                requests.append(request)
                return {}

        view_model = StudioViewModel(store=Store(), jobs=Jobs(), job_service=Service())
        view_model.select_project("scholoraid")

        self.assertTrue(view_model.start_reference_rerender("actor", "warrior_lead"))
        self.assertEqual(requests[0].reference_kind, "actor")
        self.assertEqual(requests[0].reference_id, "warrior_lead")

    def test_import_audio_uses_native_file_stream(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        uploads = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

            def store_audio_upload(self, project_id, filename, content_type, source):
                uploads.append((project_id, filename, content_type, source.read()))
                return {"path": f"input/{filename}"}

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "song.wav"
            audio_path.write_bytes(b"wave-data")
            view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
            view_model.select_project("scholoraid")

            self.assertTrue(view_model.import_audio(audio_path.as_uri()))

        self.assertEqual(uploads, [("scholoraid", "song.wav", "audio/x-wav", b"wave-data")])


class StudioQmlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def test_main_qml_loads_and_exposes_editor_shell(self):
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.studio.desktop.runtime import qml_entrypoint

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        engine = QQmlApplicationEngine()
        engine.load(qml_entrypoint())

        self.assertEqual(len(engine.rootObjects()), 1)
        root = engine.rootObjects()[0]
        self.assertIsNotNone(root.findChild(object, "projectSidebar"))
        self.assertIsNotNone(root.findChild(object, "workspace"))
        self.assertIsNotNone(root.findChild(object, "jobPanel"))
        self.assertIsNotNone(root.findChild(object, "reviewTimeline"))

    def test_main_qml_loads_scene_workspace_with_both_view_models(self):
        from PySide6.QtCore import QObject
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.studio.desktop.runtime import qml_entrypoint

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("studioViewModel", QObject())
        engine.rootContext().setContextProperty("sceneWorkspaceViewModel", QObject())
        engine.load(qml_entrypoint())

        self.assertEqual(len(engine.rootObjects()), 1)
        root = engine.rootObjects()[0]
        self.assertIsNotNone(root.findChild(object, "sceneWorkspacePage"))
        self.assertIsNotNone(root.findChild(object, "sceneInspector"))
        for object_name in (
            "renderSelectedScenesButton",
            "rerenderSelectedScenesButton",
            "retakeSelectedScenesButton",
        ):
            action = root.findChild(object, object_name)
            self.assertIsNotNone(action)
            self.assertFalse(action.property("enabled"))
        reload_button = root.findChild(object, "reloadSceneConflictButton")
        discard_button = root.findChild(object, "discardSceneConflictButton")
        self.assertIsNotNone(reload_button)
        self.assertIsNotNone(discard_button)
        self.assertEqual("Reload from disk", reload_button.property("text"))
        self.assertEqual("Discard local edits", discard_button.property("text"))

    def test_scene_list_arrow_navigation_and_space_toggle_current_scene(self):
        from PySide6.QtCore import QMetaObject, Qt
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtTest import QTest

        from feverslop.studio.desktop.runtime import qml_entrypoint
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 1},
                    {"scene_number": 2},
                )

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        studio_vm = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio_vm.select_project("demo")
        scene_vm = SceneWorkspaceViewModel(
            service=SceneService(),
            studio_view_model=studio_vm,
        )
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("studioViewModel", studio_vm)
        engine.rootContext().setContextProperty("sceneWorkspaceViewModel", scene_vm)
        engine.load(qml_entrypoint())
        root = engine.rootObjects()[0]
        root.setProperty("currentPage", 11)
        self.qml_app.processEvents()
        scene_list = root.findChild(object, "sceneCardList")
        scene_list.setProperty("currentIndex", 0)
        QMetaObject.invokeMethod(scene_list, "forceActiveFocus")

        QTest.keyClick(root, Qt.Key.Key_Down)
        self.assertEqual(1, scene_list.property("currentIndex"))
        QTest.keyClick(root, Qt.Key.Key_Space)
        self.qml_app.processEvents()

        self.assertEqual([2], scene_vm.selected_scene_numbers)

    def test_studio_palette_does_not_inherit_desktop_dark_mode(self):
        from PySide6.QtGui import QPalette

        from feverslop.studio.desktop.runtime import studio_palette

        palette = studio_palette()

        self.assertEqual(palette.color(QPalette.ColorRole.Window).name(), "#f5f5f7")
        self.assertEqual(palette.color(QPalette.ColorRole.WindowText).name(), "#1c1c1e")
        self.assertEqual(palette.color(QPalette.ColorRole.ButtonText).name(), "#1c1c1e")
        self.assertEqual(palette.color(QPalette.ColorRole.Highlight).name(), "#5b5fc7")


if __name__ == "__main__":
    unittest.main()
