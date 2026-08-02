from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
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

    def test_scene_video_thumbnail_url_uses_cached_ffmpeg_frame(self):
        from unittest.mock import patch

        from PySide6.QtCore import QUrl

        from feverslop.studio.desktop.runtime import scene_video_thumbnail_url

        with patch(
            "feverslop.studio.desktop.runtime.thumbnail_path",
            return_value=Path("C:/cache/scene.jpg"),
        ) as generate:
            result = scene_video_thumbnail_url(
                object(),
                "demo",
                "output/render/scenes/scene_0009/final.mp4",
            )

        # Match production code: QUrl.fromLocalFile instead of Path.as_uri()
        expected = QUrl.fromLocalFile(str(Path("C:/cache/scene.jpg"))).toString()
        self.assertEqual(expected, result)
        generate.assert_called_once()

    def test_scene_video_thumbnail_url_ignores_disappearing_video(self):
        from unittest.mock import patch

        from feverslop.studio.desktop.runtime import scene_video_thumbnail_url

        with patch(
            "feverslop.studio.desktop.runtime.thumbnail_path",
            side_effect=FileNotFoundError("final.mp4"),
        ):
            result = scene_video_thumbnail_url(object(), "demo", "final.mp4")

        self.assertEqual("", result)

    def test_headless_desktop_smoke_test_loads_the_production_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["QT_QUICK_BACKEND"] = "software"
            result = subprocess.run(
                [
                    sys.executable,
                    "-W",
                    "error::RuntimeWarning",
                    "-m",
                    "feverslop.studio.desktop",
                    "--projects-root",
                    temp_dir,
                    "--smoke-test",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

        self.assertEqual(
            result.returncode,
            0,
            f"desktop smoke test failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


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
            video_path = values.pop("video_path", None)
            items.append(
                SceneWorkspaceItem(
                    media=SceneMedia(
                        thumbnail_path=thumbnail_path,
                        video_path=video_path,
                    ),
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

    def test_scene_list_model_generates_preview_from_video_when_still_is_missing(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneListModel

        model = SceneListModel(
            thumbnail_url=lambda path: f"file:///project/{path}",
            video_thumbnail_url=lambda path: f"file:///cache/{Path(path).stem}.jpg",
        )
        model.replace(
            self._snapshot(
                {
                    "scene_number": 9,
                    "video_path": "output/render/scenes/scene_0009/final.mp4",
                }
            ).workspace.items
        )

        roles = {bytes(name).decode(): role for role, name in model.roleNames().items()}

        self.assertEqual(
            "file:///cache/final.jpg",
            model.data(model.index(0, 0), roles["thumbnailUrl"]),
        )

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

    def test_switch_to_movie_clears_standard_workspace_and_exposes_error(self):
        from PySide6.QtCore import QObject, Signal

        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Studio(QObject):
            currentProjectChanged = Signal()

            def __init__(self):
                super().__init__()
                self.current_project_id = "song"

        class Service:
            def load(self, project_id):
                if project_id == "film":
                    raise ValueError("Scene workspace is unavailable for movie projects")
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 1})

        studio = Studio()
        view_model = SceneWorkspaceViewModel(service=Service(), studio_view_model=studio)
        self.assertTrue(view_model.reload())
        self.assertEqual(1, view_model.scenes.rowCount())

        studio.current_project_id = "film"
        studio.currentProjectChanged.emit()

        self.assertEqual("film", view_model.current_project_id)
        self.assertEqual(0, view_model.scenes.rowCount())
        self.assertEqual(
            "Scene workspace is unavailable for movie projects",
            view_model.error,
        )

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

    def test_inspected_scene_exposes_every_ltx_prompt_source(self):
        from feverslop.application.scene_workspace import SceneWorkspaceSnapshot
        from feverslop.domain.scene_workspace import SceneWorkspace
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceSnapshot(
                    workspace=SceneWorkspace.from_scenes([{
                        "scene": 1,
                        "ltx": {
                            "original_style_i2v_prompt": "A",
                            "i2v_prompt_from_t2i": "B",
                            "base_prompt": "C",
                        },
                    }]),
                    revision="revision-1",
                )

        view_model = SceneWorkspaceViewModel(
            service=Service(),
            studio_view_model=SimpleNamespace(current_project_id="demo"),
        )
        view_model.reload()
        view_model.toggleSelection(1)

        self.assertIn("ltxPrompts", view_model.inspectedScene)
        self.assertEqual(
            {
                "original_style_i2v_prompt": "A",
                "i2v_prompt_from_t2i": "B",
                "base_prompt": "C",
            },
            view_model.inspectedScene["ltxPrompts"],
        )

    def test_selected_action_blocks_reentry_and_refreshes_jobs_immediately(self):
        from PySide6.QtTest import QSignalSpy

        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Studio:
            current_project_id = "demo"

            def __init__(self):
                self.refresh_count = 0

            def refresh_jobs(self):
                self.refresh_count += 1

        class Service:
            def __init__(self):
                self.calls = []
                self.view_model = None

            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 1})

            def start_action(self, **kwargs):
                self.calls.append(kwargs)
                self.test_case.assertTrue(self.view_model.submitting)
                self.test_case.assertFalse(self.view_model.startSelectedAction("render"))

        studio = Studio()
        service = Service()
        service.test_case = self
        view_model = SceneWorkspaceViewModel(service=service, studio_view_model=studio)
        service.view_model = view_model
        view_model.reload()
        view_model.toggleSelection(1)

        self.assertTrue(hasattr(view_model, "submittingChanged"))
        changed = QSignalSpy(view_model.submittingChanged)
        self.assertTrue(view_model.startSelectedAction("render"))

        self.assertEqual(1, len(service.calls))
        self.assertEqual(1, studio.refresh_count)
        self.assertFalse(view_model.submitting)
        self.assertEqual(2, changed.count())

    def test_failed_selected_action_clears_submitting_without_refresh(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        class Studio:
            current_project_id = "demo"
            refresh_count = 0

            def refresh_jobs(self):
                self.refresh_count += 1

        class Service:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 1})

            def start_action(self, **_kwargs):
                raise RuntimeError("dispatch failed")

        studio = Studio()
        view_model = SceneWorkspaceViewModel(service=Service(), studio_view_model=studio)
        view_model.reload()
        view_model.toggleSelection(1)

        self.assertTrue(hasattr(view_model, "submitting"))
        self.assertFalse(view_model.startSelectedAction("render"))
        self.assertFalse(view_model.submitting)
        self.assertEqual(0, studio.refresh_count)
        self.assertIn("dispatch failed", view_model.error)

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
            def read_artifact(self, project_id, path):
                return {"path": path, "data": None}

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

    def test_missing_render_plan_loads_as_empty_json_array(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def read_artifact(self, project_id, path):
                return {"path": path, "data": None, "exists": False, "revision": None}

            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")
        view_model.load_json_artifact("render_plan.json")

        self.assertEqual("[]", view_model.editor_text)

    def test_render_plan_editor_rejects_null_document(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        writes = []

        class Store:
            def write_artifact(self, project_id, request):
                writes.append((project_id, request))
                return {"path": request.path, "data": request.data, "exists": True}

            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")

        self.assertFalse(view_model.save_json_artifact("render_plan.json", "null"))
        self.assertEqual([], writes)
        self.assertIn("Render plan", view_model.error)

    def test_json_editor_draft_marks_dirty_without_editor_changed_feedback(self):
        from PySide6.QtTest import QSignalSpy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": {"scene": 1}, "revision": "r1", "exists": True}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("plan.json")
        editor_changed = QSignalSpy(view_model.editorChanged)

        view_model.set_json_editor_draft('{"scene": 1, "draft": true}')

        self.assertTrue(view_model.editor_dirty)
        self.assertEqual('{"scene": 1, "draft": true}', view_model.editor_text)
        self.assertEqual(0, editor_changed.count())

    def test_dirty_scene_refresh_preserves_raw_draft_and_forces_revision_conflict(self):
        import copy

        from feverslop.studio.projects import ArtifactConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        disk = [{"scene": 1, "shot_description": "Original"}]
        revision = ["r1"]
        writes = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(disk), "revision": revision[0], "exists": True}

            def write_artifact(self, project_id, request):
                if request.expected_revision != revision[0]:
                    raise ArtifactConflict(request.path, request.expected_revision, revision[0])
                writes.append(request.data)
                disk[:] = copy.deepcopy(request.data)
                revision[0] = "raw-r2"
                return {"path": request.path, "data": request.data, "revision": revision[0]}

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 1, "shot_description": disk[0]["shot_description"]},
                    revision=revision[0],
                )

            def patch_scene(self, **kwargs):
                disk[0]["shot_description"] = kwargs["changes"]["shot_description"]
                revision[0] = "scene-r2"
                return SimpleNamespace(revision=revision[0])

        studio = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio.select_project("song")
        studio.load_json_artifact("plan.json")
        draft = '[{"scene": 1, "shot_description": "Raw draft"}]'
        studio.set_json_editor_draft(draft)
        scenes = SceneWorkspaceViewModel(service=SceneService(), studio_view_model=studio)
        scenes.reload()

        self.assertTrue(scenes.savePromptFields(1, {"shotDescription": "Scene edit"}, ""))
        self.assertEqual("Scene edit", disk[0]["shot_description"])
        self.assertEqual(draft, studio.editor_text)
        self.assertTrue(studio.editor_dirty)
        self.assertIn("disk changed", studio.error.lower())
        self.assertFalse(studio.save_json_artifact("plan.json", draft))
        self.assertEqual([], writes)
        self.assertEqual("Scene edit", disk[0]["shot_description"])

    def test_save_loaded_json_rejects_external_change_without_writing(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        writes = []
        current_data = [{"scene": 1, "shot_description": "Original"}]

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "artifacts": {"render_plans": ["render_plan.json"]},
                }

            def read_artifact(self, project_id, path):
                return {"path": path, "data": current_data}

            def write_artifact(self, project_id, request):
                writes.append(request.data)
                return {"path": request.path, "data": request.data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("render_plan.json")
        current_data[0]["shot_description"] = "Workspace edit"

        saved = view_model.save_json_artifact(
            "render_plan.json",
            '[{"scene": 1, "shot_description": "Raw edit"}]',
        )

        self.assertFalse(saved)
        self.assertEqual([], writes)
        self.assertIn("changed externally", view_model.error)
        self.assertIn("reload", view_model.error.lower())

    def test_nested_editor_scenes_mutation_cannot_hide_external_change(self):
        import copy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        writes = []
        current_data = [{"scene": 1, "ltx": {"base_prompt": "Original"}}]

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "artifacts": {"render_plans": ["render_plan.json"]},
                }

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(current_data)}

            def write_artifact(self, project_id, request):
                writes.append(request.data)
                return {"path": request.path, "data": request.data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("render_plan.json")

        exposed_scenes = view_model.editor_scenes
        exposed_scenes[0]["ltx"]["base_prompt"] = "Workspace edit"
        self.assertEqual("Original", view_model.editor_scenes[0]["ltx"]["base_prompt"])
        current_data[0]["ltx"]["base_prompt"] = "Workspace edit"

        self.assertFalse(view_model.save_json_artifact(
            "render_plan.json",
            '[{"scene": 1, "ltx": {"base_prompt": "Raw edit"}}]',
        ))
        self.assertEqual([], writes)
        self.assertIn("changed externally", view_model.error)

    def test_save_as_rejects_existing_target_that_was_not_loaded(self):
        import copy

        from feverslop.studio.projects import ArtifactConflict
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        artifacts = {
            "a.json": {"name": "A"},
            "b.json": {"name": "B"},
        }
        writes = []

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "artifacts": {"generated_json": list(artifacts)},
                }

            def read_artifact(self, project_id, path):
                if path not in artifacts:
                    raise FileNotFoundError(path)
                return {"path": path, "data": copy.deepcopy(artifacts[path])}

            def write_artifact(self, project_id, request):
                if request.create_only and request.path in artifacts:
                    raise ArtifactConflict(request.path, None, "existing")
                writes.append(request.path)
                artifacts[request.path] = request.data
                return {"path": request.path, "data": request.data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("a.json")

        self.assertFalse(view_model.save_json_artifact("b.json", '{"name": "overwrite"}'))
        self.assertEqual([], writes)
        self.assertEqual({"name": "B"}, artifacts["b.json"])
        self.assertIn("load target before saving", view_model.error.lower())

    def test_save_as_allows_new_target_and_establishes_its_baseline(self):
        import copy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        artifacts = {"a.json": {"name": "A"}}

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "artifacts": {"generated_json": list(artifacts)},
                }

            def read_artifact(self, project_id, path):
                if path not in artifacts:
                    raise FileNotFoundError(path)
                return {"path": path, "data": copy.deepcopy(artifacts[path])}

            def write_artifact(self, project_id, request):
                artifacts[request.path] = copy.deepcopy(request.data)
                return {"path": request.path, "data": request.data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("a.json")

        self.assertTrue(view_model.save_json_artifact("c.json", '{"name": "C"}'))
        self.assertEqual("c.json", view_model.editor_path)
        self.assertTrue(view_model.save_json_artifact("c.json", '{"name": "C2"}'))
        self.assertEqual({"name": "C2"}, artifacts["c.json"])

    def test_loaded_missing_artifact_uses_create_only_against_concurrent_creator(self):
        from feverslop.studio.projects import ArtifactConflict
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        target = {"exists": False, "data": None}
        requests = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": target["data"], "revision": None, "exists": target["exists"]}

            def write_artifact(self, project_id, request):
                requests.append(request)
                if request.create_only and target["exists"]:
                    raise ArtifactConflict(request.path, None, "concurrent")
                target.update(exists=True, data=request.data)
                return {"path": request.path, "data": request.data, "revision": "ours"}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("new.json")
        target.update(exists=True, data={"owner": "concurrent"})

        self.assertFalse(view_model.save_json_artifact("new.json", '{"owner": "ours"}'))
        self.assertEqual(1, len(requests))
        self.assertTrue(requests[0].create_only)
        self.assertEqual({"owner": "concurrent"}, target["data"])

    def test_save_loaded_json_succeeds_when_baseline_is_unchanged(self):
        from PySide6.QtTest import QSignalSpy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        current_data = [{"scene": 1, "shot_description": "Original"}]
        writes = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["render_plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": current_data}

            def write_artifact(self, project_id, request):
                writes.append(request.data)
                current_data[:] = request.data
                return {"path": request.path, "data": request.data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("render_plan.json")
        project_changed = QSignalSpy(view_model.currentProjectChanged)

        self.assertTrue(view_model.save_json_artifact(
            "render_plan.json",
            '[{"scene": 1, "shot_description": "Raw edit"}]',
        ))
        self.assertEqual("Raw edit", writes[0][0]["shot_description"])
        self.assertEqual(1, project_changed.count())

    def test_refresh_render_plan_editor_reloads_open_catalogued_plan_only(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        current_data = [{"scene": 1, "shot_description": "Original"}]

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["render_plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": current_data}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("render_plan.json")
        current_data[0]["shot_description"] = "Workspace edit"

        self.assertTrue(view_model.refresh_render_plan_editor())
        self.assertEqual("Workspace edit", json.loads(view_model.editor_text)[0]["shot_description"])

    def test_structured_patch_refreshes_loaded_nonpreferred_render_plan(self):
        import copy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        plans = {
            "preferred.json": [{"scene": 1, "prompt": "Preferred"}],
            "alternate.json": [{"scene": 1, "prompt": "Alternate"}],
        }
        revisions = {"preferred.json": "p1", "alternate.json": "a1"}

        class Store:
            def describe_project(self, project_id):
                return {
                    "id": project_id,
                    "artifacts": {"render_plans": ["preferred.json", "alternate.json"]},
                }

            def read_artifact(self, project_id, path):
                return {
                    "path": path,
                    "data": copy.deepcopy(plans[path]),
                    "revision": revisions[path],
                    "exists": True,
                }

            def patch_render_plan(self, project_id, patch):
                self.seen_revision = patch.expected_revision
                plans[patch.path][0].update(patch.updates)
                revisions[patch.path] = "a2"
                return {"revision": "a2", "scene": plans[patch.path][0]}

        store = Store()
        view_model = StudioViewModel(store=store, jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("alternate.json")

        self.assertTrue(view_model.patch_render_scene("alternate.json", 1, {"prompt": "Patched"}))
        self.assertEqual("a1", store.seen_revision)
        self.assertIn("Patched", view_model.editor_text)
        self.assertEqual("Preferred", plans["preferred.json"][0]["prompt"])

    def test_workspace_save_refreshes_open_raw_editor_and_its_baseline(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        current_data = [{"scene": 1, "shot_description": "Original"}]

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["render_plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": current_data}

            def write_artifact(self, project_id, request):
                current_data[:] = request.data
                return {"path": request.path, "data": request.data}

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 1, "shot_description": current_data[0]["shot_description"]}
                )

            def patch_scene(self, **kwargs):
                current_data[0]["shot_description"] = kwargs["changes"]["shot_description"]
                return SimpleNamespace(revision="revision-2")

        studio = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio.select_project("song")
        studio.load_json_artifact("render_plan.json")
        scenes = SceneWorkspaceViewModel(service=SceneService(), studio_view_model=studio)
        scenes.reload()

        self.assertTrue(scenes.savePromptFields(1, {"shotDescription": "Workspace edit"}, ""))
        self.assertEqual("Workspace edit", json.loads(studio.editor_text)[0]["shot_description"])
        self.assertTrue(studio.save_json_artifact(
            "render_plan.json",
            '[{"scene": 1, "shot_description": "Raw after workspace"}]',
        ))

    def test_failed_workspace_save_does_not_refresh_raw_editor(self):
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel

        refreshes = []

        class Studio:
            current_project_id = "song"

            def refresh_render_plan_editor(self):
                refreshes.append(True)

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 1})

            def patch_scene(self, **_kwargs):
                raise OSError("write failed")

        scenes = SceneWorkspaceViewModel(service=SceneService(), studio_view_model=Studio())
        scenes.reload()

        self.assertFalse(scenes.savePromptFields(1, {"shotDescription": "Unsaved"}, ""))
        self.assertEqual([], refreshes)

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
                return {
                    "id": project_id,
                    "name": project_id,
                    "status": {},
                    "artifacts": {"render_plans": ["render_plan_msr.json"]},
                }

            def read_artifact(self, project_id, path):
                return {
                    "path": path,
                    "data": [{"scene": 5, "prompt": "Old"}],
                    "revision": "r1",
                    "exists": True,
                }

            def patch_render_plan(self, project_id, patch):
                patches.append((project_id, patch))
                return {"scene": {"scene": patch.scene, **patch.updates}}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("scholoraid")
        view_model.load_json_artifact("render_plan_msr.json")

        self.assertTrue(view_model.patch_render_scene("render_plan_msr.json", 5, {"prompt": "The party reaches the gate."}))
        self.assertEqual(patches[0][1].scene, 5)
        self.assertEqual(patches[0][1].updates["prompt"], "The party reaches the gate.")
        self.assertEqual("r1", patches[0][1].expected_revision)

    def test_patch_render_scene_rejects_path_other_than_loaded_editor(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        patches = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["a.json", "b.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": [{"scene": 1}], "revision": "r1", "exists": True}

            def patch_render_plan(self, project_id, patch):
                patches.append(patch)

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("a.json")

        self.assertFalse(view_model.patch_render_scene("b.json", 1, {"prompt": "Wrong"}))
        self.assertEqual([], patches)
        self.assertIn("load target before patching", view_model.error.lower())

    def test_patch_render_scene_rejects_dirty_raw_draft_without_writing(self):
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        patches = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {
                    "path": path,
                    "data": [{"scene": 1, "prompt": "Original"}],
                    "revision": "r1",
                    "exists": True,
                }

            def patch_render_plan(self, project_id, patch):
                patches.append(patch)

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("plan.json")
        draft = '[{"scene": 1, "prompt": "Raw draft"}]'
        view_model.set_json_editor_draft(draft)

        self.assertFalse(view_model.patch_render_scene("plan.json", 1, {"prompt": "Structured"}))
        self.assertEqual([], patches)
        self.assertEqual(draft, view_model.editor_text)
        self.assertTrue(view_model.editor_dirty)
        self.assertIn("save or reload raw draft first", view_model.error.lower())

    def test_clean_structured_patch_refreshes_raw_revision_and_baseline(self):
        import copy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        disk = [{"scene": 1, "prompt": "Original"}]
        revision = ["r1"]
        raw_requests = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {
                    "path": path,
                    "data": copy.deepcopy(disk),
                    "revision": revision[0],
                    "exists": True,
                }

            def patch_render_plan(self, project_id, patch):
                disk[0].update(patch.updates)
                revision[0] = "r2"

            def write_artifact(self, project_id, request):
                raw_requests.append(request)
                disk[:] = copy.deepcopy(request.data)
                revision[0] = "r3"
                return {"path": request.path, "data": request.data, "revision": revision[0]}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("plan.json")

        self.assertTrue(view_model.patch_render_scene("plan.json", 1, {"prompt": "Structured"}))
        self.assertIn("Structured", view_model.editor_text)
        self.assertTrue(view_model.save_json_artifact(
            "plan.json",
            '[{"scene": 1, "prompt": "Raw after structured"}]',
        ))
        self.assertEqual("r2", raw_requests[0].expected_revision)

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

    def test_dirty_review_survives_disk_refresh_and_later_save_conflicts(self):
        import copy

        from feverslop.studio.projects import ArtifactConflict
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        disk = [
            {"scene": 1, "description": "One", "duration_seconds": 1},
            {"scene": 2, "description": "Two", "duration_seconds": 1},
        ]
        revision = ["r1"]
        writes = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(disk), "revision": revision[0], "exists": True}

            def write_artifact(self, project_id, request):
                if request.expected_revision != revision[0]:
                    raise ArtifactConflict(request.path, request.expected_revision, revision[0])
                writes.append(request.data)
                return {"path": request.path, "data": request.data, "revision": "saved"}

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot(
                    {"scene_number": 1, "shot_description": disk[0]["description"]},
                    revision=revision[0],
                )

            def patch_scene(self, **kwargs):
                disk[0]["description"] = kwargs["changes"]["shot_description"]
                revision[0] = "r2"
                return SimpleNamespace(revision=revision[0])

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        self.assertTrue(view_model.load_review_timeline())
        self.assertTrue(view_model.move_review_scene(1, 0))
        scenes = SceneWorkspaceViewModel(service=SceneService(), studio_view_model=view_model)
        scenes.reload()

        self.assertTrue(scenes.savePromptFields(1, {"shotDescription": "Workspace prompt"}, ""))
        self.assertEqual([2, 1], [item["scene"] for item in view_model.review_items])
        self.assertTrue(view_model.review_dirty)
        self.assertIn("save/reload review", view_model.error.lower())
        self.assertFalse(view_model.save_review_timeline())
        self.assertTrue(view_model.review_dirty)
        self.assertEqual([], writes)
        self.assertEqual("Workspace prompt", disk[0]["description"])

    def test_clean_review_refreshes_without_raw_editor_and_updates_revision(self):
        import copy

        from PySide6.QtTest import QSignalSpy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        disk = [{"scene": 1, "description": "Original", "duration_seconds": 1}]
        revision = ["r1"]
        requests = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(disk), "revision": revision[0], "exists": True}

            def write_artifact(self, project_id, request):
                requests.append(request)
                return {"path": request.path, "data": request.data, "revision": "r3"}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        self.assertTrue(view_model.load_review_timeline())
        changed = QSignalSpy(view_model.reviewChanged)
        disk[0]["description"] = "Workspace prompt"
        revision[0] = "r2"

        self.assertTrue(view_model.refresh_render_plan_editor())
        self.assertEqual("Workspace prompt", view_model.review_items[0]["preview"])
        self.assertEqual(1, changed.count())
        self.assertTrue(view_model.save_review_timeline())
        self.assertEqual("r2", requests[0].expected_revision)

    def test_refresh_preserves_dirty_raw_and_review_with_consolidated_warning(self):
        import copy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        disk = [
            {"scene": 1, "description": "One", "duration_seconds": 1},
            {"scene": 2, "description": "Two", "duration_seconds": 1},
        ]
        revision = ["r1"]

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(disk), "revision": revision[0], "exists": True}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("plan.json")
        self.assertTrue(view_model.load_review_timeline())
        draft = '[{"scene": 1, "description": "Raw draft"}]'
        view_model.set_json_editor_draft(draft)
        self.assertTrue(view_model.move_review_scene(1, 0))
        disk[0]["description"] = "Workspace"
        revision[0] = "r2"

        self.assertFalse(view_model.refresh_render_plan_editor())
        self.assertEqual(draft, view_model.editor_text)
        self.assertEqual([2, 1], [item["scene"] for item in view_model.review_items])
        self.assertIn("save/reload raw draft", view_model.error.lower())
        self.assertIn("save/reload review", view_model.error.lower())

    def test_same_project_raw_save_signal_preserves_dirty_review_and_revision(self):
        import copy

        from feverslop.studio.projects import ArtifactConflict
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        disk = [
            {"scene": 1, "description": "One", "duration_seconds": 1},
            {"scene": 2, "description": "Two", "duration_seconds": 1},
        ]
        revision = ["r1"]

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(disk), "revision": revision[0], "exists": True}

            def write_artifact(self, project_id, request):
                if request.expected_revision != revision[0]:
                    raise ArtifactConflict(request.path, request.expected_revision, revision[0])
                disk[:] = copy.deepcopy(request.data)
                revision[0] = "r2"
                return {"path": request.path, "data": request.data, "revision": revision[0]}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("song")
        view_model.load_json_artifact("plan.json")
        self.assertTrue(view_model.load_review_timeline())
        self.assertTrue(view_model.move_review_scene(1, 0))
        view_model.currentProjectChanged.connect(view_model.load_review_timeline)

        self.assertTrue(view_model.save_json_artifact(
            "plan.json",
            '[{"scene": 1, "description": "Raw saved", "duration_seconds": 1}, {"scene": 2, "description": "Two", "duration_seconds": 1}]',
        ))
        self.assertTrue(view_model.review_dirty)
        self.assertEqual([2, 1], [item["scene"] for item in view_model.review_items])
        self.assertIn("dirty review", view_model.error.lower())
        self.assertFalse(view_model.save_review_timeline())
        self.assertEqual("Raw saved", disk[0]["description"])

    def test_actual_project_switch_clears_transients_before_loading_new_review(self):
        import copy

        from PySide6.QtTest import QSignalSpy

        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        documents = {
            "alpha": [{"scene": 1, "description": "Alpha", "duration_seconds": 1}, {"scene": 2, "description": "A2", "duration_seconds": 1}],
            "beta": [{"scene": 7, "description": "Beta", "duration_seconds": 1}],
        }

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {"path": path, "data": copy.deepcopy(documents[project_id]), "revision": project_id, "exists": True}

        view_model = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        view_model.select_project("alpha")
        view_model.load_json_artifact("plan.json")
        self.assertTrue(view_model.load_review_timeline())
        self.assertTrue(view_model.move_review_scene(1, 0))
        view_model.set_json_editor_draft("dirty raw")
        editor_changed = QSignalSpy(view_model.editorChanged)
        review_changed = QSignalSpy(view_model.reviewChanged)
        view_model.currentProjectChanged.connect(view_model.load_review_timeline)

        view_model.select_project("beta")

        self.assertEqual("", view_model.editor_path)
        self.assertFalse(view_model.editor_dirty)
        self.assertFalse(view_model.review_dirty)
        self.assertEqual([7], [item["scene"] for item in view_model.review_items])
        self.assertGreaterEqual(editor_changed.count(), 1)
        self.assertGreaterEqual(review_changed.count(), 2)

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
        from mimetypes import guess_type

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

        # Use platform-specific MIME type (audio/wav on Windows, audio/x-wav on Linux)
        expected_mime = guess_type("song.wav")[0]
        self.assertEqual(uploads, [("scholoraid", "song.wav", expected_mime, b"wave-data")])


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
        selection_hint = root.findChild(object, "selectedSceneCount")
        self.assertIn("Ctrl+click", selection_hint.property("text"))

    def test_scene_workspace_uses_readable_dark_surfaces_and_roomy_controls(self):
        from PySide6.QtGui import QColor, QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.studio.desktop.runtime import qml_entrypoint

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        engine = QQmlApplicationEngine()
        engine.load(qml_entrypoint())
        root = engine.rootObjects()[0]
        root.setProperty("currentPage", 11)
        self.qml_app.processEvents()
        page = root.findChild(object, "sceneWorkspacePage")
        toolbar = root.findChild(object, "sceneWorkspaceToolbar")
        header = root.findChild(object, "workspaceHeader")
        scene_scroll_thumb = root.findChild(object, "sceneListScrollThumb")
        image_prompt = root.findChild(object, "sceneImagePrompt")

        self.assertEqual(QColor("#18181B"), page.property("color"))
        self.assertGreaterEqual(page.property("contentPadding"), 24)
        self.assertGreaterEqual(page.property("sceneCardHeight"), 108)
        self.assertEqual(QColor("#27272A"), toolbar.property("color"))
        self.assertEqual(QColor("#202024"), header.property("color"))
        self.assertEqual(QColor("#52525B"), scene_scroll_thumb.property("color"))
        self.assertEqual(QColor("#F4F4F5"), image_prompt.property("color"))
        self.assertGreaterEqual(image_prompt.property("leftPadding"), 14)
        self.assertGreaterEqual(image_prompt.property("topPadding"), 12)

    def test_movie_project_disables_scene_workspace_navigation_and_leaves_page(self):
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.studio.desktop.runtime import qml_entrypoint
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                project_type = "movie" if project_id == "film" else "standard_music_video"
                return {
                    "id": project_id,
                    "name": project_id,
                    "project_type": project_type,
                    "artifacts": {},
                }

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        studio_vm = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio_vm.select_project("song")
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("studioViewModel", studio_vm)
        engine.load(qml_entrypoint())
        root = engine.rootObjects()[0]
        root.setProperty("currentPage", 11)

        studio_vm.select_project("film")
        self.qml_app.processEvents()

        navigation = root.findChild(object, "sceneWorkspaceNavigation")
        self.assertIsNotNone(navigation)
        self.assertFalse(navigation.property("enabled"))
        self.assertNotEqual(11, root.property("currentPage"))

    def test_render_plan_editor_reports_drafts_and_clears_stale_selected_scene(self):
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.studio.desktop.runtime import qml_entrypoint
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "artifacts": {"render_plans": ["plan.json"]}}

            def read_artifact(self, project_id, path):
                return {
                    "path": path,
                    "data": [{"scene": 1, "prompt": "Original"}],
                    "revision": "r1",
                    "exists": True,
                }

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        studio_vm = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio_vm.select_project("song")
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("studioViewModel", studio_vm)
        engine.load(qml_entrypoint())
        root = engine.rootObjects()[0]
        workspace = root.findChild(object, "renderPlanWorkspace")
        editor = root.findChild(object, "renderPlanJsonEditor")

        self.assertIsNotNone(workspace)
        self.assertIsNotNone(editor)
        workspace.setProperty("selectedScene", {"scene": 1, "prompt": "Stale"})
        studio_vm.load_json_artifact("plan.json")
        self.qml_app.processEvents()
        self.assertIsNone(workspace.property("selectedScene"))

        editor.setProperty("text", '[{"scene": 1, "prompt": "Draft"}]')
        self.qml_app.processEvents()
        self.assertTrue(studio_vm.editor_dirty)
        self.assertIn("Draft", studio_vm.editor_text)

    def test_scene_list_arrow_navigation_and_space_toggle_current_scene(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuick import QQuickItem
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
                    {"scene_number": 3},
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
        scene_vm.toggleSelection(1)
        scene_vm.toggleSelection(3)
        scene_list = root.findChild(QQuickItem, "sceneCardList")
        page = root.findChild(object, "sceneWorkspacePage")
        scene_list.setProperty("currentIndex", 0)
        second_card_center = (
            page.property("sceneCardHeight")
            + scene_list.property("spacing")
            + page.property("sceneCardHeight") / 2
        )
        click_point = scene_list.mapToScene(QPointF(40, second_card_center)).toPoint()

        QTest.mouseClick(root, Qt.MouseButton.LeftButton, pos=click_point)
        self.qml_app.processEvents()
        self.assertEqual(1, scene_list.property("currentIndex"))
        self.assertEqual([2], scene_vm.selected_scene_numbers)
        first_card_center = page.property("sceneCardHeight") / 2
        first_click_point = scene_list.mapToScene(QPointF(40, first_card_center)).toPoint()
        QTest.mouseClick(
            root,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            pos=first_click_point,
        )
        self.qml_app.processEvents()
        self.assertEqual([1, 2], scene_vm.selected_scene_numbers)
        QTest.mouseClick(
            root,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            pos=first_click_point,
        )
        self.qml_app.processEvents()
        self.assertEqual([2], scene_vm.selected_scene_numbers)
        scene_list.setProperty("currentIndex", 1)
        QTest.keyClick(root, Qt.Key.Key_Down)
        self.assertEqual(2, scene_list.property("currentIndex"))
        QTest.keyClick(root, Qt.Key.Key_Space)
        self.qml_app.processEvents()

        self.assertEqual([2, 3], scene_vm.selected_scene_numbers)

    def test_scene_inspector_switches_ltx_source_before_saving(self):
        from PySide6.QtCore import QMetaObject
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.application.scene_workspace import SceneWorkspaceSnapshot
        from feverslop.domain.scene_workspace import SceneWorkspace
        from feverslop.studio.desktop.runtime import qml_entrypoint
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        patches = []

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceSnapshot(
                    SceneWorkspace.from_scenes([{
                        "scene": 1,
                        "ltx": {
                            "original_style_i2v_prompt": "A",
                            "i2v_prompt_from_t2i": "B",
                            "base_prompt": "C",
                        },
                    }]),
                    "revision-1",
                )

            def patch_scene(self, **kwargs):
                patches.append(kwargs)
                return SimpleNamespace(revision="revision-2")

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        studio_vm = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio_vm.select_project("demo")
        scene_vm = SceneWorkspaceViewModel(service=SceneService(), studio_view_model=studio_vm)
        scene_vm.reload()
        scene_vm.toggleSelection(1)
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("studioViewModel", studio_vm)
        engine.rootContext().setContextProperty("sceneWorkspaceViewModel", scene_vm)
        engine.load(qml_entrypoint())
        root = engine.rootObjects()[0]
        root.setProperty("currentPage", 11)
        self.qml_app.processEvents()
        source = root.findChild(object, "sceneLtxPromptSource")
        editor = root.findChild(object, "sceneLtxPrompt")
        save = root.findChild(object, "saveScenePromptsButton")

        self.assertEqual("A", editor.property("text"))
        source.setProperty("currentIndex", 1)
        self.qml_app.processEvents()
        self.assertEqual("B", editor.property("text"))
        editor.setProperty("text", "B edited")
        QMetaObject.invokeMethod(save, "click")

        self.assertEqual("B edited", patches[0]["changes"]["ltx.i2v_prompt_from_t2i"])

    def test_dirty_save_error_is_visible_and_can_be_discarded_without_conflict(self):
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from feverslop.studio.desktop.runtime import qml_entrypoint
        from feverslop.studio.desktop.viewmodels.scenes import SceneWorkspaceViewModel
        from feverslop.studio.desktop.viewmodels.studio import StudioViewModel

        class Store:
            def describe_project(self, project_id):
                return {"id": project_id, "name": project_id, "status": {}, "artifacts": {}}

        class SceneService:
            def load(self, _project_id):
                return SceneWorkspaceViewModelTests._snapshot({"scene_number": 1})

            def patch_scene(self, **_kwargs):
                raise OSError("cannot save prompts")

        self.qml_app = QGuiApplication.instance() or QGuiApplication([])
        studio_vm = StudioViewModel(store=Store(), jobs=object(), job_service=object())
        studio_vm.select_project("demo")
        scene_vm = SceneWorkspaceViewModel(service=SceneService(), studio_view_model=studio_vm)
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("studioViewModel", studio_vm)
        engine.rootContext().setContextProperty("sceneWorkspaceViewModel", scene_vm)
        engine.load(qml_entrypoint())
        root = engine.rootObjects()[0]
        root.setProperty("currentPage", 11)
        scene_vm.toggleSelection(1)
        scene_vm.savePromptFields(1, {"shotDescription": "Local"}, "")
        self.qml_app.processEvents()
        error_banner = root.findChild(object, "sceneWorkspaceErrorBanner")
        discard = root.findChild(object, "discardDirtySceneButton")

        self.assertIsNotNone(error_banner)
        self.assertIsNotNone(discard)
        self.assertTrue(scene_vm.dirty)
        self.assertFalse(scene_vm.conflict)
        self.assertTrue(error_banner.property("visible"))
        self.assertIn("cannot save prompts", error_banner.property("text"))
        self.assertTrue(discard.property("visible"))

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
