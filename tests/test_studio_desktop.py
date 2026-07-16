from __future__ import annotations

import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

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

        with tempfile.TemporaryDirectory() as temp_dir:
            context = create_studio_context(Path(temp_dir))

        self.assertIsInstance(context.store, ProjectStore)
        self.assertIsInstance(context.jobs, JobRegistry)
        self.assertIsInstance(context.job_service, StudioJobService)


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
