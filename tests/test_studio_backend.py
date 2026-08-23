import gc
import hashlib
import importlib.util
import json
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path

from rich.panel import Panel

from feverslop.studio.job_service import StudioFullAutoConsole as _StudioFullAutoConsole
from feverslop.studio.job_service import (
    build_full_auto_handler,
    pipeline_mode_from_config,
)
from feverslop.studio.jobs import (
    JobRegistry,
    build_ffmpeg_recut_command,
    build_pipeline_options,
    run_with_stream_logging,
)
from feverslop.studio.projects import (
    ArtifactRequest,
    ProjectCreateRequest,
    ProjectStore,
    RenderPlanPatch,
    StudioPathError,
    sanitize_audio_filename,
    slugify_project_name,
)
from tests.studio_harness import NativeStudioHarness


class StudioBackendTests(unittest.TestCase):
    def _project_store(self, root: Path) -> ProjectStore:
        project = root / "demo"
        (project / "output" / "render").mkdir(parents=True)
        (project / "output" / "references").mkdir(parents=True)
        (project / "output" / "render" / "render_plan_song.json").write_text(
            json.dumps(
                [
                    {"scene": 1, "prompt": "old prompt", "actor_references": []},
                    {"scene": 2, "prompt": "second", "location_references": []},
                ],
            ),
            encoding="utf-8",
        )
        (project / "output" / "references" / "actor_manifest.json").write_text("{}", encoding="utf-8")
        (project / "output" / "references" / "still.png").write_bytes(b"png")
        (project / "input").mkdir()
        (project / "input" / "song.mp3").write_bytes(b"audio")
        (project / "config.json").write_text('{"project_name": "Demo", "input_audio": "input/song.mp3"}', encoding="utf-8")
        return ProjectStore(root)

    def test_discovers_projects_with_status_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            projects = store.list_projects()

            self.assertEqual(["demo"], [project["id"] for project in projects])
            self.assertEqual("Demo", projects[0]["name"])
            self.assertEqual("present", projects[0]["status"]["config"])
            self.assertEqual("present", projects[0]["status"]["render_plan"])
            self.assertIn("output/render/render_plan_song.json", projects[0]["artifacts"]["render_plans"])
            self.assertIn("input/song.mp3", projects[0]["artifacts"]["audio"])
            self.assertEqual(5, projects[0]["artifact_sizes"]["by_type"]["audio"])
            self.assertGreater(projects[0]["artifact_sizes"]["total_bytes"], 5)

    def test_artifact_catalog_lists_canonical_render_plans_before_legacy_plans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            project = Path(temp_dir) / "demo"
            plans = project / "output" / "render" / "plans"
            plans.mkdir()
            (plans / "base.json").write_text("[]", encoding="utf-8")
            (plans / "ingredients.json").write_text("[]", encoding="utf-8")

            described = store.describe_project("demo")

            self.assertEqual("present", described["status"]["render_plan"])
            self.assertEqual(
                [
                    "output/render/plans/base.json",
                    "output/render/plans/ingredients.json",
                    "output/render/render_plan_song.json",
                ],
                described["artifacts"]["render_plans"],
            )
            self.assertEqual(
                sum((project / path).stat().st_size for path in described["artifacts"]["render_plans"]),
                described["artifact_sizes"]["by_type"]["render_plans"],
            )

    def test_thumbnail_cache_is_not_reported_as_project_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            cache = Path(temp_dir) / "demo" / ".studio" / "thumbnails"
            cache.mkdir(parents=True)
            (cache / "scene.jpg").write_bytes(b"cache")

            project = store.describe_project("demo")

            self.assertNotIn(".studio/thumbnails/scene.jpg", project["artifacts"]["images"])
            self.assertEqual(0, project["artifact_sizes"]["by_type"]["images"])

    def test_describe_project_uses_single_artifact_catalog_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            class SnapshotOnlyCatalog:
                def catalog_snapshot(self, project_id):
                    self.project_id = project_id
                    return {
                        "artifacts": {
                            "configs": ["config.json"],
                            "render_plans": [],
                            "references": [],
                            "generated_json": [],
                            "videos": [],
                            "images": [],
                            "audio": [],
                        },
                        "artifact_sizes": {
                            "total_bytes": 2,
                            "by_type": {
                                "configs": 2,
                                "render_plans": 0,
                                "references": 0,
                                "generated_json": 0,
                                "videos": 0,
                                "images": 0,
                                "audio": 0,
                                "other": 0,
                            },
                        },
                    }

                def list_artifacts(self, _project_id):
                    raise AssertionError("describe_project should not scan artifacts separately")

                def artifact_sizes(self, _project_id):
                    raise AssertionError("describe_project should not scan sizes separately")

            store.artifact_catalog = SnapshotOnlyCatalog()

            project = store.describe_project("demo")

            self.assertEqual(["config.json"], project["artifacts"]["configs"])
            self.assertEqual(2, project["artifact_sizes"]["total_bytes"])

    def test_clear_thumbnail_cache_removes_cached_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            cache = store.thumbnail_cache_path("demo", "scene")
            cache.write_bytes(b"jpg")

            removed = store.clear_thumbnail_cache("demo")

            self.assertEqual(1, removed)
            self.assertFalse(cache.exists())

    def test_slugify_project_name_uses_filesystem_safe_lowercase_slug(self):
        self.assertEqual("my-cool-video", slugify_project_name("My Cool Video!"))
        self.assertEqual("neon-wolves", slugify_project_name("  Neon Wolves  "))
        self.assertEqual("", slugify_project_name(" !!! "))

    def test_sanitize_audio_filename_removes_paths_and_unsafe_characters(self):
        self.assertEqual("bad_name.mp3", sanitize_audio_filename("../bad name.mp3"))
        self.assertEqual("song.wav", sanitize_audio_filename(r"..\song.wav"))
        with self.assertRaises(ValueError):
            sanitize_audio_filename("../")

    def test_create_standard_project_writes_config_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            project = store.create_project(
                ProjectCreateRequest(
                    project_type="standard_music_video",
                    name="My Cool Video",
                ),
            )

            root = Path(temp_dir) / "my-cool-video"
            config = json.loads((root / "config.json").read_text())
            self.assertEqual("my-cool-video", project["id"])
            self.assertEqual("My Cool Video", config["project_name"])
            self.assertEqual("en", config["audio"]["language"])
            self.assertEqual(-1, config["scene_generation"]["seed"])
            metadata = json.loads((root / ".studio" / "project.json").read_text())
            self.assertEqual("standard_music_video", metadata["project_type"])
            self.assertEqual("My Cool Video", metadata["display_name"])
            self.assertFalse(project["silent_mode"])

    def test_create_standard_project_persists_selected_video_pipeline(self):
        expected_video_pipelines = {
            "classic": "ltx_i2v",
            "msr": "ltx_msr",
            "ingredients": "ltx_ingredients",
            "minimax_h3_r2v": "minimax-h3-r2v",
            "minimax_h3_t2v": "minimax-h3-t2v",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            for pipeline_mode, expected_video_pipeline in expected_video_pipelines.items():
                with self.subTest(pipeline_mode=pipeline_mode):
                    project = store.create_project(
                        ProjectCreateRequest(
                            project_type="standard_music_video",
                            name=f"Pipeline {pipeline_mode}",
                            pipeline_mode=pipeline_mode,
                        ),
                    )
                    config = json.loads((Path(temp_dir) / project["id"] / "config.json").read_text())

                    self.assertEqual(expected_video_pipeline, config["video_pipeline"])

    def test_create_standard_project_rejects_unknown_pipeline_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            with self.assertRaisesRegex(ValueError, "classic|msr|ingredients|minimax_h3_r2v|minimax_h3_t2v"):
                store.create_project(
                    ProjectCreateRequest(
                        project_type="standard_music_video",
                        name="Unknown pipeline",
                        pipeline_mode="unknown",
                    ),
                )

    def test_create_project_persists_silent_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            project = store.create_project(
                ProjectCreateRequest(
                    project_type="standard_music_video",
                    name="Silent Video",
                    silent_mode=True,
                ),
            )

            root = Path(temp_dir) / "silent-video"
            config = json.loads((root / "config.json").read_text())
            metadata = json.loads((root / ".studio" / "project.json").read_text())
            self.assertTrue(config["silent_mode"])
            self.assertTrue(metadata["silent_mode"])
            self.assertTrue(project["silent_mode"])

    def test_create_full_auto_project_writes_inputs_and_rejects_duplicate_slug(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(
                ProjectCreateRequest(
                    project_type="full_auto",
                    name="Neon Wolves",
                    idea="A cyberpunk chase",
                    song_style="dark synthwave",
                    duration_seconds=95.5,
                    width=1920,
                    height=1080,
                    fps=50,
                    pipeline_mode="msr",
                ),
            )

            root = Path(temp_dir) / "neon-wolves"
            metadata = json.loads((root / ".studio" / "project.json").read_text())
            self.assertEqual("full_auto", metadata["project_type"])
            self.assertEqual(
                {
                    "idea": "A cyberpunk chase",
                    "song_style": "dark synthwave",
                    "duration_seconds": 95.5,
                    "width": 1920,
                    "height": 1080,
                    "fps": 50,
                    "silent_mode": False,
                    "pipeline_mode": "msr",
                },
                metadata["full_auto"],
            )
            with self.assertRaises(ValueError):
                store.create_project(
                    ProjectCreateRequest(
                        project_type="full_auto",
                        name="Neon Wolves",
                        idea="again",
                        song_style="again",
                    ),
                )

    def test_create_full_auto_project_validates_render_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            for kwargs in [
                {"duration_seconds": 0},
                {"width": 0},
                {"height": -1},
                {"fps": 30},
                {"pipeline_mode": "unknown"},
            ]:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        store.create_project(
                            ProjectCreateRequest(
                                project_type="full_auto",
                                name=f"Bad {len(kwargs)} {list(kwargs)[0]}",
                                idea="idea",
                                song_style="style",
                                **kwargs,
                            ),
                        )

    def test_artifact_read_write_is_project_relative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            artifact = store.read_artifact("demo", "config.json")
            store.write_artifact("demo", ArtifactRequest(path="config.json", data={**artifact["data"], "project_name": "Changed"}))

            self.assertEqual("Changed", json.loads((Path(temp_dir) / "demo" / "config.json").read_text())["project_name"])
            with self.assertRaises(StudioPathError):
                store.read_artifact("demo", "../outside.json")

    def test_artifact_read_reports_exact_byte_revision_and_existence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            path = Path(temp_dir) / "demo" / "config.json"

            artifact = store.read_artifact("demo", "config.json")
            missing = store.read_artifact("demo", "missing.json")

            self.assertTrue(artifact["exists"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["revision"])
            self.assertFalse(missing["exists"])
            self.assertIsNone(missing["revision"])
            self.assertIsNone(missing["data"])

    def test_artifact_write_rejects_stale_revision_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            artifact = store.read_artifact("demo", "output/render/render_plan_song.json")
            path = Path(temp_dir) / "demo" / "output" / "render" / "render_plan_song.json"
            external = b'[{"scene":1,"prompt":"external"}]'
            path.write_bytes(external)

            with self.assertRaisesRegex(Exception, "changed") as caught:
                store.write_artifact(
                    "demo",
                    ArtifactRequest(
                        path="output/render/render_plan_song.json",
                        data=[{"scene": 1, "prompt": "ours"}],
                        expected_revision=artifact["revision"],
                    ),
                )

            self.assertEqual("ArtifactConflict", type(caught.exception).__name__)
            self.assertEqual(external, path.read_bytes())
            self.assertFalse(any(path.parent.glob(f".{path.name}.*.tmp")))

    def test_artifact_write_with_current_revision_returns_new_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            artifact = store.read_artifact("demo", "output/render/render_plan_song.json")

            written = store.write_artifact(
                "demo",
                ArtifactRequest(
                    path="output/render/render_plan_song.json",
                    data=[{"scene": 1, "prompt": "ours"}],
                    expected_revision=artifact["revision"],
                ),
            )

            current = store.read_artifact("demo", "output/render/render_plan_song.json")
            self.assertEqual(current["revision"], written["revision"])
            self.assertNotEqual(artifact["revision"], written["revision"])

    def test_artifact_create_only_allows_exactly_one_concurrent_creator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            barrier = threading.Barrier(2)
            results = []

            def create(value):
                barrier.wait()
                try:
                    results.append(store.write_artifact(
                        "demo",
                        ArtifactRequest(path="new.json", data={"value": value}, create_only=True),
                    ))
                except Exception as exc:  # noqa: BLE001 - assertion records race loser
                    results.append(exc)

            threads = [threading.Thread(target=create, args=(value,)) for value in (1, 2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            successes = [result for result in results if isinstance(result, dict)]
            conflicts = [result for result in results if isinstance(result, Exception)]
            self.assertEqual(1, len(successes))
            self.assertEqual(1, len(conflicts))
            self.assertEqual("ArtifactConflict", type(conflicts[0]).__name__)

    def test_artifact_create_only_removes_partial_file_when_fsync_fails(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            path = Path(temp_dir) / "demo" / "new.json"

            with patch("feverslop.studio.projects.os.fsync", side_effect=OSError("fsync failed")):
                with self.assertRaisesRegex(OSError, "fsync failed"):
                    store.write_artifact(
                        "demo",
                        ArtifactRequest(path="new.json", data={"value": 1}, create_only=True),
                    )

            self.assertFalse(path.exists())

    def test_artifact_create_only_is_invisible_until_complete_temp_is_published(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            destination = Path(temp_dir) / "demo" / "new.json"
            publish_started = threading.Event()
            allow_publish = threading.Event()
            captured = []
            real_link = __import__("os").link

            def observed_link(source, target):
                captured.append(Path(source).read_bytes())
                publish_started.set()
                allow_publish.wait(1)
                real_link(source, target)

            result = []
            with patch("feverslop.studio.projects.os.link", side_effect=observed_link):
                worker = threading.Thread(
                    target=lambda: result.append(store.write_artifact(
                        "demo",
                        ArtifactRequest(path="new.json", data={"complete": True}, create_only=True),
                    )),
                )
                worker.start()
                self.assertTrue(publish_started.wait(1))
                self.assertFalse(destination.exists())
                self.assertEqual({"complete": True}, json.loads(captured[0]))
                allow_publish.set()
                worker.join(1)

            self.assertFalse(worker.is_alive())
            self.assertEqual({"complete": True}, json.loads(destination.read_text()))
            self.assertFalse(any(destination.parent.glob(f".{destination.name}.*.tmp")))

    def test_artifact_create_only_preserves_existing_target_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            destination = Path(temp_dir) / "demo" / "existing.json"
            original = b'{"owner":"existing"}'
            destination.write_bytes(original)

            with self.assertRaisesRegex(Exception, "already exists"):
                store.write_artifact(
                    "demo",
                    ArtifactRequest(path="existing.json", data={"owner": "ours"}, create_only=True),
                )

            self.assertEqual(original, destination.read_bytes())
            self.assertFalse(any(destination.parent.glob(f".{destination.name}.*.tmp")))

    def test_artifact_locking_module_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("feverslop.studio.artifact_locking"))

    def test_artifact_locks_are_released_when_no_writer_uses_them(self):
        from feverslop.studio import artifact_locking

        path = Path("artifact.json").resolve()
        with artifact_locking._LOCKS_GUARD:
            lock = artifact_locking._LOCKS.setdefault(path, threading.RLock())
        lock_reference = weakref.ref(lock)

        del lock
        gc.collect()

        self.assertIsNone(lock_reference())
        self.assertNotIn(path, artifact_locking._LOCKS)

    def test_config_write_validates_required_fields_and_preserves_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            store.write_artifact(
                "demo",
                ArtifactRequest(
                    path="config.json",
                    data={
                        "project_name": "Changed",
                        "input_audio": "input/song.mp3",
                        "subject_mode": "single",
                        "max_scene_actors": 1,
                        "custom_plugin": {"empty_but_intentional": ""},
                    },
                ),
            )

            config = json.loads((Path(temp_dir) / "demo" / "config.json").read_text())
            self.assertEqual({"empty_but_intentional": ""}, config["custom_plugin"])
            with self.assertRaises(ValueError):
                store.write_artifact(
                    "demo",
                    ArtifactRequest(path="config.json", data={"project_name": "", "input_audio": "input/song.mp3"}),
                )
            with self.assertRaises(ValueError):
                store.write_artifact(
                    "demo",
                    ArtifactRequest(path="config.json", data={"project_name": "Demo", "input_audio": "input/song.mp3", "subject_mode": "group"}),
                )

    def test_config_write_rejects_non_boolean_silent_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            with self.assertRaises(ValueError):
                store.write_artifact(
                    "demo",
                    ArtifactRequest(
                        path="config.json",
                        data={"project_name": "Demo", "input_audio": "input/song.mp3", "silent_mode": "true"},
                    ),
                )

    def test_config_write_defaults_null_silent_mode_to_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            store.write_artifact(
                "demo",
                ArtifactRequest(
                    path="config.json",
                    data={"project_name": "Demo", "input_audio": "input/song.mp3", "silent_mode": None},
                ),
            )

            config = json.loads((Path(temp_dir) / "demo" / "config.json").read_text())
            self.assertFalse(config["silent_mode"])

    def test_patch_render_plan_updates_selected_scene_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            updated = store.patch_render_plan(
                "demo",
                RenderPlanPatch(
                    path="output/render/render_plan_song.json",
                    scene=1,
                    updates={"prompt": "new prompt", "actor_references": ["hero"]},
                ),
            )

            self.assertEqual("new prompt", updated["scene"]["prompt"])
            self.assertEqual(["hero"], updated["scene"]["actor_references"])

    def test_patch_render_plan_rejects_stale_expected_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            relative = "output/render/render_plan_song.json"
            loaded = store.read_artifact("demo", relative)
            path = Path(temp_dir) / "demo" / relative
            external = b'[{"scene":1,"prompt":"external"}]'
            path.write_bytes(external)

            with self.assertRaisesRegex(Exception, "changed"):
                store.patch_render_plan(
                    "demo",
                    RenderPlanPatch(
                        path=relative,
                        scene=1,
                        updates={"prompt": "ours"},
                        expected_revision=loaded["revision"],
                    ),
                )

            self.assertEqual(external, path.read_bytes())

    def test_patch_render_plan_with_current_revision_returns_new_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            relative = "output/render/render_plan_song.json"
            loaded = store.read_artifact("demo", relative)

            updated = store.patch_render_plan(
                "demo",
                RenderPlanPatch(
                    path=relative,
                    scene=1,
                    updates={"prompt": "ours"},
                    expected_revision=loaded["revision"],
                ),
            )

            self.assertEqual(store.read_artifact("demo", relative)["revision"], updated["revision"])

    def test_patch_render_plan_uses_shared_artifact_write_lock(self):
        from feverslop.studio.artifact_locking import artifact_write_lock

        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            path = Path(temp_dir) / "demo" / "output" / "render" / "render_plan_song.json"
            started = threading.Event()
            finished = threading.Event()

            def patch_plan():
                started.set()
                store.patch_render_plan(
                    "demo",
                    RenderPlanPatch(path="output/render/render_plan_song.json", scene=1, updates={"prompt": "new"}),
                )
                finished.set()

            with artifact_write_lock(path):
                worker = threading.Thread(target=patch_plan)
                worker.start()
                self.assertTrue(started.wait(1))
                self.assertFalse(finished.wait(0.05))
            worker.join(1)

            self.assertTrue(finished.is_set())

    def test_scene_and_raw_revision_writers_cannot_lose_update(self):
        from feverslop.adapters.project_scene_documents import ProjectSceneDocuments
        from feverslop.studio.artifact_catalog import ArtifactCatalog

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = self._project_store(root)
            path = "output/render/render_plan_song.json"
            loaded = store.read_artifact("demo", path)
            documents = ProjectSceneDocuments(
                store.project_root,
                catalog=ArtifactCatalog(store.project_root),
            )
            barrier = threading.Barrier(2)
            results = []

            def raw_write():
                barrier.wait()
                try:
                    results.append(store.write_artifact(
                        "demo",
                        ArtifactRequest(
                            path=path,
                            data=[{"scene": 1, "prompt": "raw"}],
                            expected_revision=loaded["revision"],
                        ),
                    ))
                except Exception as exc:  # noqa: BLE001 - records expected loser
                    results.append(exc)

            def scene_write():
                barrier.wait()
                try:
                    results.append(documents.patch_scene(
                        "demo",
                        1,
                        {"shot_description": "scene"},
                        loaded["revision"],
                    ))
                except Exception as exc:  # noqa: BLE001 - records expected loser
                    results.append(exc)

            workers = [threading.Thread(target=raw_write), threading.Thread(target=scene_write)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            failures = [result for result in results if isinstance(result, Exception)]
            self.assertEqual(1, len(failures))
            self.assertIn(type(failures[0]).__name__, {"ArtifactConflict", "SceneDocumentConflict"})

    def test_media_path_must_stay_inside_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            media_path = store.resolve_media_path("demo", "output/references/still.png")

            self.assertTrue(media_path.name.endswith(".png"))
            with self.assertRaises(StudioPathError):
                store.resolve_media_path("demo", "../../etc/passwd")

    def test_write_media_data_url_stays_inside_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            result = store.write_media_data_url("demo", "output/references/actors/hero/sheet.png", "data:image/png;base64,cG5n")

            self.assertEqual("output/references/actors/hero/sheet.png", result["path"])
            self.assertEqual(b"png", (Path(temp_dir) / "demo" / "output/references/actors/hero/sheet.png").read_bytes())
            with self.assertRaises(StudioPathError):
                store.write_media_data_url("demo", "../bad.png", "data:image/png;base64,cG5n")

    def test_write_media_data_url_rejects_oversized_payload(self):
        def _project_store_with_limit(root, max_upload_size):
            project = root / "demo"
            (project / "output").mkdir(parents=True, exist_ok=True)
            (project / "input").mkdir(exist_ok=True)
            (project / "config.json").write_text('{"project_name": "Demo"}', encoding="utf-8")
            return ProjectStore(root, max_upload_size=max_upload_size)

        with tempfile.TemporaryDirectory() as temp_dir:
            # 2-byte limit; payload decodes to 3 bytes ("png")
            store = _project_store_with_limit(Path(temp_dir), max_upload_size=2)
            with self.assertRaisesRegex(StudioPathError, "exceeds size limit"):
                store.write_media_data_url("demo", "output/big.png", "data:image/png;base64,cG5n")

            # 3-byte limit is exactly the decoded size; should succeed
            store2 = _project_store_with_limit(Path(temp_dir), max_upload_size=3)
            result = store2.write_media_data_url("demo", "output/exact.png", "data:image/png;base64,cG5n")
            self.assertEqual("output/exact.png", result["path"])

    def test_audio_upload_endpoint_stores_file_in_input_and_updates_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(ProjectCreateRequest(project_type="standard_music_video", name="Demo"))
            client = NativeStudioHarness(temp_dir)

            response = client.post(
                "/api/projects/demo/upload-audio",
                files={"file": ("../My Song.mp3", b"audio bytes", "audio/mpeg")},
            )

            self.assertEqual(200, response.status_code)
            self.assertEqual({"path": "input/My_Song.mp3"}, response.json())
            self.assertEqual(b"audio bytes", (Path(temp_dir) / "demo" / "input" / "My_Song.mp3").read_bytes())
            config = json.loads((Path(temp_dir) / "demo" / "config.json").read_text())
            self.assertEqual("input/My_Song.mp3", config["input_audio"])

    def test_audio_upload_endpoint_rejects_non_audio_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(ProjectCreateRequest(project_type="standard_music_video", name="Demo"))
            client = NativeStudioHarness(temp_dir)

            response = client.post(
                "/api/projects/demo/upload-audio",
                files={"file": ("notes.txt", b"not audio", "text/plain")},
            )

            self.assertEqual(400, response.status_code)
            self.assertIn("Unsupported audio type", response.json()["detail"])

    def test_job_registry_tracks_success_and_failure(self):
        registry = JobRegistry()

        ok = registry.start("demo", "ok", lambda log: log("done") or "result")
        bad = registry.start("demo", "bad", lambda _log: (_ for _ in ()).throw(RuntimeError("boom")))

        for _ in range(50):
            if all(
                registry.get(job_id)["status"] not in {"queued", "running"}
                for job_id in (ok, bad)
            ):
                break
            time.sleep(0.01)

        self.assertEqual("succeeded", registry.get(ok)["status"])
        self.assertEqual("failed", registry.get(bad)["status"])
        self.assertIn("done", registry.get(ok)["logs"])
        self.assertIn("boom", registry.get(bad)["error"])

    def test_job_registry_limits_parallel_jobs_to_configured_workers(self):
        registry = JobRegistry(max_workers=2)
        release = threading.Event()
        started = 0
        started_lock = threading.Lock()

        def blocking_job(_log):
            nonlocal started
            with started_lock:
                started += 1
            release.wait(1.0)

        try:
            job_ids = [registry.start("demo", f"job-{index}", blocking_job) for index in range(4)]
            for _ in range(100):
                with started_lock:
                    if started >= 2:
                        break
                time.sleep(0.005)

            with started_lock:
                self.assertEqual(2, started)
            self.assertEqual(
                2,
                sum(registry.get(job_id)["status"] == "queued" for job_id in job_ids),
            )
        finally:
            release.set()
            shutdown = getattr(registry, "shutdown", None)
            if shutdown is not None:
                shutdown(wait=True)

    def test_job_registry_returns_isolated_nested_snapshots(self):
        registry = JobRegistry()
        job_id = registry.start("demo", "snapshot", lambda log: log("original"))
        for _ in range(100):
            if registry.get(job_id)["status"] == "succeeded":
                break
            time.sleep(0.005)

        snapshot = registry.get(job_id)
        snapshot["logs"].append("external")
        snapshot["steps"][0]["status"] = "corrupted"
        listed = registry.list("demo")[0]
        listed["logs"].append("listed-external")

        current = registry.get(job_id)
        self.assertEqual(["original"], current["logs"])
        self.assertEqual("completed", current["steps"][0]["status"])

    def test_job_registry_serializes_steps_elapsed_and_recent_logs(self):
        registry = JobRegistry()

        job_id = registry.start("demo", "full-pipeline", lambda log: log("line") or "result")

        for _ in range(50):
            if registry.get(job_id)["status"] == "succeeded":
                break
            time.sleep(0.01)

        job = registry.get(job_id)
        self.assertEqual("full-pipeline", job["pipeline_type"])
        self.assertGreaterEqual(job["elapsed_seconds"], 0)
        self.assertIn("steps", job)
        self.assertTrue(any(step["status"] == "completed" for step in job["steps"]))
        self.assertEqual(["line"], job["recent_logs"])

    def test_ltx_scene_progress_and_finished_action_advance_ltx_step(self):
        registry = JobRegistry()
        job = {"steps": registry._initial_steps("ltx-render-scenes"), "progress": 0}

        registry._advance_step_from_log(job, "Rendered scene 1/3")

        step = job["steps"][0]
        self.assertEqual("Video render", step["name"])
        self.assertEqual("running", step["status"])
        self.assertEqual(33, step["progress"])

        registry._advance_step_from_log(job, "Finished ltx-render-scenes")

        self.assertEqual("completed", step["status"])
        self.assertEqual(100, job["overall_progress"])

    def test_full_pipeline_stage_logs_advance_progress_for_classic_mode(self):
        registry = JobRegistry()
        job = {"steps": registry._initial_steps("full-pipeline", pipeline_mode="classic"), "progress": 0}

        registry._advance_step_from_log(job, "Starting full-pipeline")
        registry._advance_step_from_log(job, "==> Stage Main pipeline")
        registry._advance_step_from_log(job, "==> Stage Storyboard frames")

        self.assertEqual(["Main pipeline", "Storyboard", "Video render", "Final concat"], [step["name"] for step in job["steps"]])
        self.assertEqual("completed", job["steps"][0]["status"])
        self.assertEqual("running", job["steps"][1]["status"])
        self.assertEqual(25, job["overall_progress"])

    def test_full_pipeline_omits_msr_steps_for_classic_mode(self):
        registry = JobRegistry()

        steps = registry._initial_steps("full-pipeline", pipeline_mode="classic")

        self.assertNotIn("MSR references", [step["name"] for step in steps])
        self.assertNotIn("MSR enrichment", [step["name"] for step in steps])

    def test_full_pipeline_steps_for_minimax_modes(self):
        registry = JobRegistry()

        r2v_steps = registry._initial_steps("full-pipeline", pipeline_mode="minimax_h3_r2v")
        t2v_steps = registry._initial_steps("full-pipeline", pipeline_mode="minimax_h3_t2v")

        r2v_step_names = [step["name"] for step in r2v_steps]
        self.assertEqual(["Main pipeline", "MSR references", "Video render", "Final concat"], r2v_step_names)
        self.assertNotIn("Storyboard", r2v_step_names)

        t2v_step_names = [step["name"] for step in t2v_steps]
        self.assertEqual(["Main pipeline", "Video render", "Final concat"], t2v_step_names)
        self.assertNotIn("MSR references", t2v_step_names)
        self.assertNotIn("Storyboard", t2v_step_names)

    def test_pipeline_option_builder_uses_minimax_pipeline_mode(self):
        r2v = build_pipeline_options("full-pipeline", pipeline_mode="minimax_h3_r2v")
        t2v = build_pipeline_options("full-pipeline", pipeline_mode="minimax_h3_t2v")

        self.assertEqual("minimax-h3-r2v", r2v["video_pipeline"])
        self.assertEqual("minimax-h3-t2v", t2v["video_pipeline"])
        self.assertFalse(r2v["skip_msr_reference_render"])
        self.assertTrue(r2v["skip_msr_prompt_enrichment"])
        self.assertTrue(r2v["skip_ingredients_sheets"])
        self.assertTrue(t2v["skip_msr_reference_render"])
        self.assertTrue(t2v["skip_msr_prompt_enrichment"])
        self.assertTrue(t2v["skip_ingredients_sheets"])

    def test_job_registry_sanitizes_rich_logs_and_tracks_acestep_step(self):
        registry = JobRegistry()

        def handler(log):
            console = _StudioFullAutoConsole(log)
            console.print(Panel.fit("[green]OK[/green] Generated audio: [cyan]/tmp/song.mp3[/cyan]", title="Audio"))
            console.rule("[bold cyan]2. Rendering ACE-Step audio[/bold cyan]")

        job_id = registry.start("demo", "full-auto", handler)

        for _ in range(50):
            if registry.get(job_id)["status"] == "succeeded":
                break
            time.sleep(0.01)

        job = registry.get(job_id)
        text = "\n".join(job["logs"])
        self.assertNotIn("<rich.", text)
        self.assertNotIn("[green]", text)
        self.assertIn("OK Generated audio: /tmp/song.mp3", text)
        self.assertIn("2. Rendering ACE-Step audio", text)
        self.assertTrue(any(step["name"] == "ACE-Step audio rendering" for step in job["steps"]))

    def test_stream_capture_sanitizes_stdout_and_stderr(self):
        lines = []

        def noisy():
            print("[green]OK[/green] stdout")
            import sys

            from rich.console import Console

            Console(file=sys.stderr, force_terminal=False, width=120).print(Panel.fit("[cyan]stderr panel[/cyan]"))
            return "result"

        result = run_with_stream_logging(noisy, lines.append)

        self.assertEqual("result", result)
        text = "\n".join(lines)
        self.assertIn("OK stdout", text)
        self.assertIn("stderr panel", text)
        self.assertNotIn("[green]", text)
        self.assertNotIn("<rich.", text)

    def test_job_registry_rejects_duplicate_pipeline_start_for_project(self):
        registry = JobRegistry()
        gate = threading.Event()

        first = registry.start("demo", "full-pipeline", lambda log: gate.wait(0.5))
        for _ in range(50):
            if registry.get(first)["status"] == "running":
                break
            time.sleep(0.01)

        with self.assertRaises(ValueError):
            registry.start("demo", "full-pipeline", lambda _log: None, reject_if_project_active=True)
        gate.set()

    def test_pipeline_option_builder_maps_actions_to_skip_flags(self):
        options = build_pipeline_options("ltx-render-scenes", scenes=[2, 4])

        self.assertTrue(options["skip_tests"])
        self.assertTrue(options["skip_main_pipeline"])
        self.assertTrue(options["skip_storyboard"])
        self.assertFalse(options["skip_ltx"])
        self.assertEqual("2,4", options["scenes"])

    def test_pipeline_option_builder_prepares_only_selected_ltx_workflows(self):
        options = build_pipeline_options("ltx-prepare-workflows", scenes=[2, 4])

        self.assertEqual(["ltx_prepare_workflows"], options["stages"])
        self.assertFalse(options["skip_ltx"])
        self.assertEqual("2,4", options["scenes"])

    def test_pipeline_option_builder_maps_atomic_actions_to_stages(self):
        self.assertEqual(["anchor_fix"], build_pipeline_options("anchor-fix")["stages"])
        self.assertEqual(["relay_compact"], build_pipeline_options("relay-compact")["stages"])
        self.assertEqual(["storyboard_frames"], build_pipeline_options("storyboard-frames")["stages"])
        self.assertEqual(["storyboard_page"], build_pipeline_options("storyboard-page")["stages"])
        self.assertEqual(["msr_reference_sheets"], build_pipeline_options("msr-reference-sheets")["stages"])
        self.assertEqual(["msr_prompt_enrich"], build_pipeline_options("msr-prompt-enrich")["stages"])
        self.assertEqual(["concat_video_only"], build_pipeline_options("concat-video-only")["stages"])
        self.assertEqual(["mux_original_audio"], build_pipeline_options("mux-original-audio")["stages"])
        self.assertEqual(["msr_reference_sheets", "msr_prompt_enrich"], build_pipeline_options("rebuild-plan")["stages"])

    def test_pipeline_option_builder_uses_selected_pipeline_mode(self):
        classic = build_pipeline_options("full-pipeline", pipeline_mode="classic")
        msr = build_pipeline_options("full-pipeline", pipeline_mode="msr")

        self.assertEqual("ltx_i2v", classic["video_pipeline"])
        self.assertEqual("ltx_msr", msr["video_pipeline"])
        self.assertFalse(msr["skip_msr_reference_render"])
        self.assertFalse(msr["skip_msr_prompt_enrichment"])

    def test_build_ffmpeg_recut_command_trims_raw_clip_to_output_clip(self):
        command = build_ffmpeg_recut_command(Path("raw.mp4"), Path("final.mp4"), raw_in_seconds=0.4, raw_out_seconds=11.9)

        self.assertEqual(
            ["ffmpeg", "-y", "-ss", "0.400", "-to", "11.900", "-i", "raw.mp4", "-c", "copy", "final.mp4"],
            command,
        )

    def test_build_ffmpeg_recut_command_supports_exact_reencode(self):
        command = build_ffmpeg_recut_command(Path("raw.mp4"), Path("final.mp4"), raw_in_seconds=0.4, raw_out_seconds=11.9, exact=True)

        self.assertIn("-c:v", command)
        self.assertIn("libx264", command)
        self.assertLess(command.index("-i"), command.index("-ss"))

    def test_api_creates_projects_and_starts_full_auto_job(self):
        calls = []

        def fake_full_auto_handler(*, store, project_id, payload):
            def run(log):
                calls.append((project_id, payload["idea"], payload["song_style"]))
                log("full auto started")
                return "ok"

            return run

        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir, full_auto_handler=fake_full_auto_handler)
            created = client.post(
                "/api/projects",
                json={
                    "project_type": "full_auto",
                    "name": "Neon Wolves",
                    "idea": "A cyberpunk chase",
                    "song_style": "dark synthwave",
                },
            )
            self.assertEqual(200, created.status_code, created.text)
            self.assertEqual("neon-wolves", created.json()["id"])
            duplicate = client.post(
                "/api/projects",
                json={
                    "project_type": "full_auto",
                    "name": "Neon Wolves",
                    "idea": "x",
                    "song_style": "y",
                },
            )
            self.assertEqual(400, duplicate.status_code)

            job_response = client.post("/api/projects/neon-wolves/jobs", json={"action": "full-auto"})
            self.assertEqual(200, job_response.status_code, job_response.text)
            job_id = job_response.json()["id"]
            for _ in range(50):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "succeeded":
                    break
                time.sleep(0.01)
            self.assertEqual([("neon-wolves", "A cyberpunk chase", "dark synthwave")], calls)
            state = json.loads((Path(temp_dir) / "neon-wolves" / ".studio" / "pipeline_state.json").read_text())
            self.assertEqual("full-auto", state["last_run"]["action"])
            self.assertEqual("succeeded", state["last_run"]["status"])

    def test_api_create_accepts_and_validates_silent_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)

            created = client.post(
                "/api/projects",
                json={"project_type": "standard_music_video", "name": "Silent Video", "silent_mode": True},
            )
            invalid = client.post(
                "/api/projects",
                json={"project_type": "standard_music_video", "name": "Bad Silent", "silent_mode": "true"},
            )

            self.assertEqual(200, created.status_code, created.text)
            self.assertTrue(created.json()["silent_mode"])
            self.assertEqual(400, invalid.status_code)

    def test_create_movie_project_persists_i2v_continuity_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            store.create_project(
                ProjectCreateRequest(
                    project_type="movie",
                    name="Door Below",
                    story_text="A locksmith follows a door that opens beneath the abandoned station.",
                    movie_mode="scaffold",
                    movie_planner_backend="deterministic",
                    movie_video_workflow="msr-i2v-startframe",
                    movie_continuity_keyframes="last-to-start",
                    movie_msr_i2v_workflow="workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json",
                ),
            )

            metadata = json.loads((Path(temp_dir) / "door-below" / ".studio" / "project.json").read_text())

            self.assertEqual("msr-i2v-startframe", metadata["movie"]["movie_video_workflow"])
            self.assertEqual("last-to-start", metadata["movie"]["continuity_keyframes"])
            self.assertEqual("workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json", metadata["movie"]["msr_i2v_workflow"])

    def test_create_movie_project_persists_i2v_edit_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            store.create_project(
                ProjectCreateRequest(
                    project_type="movie",
                    name="I2V Movie",
                    story_text="A witch traps a hiker in a forest.",
                    movie_mode="scaffold",
                    movie_planner_backend="deterministic",
                    movie_video_workflow="i2v-edit",
                ),
            )

            metadata = json.loads((Path(temp_dir) / "i2v-movie" / ".studio" / "project.json").read_text())

            self.assertEqual("i2v-edit", metadata["movie"]["movie_video_workflow"])
            self.assertEqual("workflows/image_edit_flux2_klein_2ref_v1.json", metadata["movie"]["edit_workflow"])

    def test_create_movie_project_persists_startframe_director_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            store.create_project(
                ProjectCreateRequest(
                    project_type="movie",
                    name="Director Movie",
                    story_text="An archivist opens a sealed ledger.",
                    movie_mode="scaffold",
                    movie_planner_backend="deterministic",
                    movie_video_workflow="startframe-director",
                ),
            )

            metadata = json.loads((Path(temp_dir) / "director-movie" / ".studio" / "project.json").read_text())

            self.assertEqual("startframe-director", metadata["movie"]["movie_video_workflow"])
            self.assertEqual("krea2", metadata["movie"]["startframe_director_backend"])
            self.assertEqual("workflows/image_t2i_startframe_krea_v1.json", metadata["movie"]["director_workflow"])
            self.assertEqual("workflows/image_mask_sam3_actor_regions_v1.json", metadata["movie"]["mask_workflow"])
            self.assertEqual("workflows/image_repair_sdxl_ipadapter_identity_v1.json", metadata["movie"]["identity_repair_workflow"])
            self.assertEqual("workflows/image_detail_easyuse_startframe_v1.json", metadata["movie"]["detail_workflow"])
            self.assertEqual("http://localhost:8188", metadata["movie"]["startframe_comfyui_base_url"])
            self.assertFalse(metadata["movie"]["startframe_write_debug_workflows"])
            self.assertEqual("http://your-llm-server.local/v1", metadata["movie"]["startframe_validator_base_url"])
            self.assertEqual("gemma4-26b-a4b:vision", metadata["movie"]["startframe_validator_model"])

    def test_create_movie_project_uses_ideogram_workflow_for_ideogram_director_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            store.create_project(
                ProjectCreateRequest(
                    project_type="movie",
                    name="Ideogram Director Movie",
                    story_text="An archivist opens a sealed ledger.",
                    movie_mode="scaffold",
                    movie_planner_backend="deterministic",
                    movie_video_workflow="startframe-director",
                    movie_startframe_director_backend="ideogram",
                ),
            )

            metadata = json.loads((Path(temp_dir) / "ideogram-director-movie" / ".studio" / "project.json").read_text())

            self.assertEqual("ideogram", metadata["movie"]["startframe_director_backend"])
            self.assertEqual("workflows/image_t2i_startframe_ideogram_director_v1.json", metadata["movie"]["director_workflow"])

    def test_create_movie_project_persists_startframe_debug_workflow_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)

            store.create_project(
                ProjectCreateRequest(
                    project_type="movie",
                    name="Debug Director Movie",
                    story_text="An archivist opens a sealed ledger.",
                    movie_mode="scaffold",
                    movie_planner_backend="deterministic",
                    movie_video_workflow="startframe-director",
                    movie_startframe_write_debug_workflows=True,
                    movie_startframe_debug_workflows_dir="debug/startframes",
                ),
            )

            metadata = json.loads((Path(temp_dir) / "debug-director-movie" / ".studio" / "project.json").read_text())

            self.assertTrue(metadata["movie"]["startframe_write_debug_workflows"])
            self.assertEqual("debug/startframes", metadata["movie"]["startframe_debug_workflows_dir"])

    def test_build_full_auto_handler_passes_render_inputs_and_pipeline_mode(self):
        captured = {}

        class FakeUseCase:
            def execute(self, request):
                captured["request"] = request
                print("[green]OK[/green] nested full-auto pipeline")

                class Result:
                    project_config_path = Path("config.json")

                return Result()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(
                ProjectCreateRequest(
                    project_type="full_auto",
                    name="Neon Wolves",
                    idea="A cyberpunk chase",
                    song_style="dark synthwave",
                    duration_seconds=90,
                    width=1280,
                    height=704,
                    fps=16,
                    silent_mode=True,
                    pipeline_mode="classic",
                ),
            )
            handler = build_full_auto_handler(
                store=store,
                project_id="neon-wolves",
                payload=store.project_metadata("neon-wolves")["full_auto"],
                use_case_factory=lambda console: FakeUseCase(),
            )

            logs = []
            handler(logs.append)

        request = captured["request"]
        self.assertEqual(90, request.duration_seconds)
        self.assertEqual(1280, request.width)
        self.assertEqual(704, request.height)
        self.assertEqual(16, request.fps)
        self.assertTrue(request.silent_mode)
        self.assertEqual("ltx_i2v", request.runner_options["video_pipeline"])
        self.assertIn("OK nested full-auto pipeline", "\n".join(logs))

    def test_api_starts_standard_pipeline_job(self):
        calls = []

        def fake_pipeline_handler(config_path, action, *, scenes=None, pipeline_mode=None):
            def run(log):
                calls.append((Path(config_path).name, action, scenes, pipeline_mode))
                log("standard pipeline started")
                return "ok"

            return run

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(ProjectCreateRequest(project_type="standard_music_video", name="My Cool Video"))
            client = NativeStudioHarness(temp_dir, pipeline_handler=fake_pipeline_handler)

            job_response = client.post("/api/projects/my-cool-video/jobs", json={"action": "full-pipeline"})

            self.assertEqual(200, job_response.status_code, job_response.text)
            job_id = job_response.json()["id"]
            for _ in range(50):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "succeeded":
                    break
                time.sleep(0.01)
            self.assertEqual([("config.json", "full-pipeline", None, "classic")], calls)

    def test_api_atomic_pipeline_job_records_completed_stage(self):
        def fake_pipeline_handler(config_path, action, *, scenes=None, pipeline_mode=None):
            return lambda log: log(f"ran {action}") or "ok"

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(ProjectCreateRequest(project_type="standard_music_video", name="My Cool Video"))
            client = NativeStudioHarness(temp_dir, pipeline_handler=fake_pipeline_handler)

            job_response = client.post("/api/projects/my-cool-video/jobs", json={"action": "anchor-fix"})

            self.assertEqual(200, job_response.status_code, job_response.text)
            job_id = job_response.json()["id"]
            for _ in range(50):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] == "succeeded":
                    break
                time.sleep(0.01)
            state = json.loads((Path(temp_dir) / "my-cool-video" / ".studio" / "pipeline_state.json").read_text())
            self.assertEqual(["anchor_fix"], state["completed_stages"])
            self.assertEqual("anchor-fix", state["last_run"]["action"])

    def test_api_rejects_duplicate_pipeline_start_while_running(self):
        gate = threading.Event()

        def fake_pipeline_handler(config_path, action, *, scenes=None, pipeline_mode=None):
            return lambda _log: gate.wait(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(ProjectCreateRequest(project_type="standard_music_video", name="My Cool Video"))
            client = NativeStudioHarness(temp_dir, pipeline_handler=fake_pipeline_handler)

            first = client.post("/api/projects/my-cool-video/jobs", json={"action": "full-pipeline"})
            self.assertEqual(200, first.status_code, first.text)
            duplicate = client.post("/api/projects/my-cool-video/jobs", json={"action": "full-pipeline"})
            gate.set()
            for _ in range(50):
                if client.get(f"/api/jobs/{first.json()['id']}").json()["status"] == "succeeded":
                    break
                time.sleep(0.01)

            self.assertEqual(400, duplicate.status_code)
            self.assertIn("Pipeline is already running", duplicate.text)

    def test_api_exposes_process_state_for_active_project_jobs(self):
        gate = threading.Event()

        def fake_pipeline_handler(config_path, action, *, scenes=None, pipeline_mode=None):
            return lambda _log: gate.wait(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProjectStore(temp_dir)
            store.create_project(ProjectCreateRequest(project_type="standard_music_video", name="My Cool Video"))
            client = NativeStudioHarness(temp_dir, pipeline_handler=fake_pipeline_handler)

            first = client.post("/api/projects/my-cool-video/jobs", json={"action": "rebuild-plan"})
            self.assertEqual(200, first.status_code, first.text)
            for _ in range(50):
                processes = client.get("/api/processes?project_id=my-cool-video").json()
                if processes and processes[0]["status"] == "running":
                    break
                time.sleep(0.01)
            gate.set()
            for _ in range(50):
                if client.get(f"/api/jobs/{first.json()['id']}").json()["status"] == "succeeded":
                    break
                time.sleep(0.01)

            self.assertEqual("rebuild-plan", processes[0]["action"])
            self.assertEqual("running", processes[0]["status"])
            self.assertIn("steps", processes[0])

    def test_pipeline_mode_from_config_maps_minimax_to_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_r2v = base / "config_r2v.json"
            config_r2v.write_text(json.dumps({"video_pipeline": "minimax-h3-r2v"}), encoding="utf-8")
            config_t2v = base / "config_t2v.json"
            config_t2v.write_text(json.dumps({"video_pipeline": "minimax-h3-t2v"}), encoding="utf-8")
            config_empty = base / "config_empty.json"
            config_empty.write_text(json.dumps({}), encoding="utf-8")

            self.assertEqual("minimax_h3_r2v", pipeline_mode_from_config(config_r2v))
            self.assertEqual("minimax_h3_t2v", pipeline_mode_from_config(config_t2v))
            self.assertIsNone(pipeline_mode_from_config(config_empty))


if __name__ == "__main__":
    unittest.main()
