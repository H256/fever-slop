import json
import tempfile
import time
import unittest
from pathlib import Path

from feverslop.studio.jobs import JobRegistry, build_ffmpeg_recut_command, build_pipeline_options
from feverslop.studio.projects import (
    ArtifactRequest,
    ProjectStore,
    RenderPlanPatch,
    StudioPathError,
)


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
                ]
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

    def test_thumbnail_cache_is_not_reported_as_project_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            cache = Path(temp_dir) / "demo" / ".studio" / "thumbnails"
            cache.mkdir(parents=True)
            (cache / "scene.jpg").write_bytes(b"cache")

            project = store.describe_project("demo")

            self.assertNotIn(".studio/thumbnails/scene.jpg", project["artifacts"]["images"])
            self.assertEqual(0, project["artifact_sizes"]["by_type"]["images"])

    def test_clear_thumbnail_cache_removes_cached_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))
            cache = store.thumbnail_cache_path("demo", "scene")
            cache.write_bytes(b"jpg")

            removed = store.clear_thumbnail_cache("demo")

            self.assertEqual(1, removed)
            self.assertFalse(cache.exists())

    def test_artifact_read_write_is_project_relative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._project_store(Path(temp_dir))

            artifact = store.read_artifact("demo", "config.json")
            store.write_artifact("demo", ArtifactRequest(path="config.json", data={**artifact["data"], "project_name": "Changed"}))

            self.assertEqual("Changed", json.loads((Path(temp_dir) / "demo" / "config.json").read_text())["project_name"])
            with self.assertRaises(StudioPathError):
                store.read_artifact("demo", "../outside.json")

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

    def test_job_registry_tracks_success_and_failure(self):
        registry = JobRegistry()

        ok = registry.start("demo", "ok", lambda log: log("done") or "result")
        bad = registry.start("demo", "bad", lambda _log: (_ for _ in ()).throw(RuntimeError("boom")))

        for _ in range(50):
            if registry.get(ok)["status"] != "running" and registry.get(bad)["status"] != "running":
                break
            time.sleep(0.01)

        self.assertEqual("succeeded", registry.get(ok)["status"])
        self.assertEqual("failed", registry.get(bad)["status"])
        self.assertIn("done", registry.get(ok)["logs"])
        self.assertIn("boom", registry.get(bad)["error"])

    def test_pipeline_option_builder_maps_actions_to_skip_flags(self):
        options = build_pipeline_options("ltx-render-scenes", scenes=[2, 4])

        self.assertTrue(options["skip_tests"])
        self.assertTrue(options["skip_main_pipeline"])
        self.assertTrue(options["skip_storyboard"])
        self.assertFalse(options["skip_ltx"])
        self.assertEqual("2,4", options["scenes"])

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


if __name__ == "__main__":
    unittest.main()
