import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from feverslop.studio.server import create_app


class MovieProjectTests(unittest.TestCase):
    def test_movie_orchestrator_scaffolds_story_arch_and_render_plan(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
            ).execute(
                MovieInput(
                    name="Door Below",
                    source_type="short_story",
                    story_text="A locksmith finds a glowing door below an abandoned station.",
                    desired_length=30,
                    width=1280,
                    height=704,
                    mode="scaffold",
                )
            )

            self.assertEqual("door-below", result.project_slug)
            self.assertTrue((Path(temp_dir) / "door-below" / "movie" / "story_arch.json").exists())
            self.assertTrue((Path(temp_dir) / "door-below" / "movie" / "render_plan.json").exists())
            render_plan = json.loads((Path(temp_dir) / "door-below" / "movie" / "render_plan.json").read_text())
            self.assertGreaterEqual(len(render_plan["shots"]), 1)
            self.assertEqual({"width": 1280, "height": 704}, render_plan["resolution"])

    def test_movie_workflow_patcher_removes_audio_inputs(self):
        from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

        workflow = {
            "1": {"class_type": "LoadAudio", "_meta": {"title": "#LOAD_AUDIO"}, "inputs": {"audio": "song.mp3"}},
            "2": {"class_type": "TrimAudio", "_meta": {"title": "#TRIM_AUDIO"}, "inputs": {"audio": ["1", 0], "duration": 3.0}},
            "3": {"class_type": "LTXVideo", "_meta": {"title": "#LTX"}, "inputs": {"audio": ["2", 0], "prompt": "movie"}},
        }

        patched = MovieWorkflowPatcher().strip_audio_inputs(workflow)

        self.assertNotIn("1", patched)
        self.assertNotIn("2", patched)
        self.assertNotIn("audio", patched["3"]["inputs"])

    def test_api_creates_movie_project_with_scaffold_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(temp_dir))

            response = client.post(
                "/api/projects",
                json={
                    "project_type": "movie",
                    "name": "Door Below",
                    "source_type": "short_story",
                    "story_text": "A locksmith finds a glowing door below an abandoned station.",
                    "desired_length": 30,
                    "width": 1280,
                    "height": 704,
                    "movie_mode": "scaffold",
                },
            )

            self.assertEqual(200, response.status_code, response.text)
            project = response.json()
            self.assertEqual("movie", project["project_type"])
            self.assertEqual("present", project["status"]["render_plan"])
            self.assertIn("movie/story_arch.json", project["artifacts"]["generated_json"])
            self.assertIn("movie/render_plan.json", project["artifacts"]["render_plans"])

    def test_api_rejects_invalid_screenplay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(temp_dir))

            response = client.post(
                "/api/projects",
                json={
                    "project_type": "movie",
                    "name": "Bad Script",
                    "source_type": "screenplay",
                    "story_text": "no scene heading here",
                    "desired_length": 30,
                },
            )

            self.assertEqual(400, response.status_code)
            self.assertIn("screenplay", response.text.lower())

    def test_api_starts_movie_full_auto_job_and_writes_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(temp_dir))
            created = client.post(
                "/api/projects",
                json={
                    "project_type": "movie",
                    "name": "Door Below",
                    "source_type": "short_story",
                    "story_text": "A locksmith finds a glowing door below an abandoned station.",
                    "desired_length": 30,
                    "movie_mode": "full_auto",
                },
            )
            self.assertEqual(200, created.status_code, created.text)

            job = client.post("/api/projects/door-below/jobs", json={"action": "movie-full-auto"})

            self.assertEqual(200, job.status_code, job.text)
            job_id = job.json()["id"]
            for _ in range(50):
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] == "succeeded":
                    break
            self.assertEqual("succeeded", status["status"])
            self.assertTrue((Path(temp_dir) / "door-below" / "output" / "movie" / "door-below.mp4").exists())
            patched_workflow_path = Path(temp_dir) / "door-below" / "movie" / "workflows" / "video_ltxv_msr_movie_native_audio.json"
            self.assertTrue(patched_workflow_path.exists())
            patched_workflow = json.loads(patched_workflow_path.read_text())
            self.assertFalse(any(node.get("class_type") in {"LoadAudio", "TrimAudioDuration"} for node in patched_workflow.values()))
            self.assertFalse(any("audio" in key.lower() for node in patched_workflow.values() for key in node.get("inputs", {})))

    def test_movie_full_auto_rejects_non_movie_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(temp_dir))
            created = client.post(
                "/api/projects",
                json={"project_type": "standard_music_video", "name": "Song"},
            )
            self.assertEqual(200, created.status_code, created.text)

            job = client.post("/api/projects/song/jobs", json={"action": "movie-full-auto"})

            self.assertEqual(400, job.status_code)
            self.assertIn("movie", job.text.lower())


if __name__ == "__main__":
    unittest.main()
