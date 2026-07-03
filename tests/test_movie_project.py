import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from feverslop.studio.server import create_app


class FakeMovieRenderQueue:
    def __init__(self):
        self.calls = []

    def queue_workflow_and_download_first_video(self, workflow, *, scene_number, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"scene {scene_number}".encode())
        self.calls.append((workflow, scene_number, output_path))
        return output_path


class NativeAudioAssetUploader:
    def __init__(self):
        self.audio_calls = []
        self.reference_calls = []

    def resolve_audio_name(self, audio_file, *, upload_audio, uploaded_audio_name):
        self.audio_calls.append((Path(audio_file), upload_audio, uploaded_audio_name))
        raise AssertionError("movie MSR rendering must not resolve or upload custom audio")

    def resolve_reference_image_name(self, image_path, *, upload_references=True):
        self.reference_calls.append((Path(image_path), upload_references))
        return Path(image_path).name


class FakeMoviePostprocessor:
    def __init__(self):
        self.trim_specs = []
        self.concat_lists = []
        self.concat_calls = []

    def trim_clip(self, spec):
        self.trim_specs.append(spec)
        spec.output_file.parent.mkdir(parents=True, exist_ok=True)
        spec.output_file.write_bytes(b"trimmed")
        return spec.output_file

    def write_concat_list(self, video_files, output_file):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(str(path) for path in video_files), encoding="utf-8")
        self.concat_lists.append((list(video_files), output_file))
        return output_file

    def concat_clips(self, concat_list, output_file, video_only=False):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"movie")
        self.concat_calls.append((Path(concat_list), output_file, video_only))
        return output_file


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
            self.assertTrue((Path(temp_dir) / "door-below" / "movie" / "references" / "manifest.json").exists())
            render_plan = json.loads((Path(temp_dir) / "door-below" / "movie" / "render_plan.json").read_text())
            self.assertGreaterEqual(len(render_plan["shots"]), 1)
            self.assertEqual({"width": 1280, "height": 704}, render_plan["resolution"])
            self.assertEqual(["main_character"], render_plan["shots"][0]["reference_ids"]["actors"])
            self.assertEqual("primary_location", render_plan["shots"][0]["reference_ids"]["location"])
            manifest = json.loads((Path(temp_dir) / "door-below" / "movie" / "references" / "manifest.json").read_text())
            self.assertEqual("main_character", manifest["actors"][0]["id"])
            self.assertEqual("primary_location", manifest["locations"][0]["id"])

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

    def test_msr_backend_renders_without_custom_audio_when_upload_disabled(self):
        from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
        from feverslop.ports.rendering import VideoRenderRequest

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "movie_msr.json"
            workflow_path.write_text(json.dumps(_movie_msr_workflow()), encoding="utf-8")
            queue = FakeMovieRenderQueue()
            asset_uploader = NativeAudioAssetUploader()

            backend = ComfyUIMSRVideoRenderBackend(
                client=object(),
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                asset_uploader=asset_uploader,
                render_queue=queue,
                postprocess=False,
            )

            output = backend.render_video(
                VideoRenderRequest(
                    scene=_movie_scene(temp),
                    scene_number=1,
                    prompt="A tense lock opens below the city.",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    audio_file=temp / "missing.mp3",
                    storyboard_dir=temp / "storyboard",
                    upload_audio=False,
                )
            )

            self.assertEqual(temp / "out" / "raw" / "scene_0001_raw.mp4", output)
            self.assertEqual([], asset_uploader.audio_calls)
            queued_workflow = queue.calls[0][0]
            self.assertFalse(any("audio" in key.lower() for node in queued_workflow.values() for key in node.get("inputs", {})))

    def test_comfyui_movie_adapter_renders_shots_with_ltx_native_audio(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "movie_msr.json"
            workflow_path.write_text(json.dumps(_movie_msr_workflow()), encoding="utf-8")
            render_plan_path = temp / "movie" / "render_plan.json"
            render_plan_path.parent.mkdir()
            render_plan_path.write_text(
                json.dumps({
                    "title": "Door Below",
                    "resolution": {"width": 1280, "height": 704},
                    "duration_seconds": 2,
                    "shots": [_movie_shot(temp)],
                }),
                encoding="utf-8",
            )
            queue = FakeMovieRenderQueue()
            asset_uploader = NativeAudioAssetUploader()
            postprocessor = FakeMoviePostprocessor()

            final = ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                render_queue=queue,
                asset_uploader=asset_uploader,
                postprocessor=postprocessor,
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path)

            self.assertEqual(temp / "output" / "movie" / "door-below.mp4", final)
            self.assertEqual([], asset_uploader.audio_calls)
            self.assertEqual(1, len(queue.calls))
            self.assertEqual([(postprocessor.concat_lists[0][1], final, False)], postprocessor.concat_calls)

    def test_comfyui_movie_adapter_resolves_reference_ids_from_manifest(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "movie_msr.json"
            workflow_path.write_text(json.dumps(_movie_msr_workflow()), encoding="utf-8")
            (temp / "movie" / "references").mkdir(parents=True)
            (temp / "movie" / "references" / "actor.png").write_bytes(b"actor")
            (temp / "movie" / "references" / "location.png").write_bytes(b"location")
            (temp / "movie" / "references" / "manifest.json").write_text(
                json.dumps({
                    "actors": [{"id": "main_character", "msr_sheet_path": "movie/references/actor.png"}],
                    "locations": [{"id": "primary_location", "msr_sheet_path": "movie/references/location.png"}],
                }),
                encoding="utf-8",
            )
            render_plan_path = temp / "movie" / "render_plan.json"
            render_plan_path.write_text(
                json.dumps({
                    "title": "Door Below",
                    "resolution": {"width": 1280, "height": 704},
                    "shots": [
                        {
                            "shot_id": "shot_0001",
                            "description": "A tense lock opens below the city.",
                            "duration_seconds": 2,
                            "reference_ids": {"actors": ["main_character"], "location": "primary_location"},
                        }
                    ],
                }),
                encoding="utf-8",
            )
            queue = FakeMovieRenderQueue()
            asset_uploader = NativeAudioAssetUploader()

            ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                render_queue=queue,
                asset_uploader=asset_uploader,
                postprocessor=FakeMoviePostprocessor(),
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path)

            self.assertEqual(
                [(temp / "movie" / "references" / "actor.png", True), (temp / "movie" / "references" / "location.png", True)],
                asset_uploader.reference_calls,
            )

    def test_movie_job_uses_comfyui_adapter_when_configured(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
        from feverslop.studio.job_service import build_movie_visual_adapter

        previous = os.environ.get("FEVERSLOP_MOVIE_RENDER_BACKEND")
        os.environ["FEVERSLOP_MOVIE_RENDER_BACKEND"] = "comfyui"
        try:
            adapter = build_movie_visual_adapter(Path("project"), Path("workflow.json"))
        finally:
            if previous is None:
                os.environ.pop("FEVERSLOP_MOVIE_RENDER_BACKEND", None)
            else:
                os.environ["FEVERSLOP_MOVIE_RENDER_BACKEND"] = previous

        self.assertIsInstance(adapter, ComfyUIMovieVisualAdapter)

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
            self.assertIn("movie/references/manifest.json", project["artifacts"]["references"])

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


def _movie_msr_workflow() -> dict:
    return {
        "1": {"class_type": "LoadImage", "_meta": {"title": "#MSR_ACTOR_1"}, "inputs": {"image": ""}},
        "2": {"class_type": "LoadImage", "_meta": {"title": "#MSR_BACKGROUND"}, "inputs": {"image": ""}},
        "3": {"class_type": "LiconMSR", "_meta": {"title": "#MSR_FRAME_COUNT"}, "inputs": {"1": ["1", 0], "frame_count": 17}},
        "4": {"class_type": "Text", "_meta": {"title": "#PROMPT"}, "inputs": {"text": ""}},
        "5": {"class_type": "Integer", "_meta": {"title": "#WIDTH"}, "inputs": {"value": 0}},
        "6": {"class_type": "Integer", "_meta": {"title": "#HEIGHT"}, "inputs": {"value": 0}},
        "7": {"class_type": "Integer", "_meta": {"title": "#FRAMES"}, "inputs": {"value": 0}},
        "8": {"class_type": "Integer", "_meta": {"title": "#FRAMERATE"}, "inputs": {"value": 0}},
        "9": {"class_type": "KSampler", "_meta": {"title": "#SEED"}, "inputs": {"seed": 0}},
        "10": {"class_type": "SaveVideo", "_meta": {"title": "#SAVE_VIDEO"}, "inputs": {"filename_prefix": ""}},
    }


def _movie_scene(project_dir: Path) -> dict:
    actor = project_dir / "refs" / "actor.png"
    location = project_dir / "refs" / "location.png"
    return {
        "scene": 1,
        "abs_start_seconds": 0,
        "duration_seconds": 2,
        "frame_count": 48,
        "fps": 24,
        "width": 1280,
        "height": 704,
        "ltx": {"original_style_i2v_prompt": "A tense lock opens below the city."},
        "references": {
            "actor_msr_paths": [actor.as_posix()],
            "location_msr_path": location.as_posix(),
        },
    }


def _movie_shot(project_dir: Path) -> dict:
    scene = _movie_scene(project_dir)
    return {
        "shot_id": "shot_0001",
        "description": scene["ltx"]["original_style_i2v_prompt"],
        "duration_seconds": 2,
        "camera": "slow dolly",
        "action": "the lock opens",
        "expression": "wary focus",
        "location": "abandoned station",
        "references": scene["references"],
        "frame_count": 48,
        "fps": 24,
    }


if __name__ == "__main__":
    unittest.main()
