import json
import time
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

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


class FakeMovieImageBackend:
    def __init__(self):
        self.requests = []

    def render_image(self, request):
        self.requests.append(request)
        output = Path(request.output_dir) / f"scene_{int(request.scene_number):04}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "white").save(output)
        return output

    def concat_clips(self, concat_list, output_file, video_only=False):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"movie")
        self.concat_calls.append((Path(concat_list), output_file, video_only))
        return output_file


class FakeMovieReferenceGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, *, project_dir):
        project_dir = Path(project_dir)
        self.calls.append(project_dir)
        manifest_path = project_dir / "movie" / "references" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["actors"][0]["msr_sheet_path"] = "movie/references/actors/main_character/msr_sheet.png"
        manifest["locations"][0]["msr_sheet_path"] = "movie/references/locations/primary_location/views/hero.png"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path


class ManifestCheckingVisualBackend:
    def __init__(self):
        self.calls = []

    def render_movie(self, *, project_dir, render_plan_path):
        project_dir = Path(project_dir)
        self.calls.append((project_dir, Path(render_plan_path)))
        manifest = json.loads((project_dir / "movie" / "references" / "manifest.json").read_text())
        if not manifest["actors"][0]["msr_sheet_path"] or not manifest["locations"][0]["msr_sheet_path"]:
            raise AssertionError("movie references must be generated before rendering")
        output = project_dir / "output" / "movie" / f"{project_dir.name}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"movie")
        return output


class FakeMoviePlanner:
    def __init__(self):
        self.story_calls = []
        self.shot_calls = []

    def generate_story_arch(self, *, title, source_type, story_text, desired_length):
        from feverslop.domain.movie import StoryArch

        self.story_calls.append((title, source_type, story_text, desired_length))
        return StoryArch(title=title, premise="LLM premise", beats=("LLM beat",))

    def plan_shots(self, *, story_arch, desired_length, width, height, min_duration=4.0, max_duration=20.0):
        from feverslop.domain.movie import CinematicShot

        self.shot_calls.append((story_arch, desired_length, width, height))
        return (
            CinematicShot(
                shot_id="shot_0001",
                description="LLM shot",
                duration_seconds=float(desired_length),
                camera="LLM camera",
                action="LLM action",
                expression="LLM expression",
                location="LLM location",
                dialogue="MARA: LLM line",
                actor_ids=("mara",),
                location_id="llm_location",
            ),
        )


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

    def test_movie_manifest_contains_all_render_plan_reference_ids(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.domain.movie import CinematicShot, StoryArch

        class Planner:
            def generate_story_arch(self, **_kwargs):
                return StoryArch(title="Void", premise="At least 3 characters cross a void.", beats=("beat",))

            def plan_shots(self, **_kwargs):
                return (
                    CinematicShot(
                        shot_id="shot_0001",
                        description="A man enters the void",
                        duration_seconds=4,
                        camera="wide",
                        action="walks",
                        expression="afraid",
                        location="The Desolate Void",
                        actor_ids=("tormented_man",),
                        location_id="desolate_void",
                    ),
                    CinematicShot(
                        shot_id="shot_0002",
                        description="A beautiful woman emerges from dark mist with predatory grace",
                        duration_seconds=4,
                        camera="close",
                        action="The succubus glides toward him, arms outstretched",
                        expression="seductive and hungry",
                        location="The Desolate Void",
                        dialogue="Come, rest your weary soul...",
                        actor_ids=("seductive_succubus",),
                        location_id="desolate_void",
                    ),
                    CinematicShot(
                        shot_id="shot_0003",
                        description="A demonic goat blocks the path with twisted horns",
                        duration_seconds=4,
                        camera="low angle",
                        action="The goat stamps in the fog",
                        expression="defiant",
                        location="The Desolate Void",
                        actor_ids=("demonic_goat",),
                        location_id="desolate_void",
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=Planner(), projects_root=Path(temp_dir)).execute(
                MovieInput(
                    name="Void",
                    source_type="short_story",
                    story_text="A man fights a succubus and demonic goat inside a desolate void.",
                    desired_length=8,
                )
            )
            root = Path(temp_dir) / "void"
            manifest = json.loads((root / "movie" / "references" / "manifest.json").read_text())
            plan = json.loads((root / "movie" / "render_plan.json").read_text())

            self.assertEqual("desolate_void", plan["shots"][0]["reference_ids"]["location"])
            self.assertIn("desolate_void", {location["id"] for location in manifest["locations"]})
            self.assertIn("demonic_goat", {actor["id"] for actor in manifest["actors"]})
            self.assertNotIn("come_rest_your_weary_soul", {actor["id"] for actor in manifest["actors"]})
            succubus = next(actor for actor in manifest["actors"] if actor["id"] == "seductive_succubus")
            self.assertIn("beautiful woman", succubus["prompt"])
            self.assertIn("predatory grace", succubus["prompt"])
            self.assertNotIn("drawn from the story premise", succubus["prompt"])

    def test_auto_produce_movie_generates_references_before_rendering(self):
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner
        from feverslop.application.movie import AutoProduceMovieUseCase, MovieInput, ScaffoldMovieUseCase

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_generator = FakeMovieReferenceGenerator()
            visual_backend = ManifestCheckingVisualBackend()

            result = AutoProduceMovieUseCase(
                scaffold=ScaffoldMovieUseCase(
                    planner=DeterministicMoviePlanner(),
                    projects_root=Path(temp_dir),
                ),
                reference_generator=reference_generator,
                visual_backend=visual_backend,
            ).execute(
                MovieInput(
                    name="Door Below",
                    source_type="short_story",
                    story_text="A locksmith finds a glowing door below an abandoned station.",
                    desired_length=30,
                    mode="full_auto",
                )
            )

            self.assertEqual([Path(temp_dir) / "door-below"], reference_generator.calls)
            self.assertEqual(Path(temp_dir) / "door-below" / "output" / "movie" / "door-below.mp4", result.final_video_path)

    def test_screenplay_scaffold_preserves_location_action_and_dialogue(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        screenplay = """
        INT. ABANDONED STATION - NIGHT

        MARA
        We go below before sunrise.

        Mara opens a glowing maintenance door and listens to the hum beneath the tracks.

        EXT. ROOFTOP - DAWN

        Mara watches the city wake as the key burns blue in her hand.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
            ).execute(
                MovieInput(
                    name="Door Below",
                    source_type="screenplay",
                    story_text=screenplay,
                    desired_length=24,
                    mode="scaffold",
                )
            )

            render_plan = json.loads(result.render_plan_path.read_text())

            self.assertEqual(2, len(render_plan["shots"]))
            first = render_plan["shots"][0]
            self.assertEqual("ABANDONED STATION - NIGHT", first["location"])
            self.assertIn("glowing maintenance door", first["action"])
            self.assertEqual("MARA: We go below before sunrise.", first["dialogue"])
            self.assertIn("interior", first["camera"].lower())

    def test_screenplay_scaffold_seeds_reference_manifest_from_screenplay_cues(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        screenplay = """
        INT. ABANDONED STATION - NIGHT

        MARA
        We go below before sunrise.

        Mara opens a glowing maintenance door beneath the tracks.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
            ).execute(
                MovieInput(
                    name="Door Below",
                    source_type="screenplay",
                    story_text=screenplay,
                    desired_length=12,
                    mode="scaffold",
                )
            )

            manifest = json.loads(result.reference_manifest_path.read_text())

            self.assertEqual("Mara", manifest["actors"][0]["name"])
            self.assertIn("Mara", manifest["actors"][0]["prompt"])
            self.assertEqual("Abandoned Station - Night", manifest["locations"][0]["name"])
            self.assertIn("Abandoned Station - Night", manifest["locations"][0]["prompt"])

    def test_movie_planner_splits_long_beats_into_varied_shot_durations(self):
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner
        from feverslop.domain.movie import StoryArch

        shots = DeterministicMoviePlanner().plan_shots(
            story_arch=StoryArch(title="Inner Fight", premise="A man fights inner demons.", beats=("A man fights his inner demons and breaks.",)),
            desired_length=120,
            width=1280,
            height=704,
        )

        durations = [shot.duration_seconds for shot in shots]
        self.assertGreater(len(shots), 5)
        self.assertLessEqual(max(durations), 20)
        self.assertGreater(len(set(durations)), 1)

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
            self.assertTrue((temp / "output" / "movie" / "ltx_msr_debug" / "scene_0001_workflow.json").exists())

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

        adapter = build_movie_visual_adapter(Path("project"), Path("workflow.json"), movie_config={"render_backend": "comfyui"})

        self.assertIsInstance(adapter, ComfyUIMovieVisualAdapter)

    def test_movie_visual_adapter_defaults_to_comfyui_not_placeholder(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
        from feverslop.studio.job_service import build_movie_visual_adapter

        adapter = build_movie_visual_adapter(Path("project"), Path("workflow.json"))

        self.assertIsInstance(adapter, ComfyUIMovieVisualAdapter)

    def test_movie_local_visual_adapter_requires_explicit_dev_override(self):
        from feverslop.adapters.movie_visual import LocalMovieVisualAdapter
        from feverslop.studio.job_service import build_movie_visual_adapter

        adapter = build_movie_visual_adapter(Path("project"), Path("workflow.json"), movie_config={"render_backend": "local"})

        self.assertIsInstance(adapter, LocalMovieVisualAdapter)

    def test_movie_reference_generator_defaults_to_comfyui_not_placeholder(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
        from feverslop.application.movie_references import MovieReferenceSheetGenerator
        from feverslop.studio.job_service import build_movie_reference_generator

        generator = build_movie_reference_generator()

        self.assertIsInstance(generator, MovieReferenceSheetGenerator)
        self.assertIsInstance(generator.backend, ComfyUIImageBackend)

    def test_movie_local_reference_generator_requires_explicit_dev_override(self):
        from feverslop.adapters.movie_references import LocalMovieImageBackend
        from feverslop.studio.job_service import build_movie_reference_generator

        generator = build_movie_reference_generator(movie_config={"reference_backend": "local"})

        self.assertIsInstance(generator.backend, LocalMovieImageBackend)

    def test_movie_workflow_patcher_uses_default_movie_msr_template(self):
        from feverslop.studio.job_service import patch_movie_msr_workflow

        output = patch_movie_msr_workflow()

        self.assertIsInstance(output, dict)
        self.assertFalse(any(node.get("class_type") in {"LoadAudio", "TrimAudioDuration"} for node in output.values()))

    def test_llm_movie_planner_sends_story_and_shot_requests_to_llm(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, system_prompt=None):
                self.calls.append((system_prompt, prompt))
                if "shot plan" in prompt.lower():
                    return json.dumps({
                        "shots": [
                            {
                                "description": "Mara opens the door",
                                "camera": "slow push-in",
                                "action": "opens a glowing maintenance door",
                                "expression": "wary focus",
                                "location": "abandoned station",
                                "dialogue": "MARA: We go below.",
                            }
                        ]
                    })
                return json.dumps({"premise": "A locksmith descends below the city.", "beats": ["Mara finds the door."]})

        llm = FakeLLM()
        planner = LLMMoviePlanner(llm)

        story_arch = planner.generate_story_arch(
            title="Door Below",
            source_type="short_story",
            story_text="A locksmith finds a glowing door below an abandoned station.",
            desired_length=12,
        )
        shots = planner.plan_shots(story_arch=story_arch, desired_length=12, width=1280, height=704)

        self.assertEqual("A locksmith descends below the city.", story_arch.premise)
        self.assertEqual(("Mara finds the door.",), story_arch.beats)
        self.assertEqual("slow push-in", shots[0].camera)
        self.assertEqual("MARA: We go below.", shots[0].dialogue)
        self.assertEqual(2, len(llm.calls))
        self.assertIn("story arch", llm.calls[0][1].lower())
        self.assertIn("shot plan", llm.calls[1][1].lower())

    def test_movie_planner_preserves_requested_minimum_actor_count(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner

        class OneActorLLM:
            def complete_prompt(self, prompt, system_prompt=None):
                if "shot plan" in prompt.lower():
                    return json.dumps({
                        "shots": [
                            {
                                "description": "Mara enters the archive",
                                "duration_seconds": 4,
                                "actor_ids": ["mara"],
                                "location_id": "archive",
                            }
                        ]
                    })
                return json.dumps({
                    "premise": "A story with at least 3 characters entering a haunted archive.",
                    "beats": ["Mara, Theo, and Lin cross the threshold."],
                })

        planner = LLMMoviePlanner(OneActorLLM())
        story_arch = planner.generate_story_arch(title="Archive", source_type="short_story", story_text="at least 3 characters", desired_length=12)
        shots = planner.plan_shots(story_arch=story_arch, desired_length=12, width=1280, height=704)
        actor_ids = {actor_id for shot in shots for actor_id in shot.actor_ids}

        self.assertGreaterEqual(len(actor_ids), 3)

    def test_movie_reference_generator_fills_manifest_paths(self):
        from feverslop.application.movie_references import MovieReferenceSheetGenerator

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            manifest_path = temp / "movie" / "references" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps({
                    "actors": [{"id": "main_character", "name": "Mara", "prompt": "gothic protagonist", "msr_sheet_path": ""}],
                    "locations": [{"id": "primary_location", "name": "Station", "prompt": "abandoned station", "msr_sheet_path": ""}],
                }),
                encoding="utf-8",
            )
            backend = FakeMovieImageBackend()

            updated = MovieReferenceSheetGenerator(
                backend=backend,
                edit_backend=backend,
            ).generate(project_dir=temp)

            manifest = json.loads(updated.read_text())
            self.assertEqual("movie/references/actors/main_character/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertEqual("movie/references/locations/primary_location/views/hero.png", manifest["locations"][0]["msr_sheet_path"])
            self.assertGreaterEqual(len(backend.requests), 2)

    def test_movie_reference_sync_adds_missing_render_plan_ids(self):
        from feverslop.studio.job_service import sync_movie_manifest_with_render_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "movie" / "references").mkdir(parents=True)
            (root / "movie" / "render_plan.json").write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "description": "A man enters the void",
                        "action": "The tormented man walks through fog",
                        "location": "The Desolate Void",
                        "reference_ids": {"actors": ["tormented_man", "demonic_goat"], "location": "desolate_void"},
                    }
                ]),
                encoding="utf-8",
            )
            (root / "movie" / "references" / "manifest.json").write_text(
                json.dumps({
                    "actors": [
                        {
                            "id": "tormented_man",
                            "name": "Tormented Man",
                            "prompt": "consistent cinematic character Tormented Man for tm3, drawn from the story premise",
                            "msr_sheet_path": "x.png",
                        }
                    ],
                    "locations": [{"id": "the_desolate_void", "name": "The Desolate Void", "prompt": "", "msr_sheet_path": "y.png"}],
                }),
                encoding="utf-8",
            )

            sync_movie_manifest_with_render_plan(root)

            manifest = json.loads((root / "movie" / "references" / "manifest.json").read_text())
            self.assertIn("demonic_goat", {actor["id"] for actor in manifest["actors"]})
            self.assertIn("desolate_void", {location["id"] for location in manifest["locations"]})
            missing_actor = next(actor for actor in manifest["actors"] if actor["id"] == "demonic_goat")
            self.assertEqual("", missing_actor["msr_sheet_path"])
            repaired_actor = next(actor for actor in manifest["actors"] if actor["id"] == "tormented_man")
            self.assertIn("A man enters the void", repaired_actor["prompt"])
            self.assertEqual("", repaired_actor["msr_sheet_path"])

    def test_api_creates_movie_project_with_scaffold_artifacts(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_planner = FakeMoviePlanner()
            with patch("feverslop.studio.project_repository.build_movie_planner", return_value=fake_planner):
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
            self.assertEqual("present", project["status"]["config"])
            self.assertEqual("present", project["status"]["render_plan"])
            self.assertIn("movie/story_arch.json", project["artifacts"]["generated_json"])
            self.assertIn("movie/render_plan.json", project["artifacts"]["render_plans"])
            self.assertIn("movie/references/manifest.json", project["artifacts"]["references"])
            config = json.loads((Path(temp_dir) / "door-below" / "config.json").read_text())
            self.assertEqual("Door Below", config["project_name"])
            self.assertEqual("A locksmith finds a glowing door below an abandoned station.", config["story_idea"])
            self.assertEqual({"fps": 24, "width": 1280, "height": 704}, config["video"])
            self.assertIn("steering", config)
            self.assertEqual("llm", project["metadata"]["movie"]["planner_backend"])
            self.assertEqual("comfyui", project["metadata"]["movie"]["reference_backend"])
            self.assertEqual("comfyui", project["metadata"]["movie"]["render_backend"])
            self.assertEqual(1, len(fake_planner.story_calls))
            self.assertEqual(1, len(fake_planner.shot_calls))

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
                    "movie_planner_backend": "deterministic",
                    "movie_reference_backend": "local",
                    "movie_render_backend": "local",
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
                time.sleep(0.01)
            self.assertEqual("succeeded", status["status"])
            self.assertTrue((Path(temp_dir) / "door-below" / "output" / "movie" / "door-below.mp4").exists())
            manifest = json.loads((Path(temp_dir) / "door-below" / "movie" / "references" / "manifest.json").read_text())
            self.assertEqual("movie/references/actors/main_character/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertEqual("movie/references/locations/primary_location/views/hero.png", manifest["locations"][0]["msr_sheet_path"])
            self.assertEqual("local", manifest["generator_backend"])
            self.assertFalse((Path(temp_dir) / "door-below" / "movie" / "workflows").exists())

    def test_api_starts_movie_reference_job_and_updates_manifest(self):
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
                    "movie_mode": "scaffold",
                    "movie_planner_backend": "deterministic",
                    "movie_reference_backend": "local",
                },
            )
            self.assertEqual(200, created.status_code, created.text)

            job = client.post("/api/projects/door-below/jobs", json={"action": "movie-references"})

            self.assertEqual(200, job.status_code, job.text)
            job_id = job.json()["id"]
            for _ in range(50):
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] == "succeeded":
                    break
                time.sleep(0.01)
            self.assertEqual("succeeded", status["status"])
            manifest = json.loads((Path(temp_dir) / "door-below" / "movie" / "references" / "manifest.json").read_text())
            self.assertEqual("movie/references/actors/main_character/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertEqual("movie/references/locations/primary_location/views/hero.png", manifest["locations"][0]["msr_sheet_path"])

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
