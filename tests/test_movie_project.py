import json
import time
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from feverslop.adapters.movie_artifact_writer import LocalMovieArtifactWriter
from tests.studio_harness import NativeStudioHarness


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
        self.last_frame_extracts = []

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

    def concat_clips(self, concat_list, output_file, video_only=False, reencode=False):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"movie")
        self.concat_calls.append((Path(concat_list), output_file, video_only, reencode))
        return output_file

    def extract_last_frame(self, source_file, output_file):
        source_file = Path(source_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"png")
        self.last_frame_extracts.append((source_file, output_file))
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

    def generate_movie_bible(self, *, title, source_type, story_text, desired_length, story_arch, config):
        from feverslop.domain.movie import MovieActor, MovieBible, MovieLocation

        return MovieBible(
            title=title,
            premise="LLM premise",
            story_arch=story_arch,
            actors=(MovieActor(id="mara", name="Mara", role="lead", visual_description="LLM actor"),),
            locations=(MovieLocation(id="llm_location", name="LLM Location", visual_description="LLM location"),),
            continuity=(),
            style_constraints=(),
            runtime_constraints={"desired_length": desired_length},
        )

    def plan_shots_from_bible(self, *, bible, screenplay=None, desired_length, width, height, min_duration=4.0, max_duration=20.0):
        from feverslop.domain.movie import CinematicShot

        self.shot_calls.append((bible, desired_length, width, height))
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

    def generate_movie_continuity_plan(self, *, title, source_type, story_text, desired_length, bible, shots, config):
        return {}

    def generate_movie_story_design(self, *, title, source_type, story_text, desired_length, bible, story_arch, config):
        return {}

    def generate_movie_screenplay(self, *, title, source_type, story_text, desired_length, bible, story_arch, story_design, config):
        return {}

    def generate_movie_narrative_plan(self, *, title, source_type, desired_length, bible, screenplay, config):
        return {}


class MovieProjectTests(unittest.TestCase):
    def test_movie_ingredients_prompt_prefers_compact_static_prompt(self):
        from feverslop.adapters.movie_ingredients_visual import _ingredients_prompt

        scene = {
            "description": "fallback",
            "ingredients": {"global_prompt": "global"},
            "ltx": {"static_prompt": "static relay summary"},
        }

        self.assertEqual("static relay summary", _ingredients_prompt(scene))

    def test_prepared_ingredients_movie_accepts_compact_prompt_contract(self):
        from unittest.mock import Mock

        from feverslop.application.movie_prepared_workflows import prepare_movie_workflows

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            plan = project / "ingredients.json"
            plan.write_text("{}", encoding="utf-8")
            materializer = Mock()
            scene = {
                "scene": 1,
                "ingredients": {
                    "sheet_path": "sheet.png",
                    "anchors": [{"id": "mara"}],
                    "global_prompt": "Character `mara`: stable silver coat and black bob.",
                },
                "ltx": {
                    "static_prompt": "Mara remains stable and begins speaking immediately.",
                    "prompt_relay": [
                        {"frame_start": 0, "frame_end": 48, "state": "dialogue", "prompt": "Mara speaks."},
                    ],
                },
            }

            prepare_movie_workflows(
                project_dir=project,
                render_plan_path=plan,
                pipeline="ltx_ingredients",
                scenes=[scene],
                selected_scenes=[1],
                materializer=materializer,
                prompt_for_scene=lambda item: item["ltx"]["static_prompt"],
            )

            self.assertEqual(scene, materializer.prepare.call_args.args[0].scene)

    def test_prepared_ingredients_movie_validates_compact_anchor_bindings(self):
        from unittest.mock import Mock

        from feverslop.application.movie_prepared_workflows import prepare_movie_workflows

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            plan = project / "ingredients.json"
            plan.write_text("{}", encoding="utf-8")
            scene = {
                "scene": 1,
                "ingredients": {
                    "sheet_path": "sheet.png",
                    "anchors": [{"id": "mara"}],
                    "global_prompt": "A generic room without a bound actor identifier.",
                },
                "ltx": {"prompt_relay": []},
            }

            with self.assertRaisesRegex(ValueError, "global prompt does not bind anchors mara"):
                prepare_movie_workflows(
                    project_dir=project,
                    render_plan_path=plan,
                    pipeline="ltx_ingredients",
                    scenes=[scene],
                    selected_scenes=[1],
                    materializer=Mock(),
                    prompt_for_scene=lambda _scene: "prompt",
                )

    def test_prepared_movie_workflows_use_identical_strict_scene_filter(self):
        from unittest.mock import Mock

        from feverslop.application.movie_prepared_workflows import prepare_movie_workflows
        from feverslop.scene_artifacts import SceneArtifactLayout

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            plan = project / "plan.json"
            plan.write_text("{}", encoding="utf-8")
            materializer = Mock()
            scenes = [
                {"scene": 1, "description": "one"},
                {"scene": 2, "description": "two"},
                {"scene": 3, "description": "three"},
            ]

            prepared = prepare_movie_workflows(
                project_dir=project,
                render_plan_path=plan,
                pipeline="ltx_msr",
                scenes=scenes,
                selected_scenes=[1, 3],
                materializer=materializer,
                prompt_for_scene=lambda scene: scene["description"],
            )

            self.assertEqual([1, 3], [call.args[0].scene["scene"] for call in materializer.prepare.call_args_list])
            self.assertEqual(SceneArtifactLayout(project).scenes_dir, prepared)

    def test_prepared_movie_render_requires_selected_manifest_and_names_prepare(self):
        from unittest.mock import Mock

        from feverslop.application.movie_prepared_workflows import render_prepared_movie_workflows

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)

            with self.assertRaisesRegex(FileNotFoundError, "prepare|--write-debug-workflows"):
                render_prepared_movie_workflows(
                    project_dir=project,
                    scenes=[{"scene": 1}, {"scene": 2}],
                    selected_scenes=[2],
                    renderer=Mock(),
                    postprocessor=Mock(),
                )

    def test_prepared_movie_render_writes_canonical_final_and_reads_legacy_fallback(self):
        from unittest.mock import Mock

        from feverslop.application.movie_prepared_workflows import render_prepared_movie_workflows
        from feverslop.scene_artifacts import SceneArtifactLayout

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.scene_manifest(2).parent.mkdir(parents=True)
            layout.scene_manifest(2).write_text("{}", encoding="utf-8")
            layout.scene_workflow(2).write_text("{}", encoding="utf-8")
            legacy = project / "output" / "movie" / "ltx_msr"
            legacy.mkdir(parents=True)
            (legacy / "scene_0001.mp4").write_bytes(b"legacy")
            renderer = Mock()
            renderer.render.return_value = layout.scene_final_video(2)
            layout.scene_final_video(2).write_bytes(b"new")
            postprocessor = FakeMoviePostprocessor()

            result = render_prepared_movie_workflows(
                project_dir=project,
                scenes=[{"scene": 1}, {"scene": 2}],
                selected_scenes=[2],
                renderer=renderer,
                postprocessor=postprocessor,
                legacy_dirs=[legacy],
            )

            self.assertEqual(layout.movie, result)
            self.assertEqual(
                [legacy / "scene_0001.mp4", layout.scene_final_video(2)],
                postprocessor.concat_lists[0][0],
            )

    def test_movie_scaffold_persists_screenplay_memory_and_shot_cards(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=DeterministicMoviePlanner(), projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
                MovieInput(
                    name="Archive Memory",
                    source_type="screenplay",
                    story_text=(
                        "INT. ARCHIVE - NIGHT\n"
                        "Mara opens the sealed ledger.\n"
                        "MARA\n"
                        "It remembers me.\n"
                    ),
                    desired_length=12,
                    config={
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "archivist in charcoal"}],
                        "structured_locations": [{"id": "archive", "name": "Archive", "visual_description": "white marble archive"}],
                        "dialogue_language": "English",
                    },
                )
            )

            root = Path(temp_dir) / "archive-memory"
            story_design = json.loads((root / "movie" / "story_design.json").read_text(encoding="utf-8"))
            screenplay = json.loads((root / "movie" / "screenplay.json").read_text(encoding="utf-8"))
            narrative = json.loads((root / "movie" / "narrative_plan.json").read_text(encoding="utf-8"))
            scene_cards = json.loads((root / "movie" / "scene_cards.json").read_text(encoding="utf-8"))
            shot_cards = json.loads((root / "movie" / "shot_cards.json").read_text(encoding="utf-8"))
            plan = json.loads((root / "movie" / "render_plan.json").read_text(encoding="utf-8"))

            self.assertEqual("movie/story_design.json", plan["movie_story_design_path"])
            self.assertEqual("movie/screenplay.json", plan["movie_screenplay_path"])
            self.assertEqual("movie/shot_cards.json", plan["movie_shot_cards_path"])
            self.assertIn("act_structure", story_design)
            self.assertIn("turning_points", story_design)
            self.assertIn("setup_payoff_threads", story_design)
            self.assertIn("character_arcs", story_design)
            self.assertIn("scene_blueprint", story_design)
            self.assertEqual("scene_0001", story_design["scene_blueprint"][0]["scene_id"])
            self.assertEqual("English", screenplay["dialogue_language"])
            self.assertEqual(["mara"], screenplay["scenes"][0]["actor_ids"])
            self.assertEqual("archive", screenplay["scenes"][0]["location_id"])
            self.assertIn("It remembers me.", screenplay["scenes"][0]["dialogue"])
            self.assertIn("dramatic_purpose", screenplay["scenes"][0])
            self.assertIn("conflict", screenplay["scenes"][0])
            self.assertIn("emotional_turn", screenplay["scenes"][0])
            self.assertIn("subtext", screenplay["scenes"][0])
            self.assertIn("dialogue_function", screenplay["scenes"][0])
            self.assertEqual("scene_0001", narrative["sequences"][0]["scene_ids"][0])
            self.assertEqual("shot_0001", scene_cards["scene_cards"][0]["shot_ids"][0])
            self.assertEqual("shot_0001", shot_cards["shot_cards"][0]["shot_id"])
            self.assertIn("start_frame_brief", shot_cards["shot_cards"][0])
            self.assertIn("end_frame_brief", shot_cards["shot_cards"][0])
            self.assertNotIn("INT. ARCHIVE", shot_cards["memory_pack"]["current_shot"]["description"])

    def test_short_story_scaffold_writes_story_design_and_authored_screenplay_fields(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(planner=DeterministicMoviePlanner(), projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
                MovieInput(
                    name="Compass Radio",
                    source_type="short_story",
                    story_text="A deserter follows a broken compass through a silent battlefield and finds a buried radio.",
                    desired_length=36,
                    config={
                        "dialogue_language": "German",
                        "actors": [{"id": "deserter", "name": "Deserter", "visual_description": "mud covered deserter"}],
                        "structured_locations": [{"id": "battlefield", "name": "Battlefield", "visual_description": "silent battlefield"}],
                        "max_scene_actors": 1,
                    },
                )
            )

            story_design = json.loads((result.project_dir / "movie" / "story_design.json").read_text(encoding="utf-8"))
            screenplay = json.loads((result.project_dir / "movie" / "screenplay.json").read_text(encoding="utf-8"))

            self.assertEqual("Compass Radio", story_design["title"])
            self.assertTrue(story_design["theme"])
            self.assertGreaterEqual(len(story_design["act_structure"]), 3)
            self.assertTrue(story_design["turning_points"])
            self.assertTrue(story_design["setup_payoff_threads"])
            self.assertTrue(story_design["character_arcs"])
            self.assertTrue(story_design["scene_blueprint"])
            blueprint = story_design["scene_blueprint"][0]
            self.assertEqual("scene_0001", blueprint["scene_id"])
            self.assertIn("purpose", blueprint)
            self.assertIn("conflict", blueprint)
            self.assertIn("subtext", blueprint)
            self.assertIn("dialogue_function", blueprint)
            self.assertLessEqual(len(blueprint["required_actors"]), 1)
            self.assertEqual("battlefield", blueprint["location_id"])
            scene = screenplay["scenes"][0]
            self.assertEqual("German", screenplay["dialogue_language"])
            self.assertEqual(blueprint["purpose"], scene["dramatic_purpose"])
            self.assertEqual(blueprint["conflict"], scene["conflict"])
            self.assertEqual(blueprint["emotional_turn"], scene["emotional_turn"])
            self.assertEqual(blueprint["subtext"], scene["subtext"])
            self.assertEqual(blueprint["dialogue_function"], scene["dialogue_function"])
            self.assertEqual(["deserter"], scene["actor_ids"])
            self.assertEqual("battlefield", scene["location_id"])

    def test_movie_scaffold_writes_bible_and_uses_it_for_manifest_and_render_plan(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieContinuityRule, MovieLocation, StoryArch

        class BiblePlanner:
            def generate_story_arch(self, **_kwargs):
                return StoryArch(title="Archive", premise="Mara unlocks a forbidden archive.", beats=("Mara enters",))

            def generate_movie_bible(self, **kwargs):
                story_arch = kwargs["story_arch"]
                return MovieBible(
                    title=story_arch.title,
                    premise=story_arch.premise,
                    story_arch=story_arch,
                    actors=(MovieActor(id="mara", name="Mara", role="archivist", visual_description="stern archivist in a charcoal coat"),),
                    locations=(MovieLocation(id="archive", name="Archive", visual_description="white marble archive hall"),),
                    continuity=(MovieContinuityRule(id="coat", description="Mara always wears the same charcoal coat"),),
                    style_constraints=("quiet gothic realism",),
                    runtime_constraints={"max_scene_actors": 1},
                )

            def plan_shots_from_bible(self, **kwargs):
                bible = kwargs["bible"]
                return (
                    CinematicShot(
                        shot_id="shot_0001",
                        description="Mara studies a sealed ledger",
                        duration_seconds=kwargs["desired_length"],
                        camera="slow dolly",
                        action="opens the ledger",
                        expression="focused",
                        location=bible.locations[0].name,
                        dialogue="MARA: It remembers me.",
                        actor_ids=("mara",),
                        location_id="archive",
                    ),
                )

            def generate_movie_continuity_plan(self, **_kwargs):
                return {}

            def generate_movie_story_design(self, **_kwargs):
                return {}

            def generate_movie_screenplay(self, **_kwargs):
                return {}

            def generate_movie_narrative_plan(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=BiblePlanner(), projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
                MovieInput(
                    name="Archive",
                    source_type="short_story",
                    story_text="Mara unlocks a forbidden archive and finds a ledger that knows her name.",
                    desired_length=12,
                )
            )
            root = Path(temp_dir) / "archive"
            bible = json.loads((root / "movie" / "bible.json").read_text(encoding="utf-8"))
            continuity = json.loads((root / "movie" / "continuity_plan.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "movie" / "references" / "manifest.json").read_text(encoding="utf-8"))
            plan = json.loads((root / "movie" / "render_plan.json").read_text(encoding="utf-8"))

            self.assertEqual("mara", bible["actors"][0]["id"])
            self.assertIn("continuity_ledger", continuity)
            self.assertIn("scene_continuity", continuity)
            self.assertIn("narrative_chain", continuity)
            self.assertEqual("mara", continuity["continuity_ledger"]["characters"]["mara"]["character_id"])
            self.assertIn("shot_0001", continuity["scene_continuity"])
            self.assertEqual("shot_0001", continuity["narrative_chain"][0]["shot_id"])
            self.assertTrue(plan["shots"][0]["story_state_after"])
            self.assertEqual("stern archivist in a charcoal coat", bible["actors"][0]["visual_description"])
            self.assertEqual("archive", bible["locations"][0]["id"])
            self.assertEqual("Mara always wears the same charcoal coat", bible["continuity"][0]["description"])
            self.assertEqual("stern archivist in a charcoal coat", manifest["actors"][0]["visual_description"])
            self.assertIn("Four vertical panels in one image", manifest["actors"][0]["prompt"])
            self.assertEqual(["mara"], plan["shots"][0]["reference_ids"]["actors"])
            self.assertEqual("archive", plan["shots"][0]["reference_ids"]["location"])
            self.assertEqual(["mara"], plan["shots"][0]["actor_ids"])
            self.assertEqual("archive", plan["shots"][0]["location_id"])

    def test_movie_bible_respects_config_actor_location_constraints(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        config = {
            "actors": [
                {"id": "mara", "name": "Mara", "role": "lead", "visual_description": "black-haired archivist in a grey suit"},
            ],
            "structured_locations": [
                {"id": "archive", "name": "Archive", "visual_description": "white seamless archive set"},
            ],
            "max_scene_actors": 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=DeterministicMoviePlanner(), projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
                MovieInput(
                    name="Configured Movie",
                    source_type="short_story",
                    story_text="Mara and an intruder argue inside the archive about a sealed book.",
                    desired_length=20,
                    config=config,
                )
            )
            root = Path(temp_dir) / "configured-movie"
            bible = json.loads((root / "movie" / "bible.json").read_text(encoding="utf-8"))
            plan = json.loads((root / "movie" / "render_plan.json").read_text(encoding="utf-8"))

            self.assertEqual(["mara"], [actor["id"] for actor in bible["actors"]])
            self.assertEqual(["archive"], [location["id"] for location in bible["locations"]])
            for shot in plan["shots"]:
                self.assertLessEqual(len(shot["reference_ids"]["actors"]), 1)
                self.assertTrue(set(shot["reference_ids"]["actors"]).issubset({"mara"}))
                self.assertEqual("archive", shot["reference_ids"]["location"])

    def test_movie_msr_enrichment_writes_video_prompts_without_reference_sheet_text(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Archive",
                        "premise": "Mara opens a forbidden book.",
                        "story_arch": {"title": "Archive", "premise": "Mara opens a forbidden book.", "beats": ["opening"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room"}],
                        "continuity": [{"id": "coat", "description": "same charcoal coat"}],
                        "style_constraints": ["gothic realism"],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Archive",
                        "resolution": {"width": 1280, "height": 704},
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara opens the ledger",
                                "duration_seconds": 4,
                                "camera": "slow dolly",
                                "acting": "controlled fear",
                                "action": "opens the ledger",
                                "dialogue": "MARA: It remembers me.",
                                "continuity_notes": "same charcoal coat",
                                "transition_from_previous": "continuous",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "continuity_plan.json").write_text(
                json.dumps(
                    {
                        "continuity_ledger": {
                            "style_bible": {"visual_style": "gothic realism", "palette": "", "lighting": "", "camera": "", "negative_constraints": []},
                            "characters": {"mara": {"character_id": "mara", "base_identity": "stern archivist", "wardrobe": "same charcoal coat", "carried_props": [], "physical_state": "", "emotional_state": "controlled fear", "last_location": "archive", "last_action": "opens the ledger"}},
                            "locations": {"archive": {"location_id": "archive", "name": "Archive", "time_of_day": "", "lighting": "", "props": [], "environmental_state": "quiet archive room"}},
                            "scene_order": ["shot_0001"],
                        },
                        "scene_continuity": {
                            "shot_0001": {
                                "shot_id": "shot_0001",
                                "location_id": "archive",
                                "incoming": ["Mara arrives with the sealed ledger"],
                                "required_carryovers": ["same charcoal coat"],
                                "allowed_changes": ["ledger opens"],
                                "outgoing": ["the ledger recognizes Mara"],
                                "characters": {"mara": {"character_id": "mara", "base_identity": "stern archivist", "wardrobe": "same charcoal coat", "carried_props": [], "physical_state": "", "emotional_state": "controlled fear", "last_location": "archive", "last_action": "opens the ledger"}},
                                "location": {"location_id": "archive", "name": "Archive", "time_of_day": "", "lighting": "", "props": [], "environmental_state": "quiet archive room"},
                            }
                        },
                        "narrative_chain": [
                            {
                                "shot_id": "shot_0001",
                                "story_state_before": "Mara has not opened the ledger.",
                                "story_state_after": "The ledger recognizes Mara.",
                                "cause_from_previous": "Opening beat.",
                                "narrative_purpose": "Reveal the supernatural hook.",
                                "conflict_or_tension": "The archive knows her.",
                                "turning_point": "The ledger responds.",
                                "sets_up_next": "Mara must decide whether to keep reading.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist", "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room", "msr_sheet_path": "movie/references/locations/archive/views/hero.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            enriched = json.loads(output.read_text(encoding="utf-8"))
            prompt = enriched["shots"][0]["ltx"]["original_style_i2v_prompt"]
            relay = enriched["shots"][0]["ltx"]["msr_prompt_relay"][0]

            self.assertEqual(movie_dir / "render_plan_msr.json", output)
            self.assertIn("slow dolly", prompt)
            self.assertIn('Mara says: "It remembers me."', prompt)
            self.assertNotIn('"MARA:', prompt)
            self.assertEqual("continuous", enriched["shots"][0]["transition_from_previous"])
            self.assertEqual("same charcoal coat", enriched["shots"][0]["continuity_notes"])
            self.assertNotIn("Continuity:", prompt)
            self.assertNotIn("CONTINUITY CONTRACT", prompt)
            self.assertNotIn("Continuity:", relay["prompt"])
            self.assertNotIn("CONTINUITY CONTRACT", relay["prompt"])
            self.assertNotIn("Actors:", relay["prompt"])
            self.assertNotIn("Location:", relay["prompt"])
            self.assertNotIn("Action:", relay["prompt"])
            self.assertNotIn("Audio contract:", relay["prompt"])
            self.assertNotIn("Dialogue language:", relay["prompt"])
            self.assertNotIn("Style:", relay["prompt"])
            self.assertNotIn("Full-body cinematic character reference sheet", prompt)
            self.assertNotIn("Four vertical panels", prompt)
            self.assertEqual(0, relay["frame_start"])
            self.assertEqual(95, relay["frame_end"])
            self.assertNotIn("start_frame", relay)
            self.assertNotIn("end_frame", relay)

    def test_movie_msr_global_prompt_binds_stable_contract_anchors_once(self):
        from feverslop.application.movie_msr_enrichment import _enrich_shot

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            actor = project / "movie" / "references" / "actor.png"
            location = project / "movie" / "references" / "location.png"
            actor.parent.mkdir(parents=True)
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            shot = {
                "shot_id": "shot_0001",
                "scene": 1,
                "description": "As before after the prior shot, Mara enters.",
                "duration_seconds": 2,
                "reference_ids": {"actors": ["mara"], "location": "archive"},
            }
            manifest = {
                "actors": [{
                    "id": "mara",
                    "name": "Mara",
                    "visual_description": (
                        "As before after the previous shot, Mara has a black bob "
                        "and graphite coat"
                    ),
                    "msr_sheet_path": "movie/references/actor.png",
                }],
                "locations": [{
                    "id": "archive",
                    "name": "Archive",
                    "visual_description": (
                        "Prior scene aside, the archive has green lamps and "
                        "brass shelves"
                    ),
                    "msr_sheet_path": "movie/references/location.png",
                }],
            }

            enriched = _enrich_shot(
                shot,
                bible={"runtime_constraints": {"fps": 24}},
                manifest=manifest,
                fps=24,
                project_dir=project,
                workflow_profile="msr-final",
            )

            global_prompt = enriched["ltx"]["msr_global_prompt"]
            self.assertEqual(
                1,
                global_prompt.count("Continuity anchors (keep unchanged):"),
            )
            self.assertIn("black bob and graphite coat", global_prompt)
            self.assertIn("green lamps and brass shelves", global_prompt)
            self.assertNotIn("same as before", global_prompt.lower())
            self.assertNotIn("previous scene", global_prompt.lower())
            self.assertNotIn("prior shot", global_prompt.lower())
            self.assertNotIn("as before", global_prompt.lower())
            self.assertEqual(
                "msr-final",
                enriched["visual_consistency"]["workflow_profile"],
            )
            self.assertEqual(
                {
                    "actors": [{
                        "id": "mara",
                        "path": "movie/references/actor.png",
                    }],
                    "location": {
                        "id": "archive",
                        "path": "movie/references/location.png",
                    },
                },
                enriched["visual_consistency_sources"],
            )

    def test_movie_msr_enrichment_rejects_external_reference_with_context(self):
        from feverslop.application.movie_msr_enrichment import _enrich_shot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            project.mkdir()
            external = root / "external-actor.png"
            external.write_bytes(b"actor")
            location = project / "movie" / "references" / "location.png"
            location.parent.mkdir(parents=True)
            location.write_bytes(b"location")
            shot = {
                "shot_id": "shot_0001",
                "scene": 1,
                "description": "Mara enters.",
                "duration_seconds": 2,
                "reference_ids": {
                    "actors": ["mara"],
                    "location": "archive",
                },
            }
            manifest = {
                "actors": [{
                    "id": "mara",
                    "name": "Mara",
                    "visual_description": "black bob",
                    "msr_sheet_path": str(external),
                }],
                "locations": [{
                    "id": "archive",
                    "name": "Archive",
                    "visual_description": "green lamps",
                    "msr_sheet_path": location.relative_to(project).as_posix(),
                }],
            }

            with self.assertRaisesRegex(
                ValueError,
                r"mara.*project-relative.*external-actor\.png",
            ):
                _enrich_shot(
                    shot,
                    bible={"runtime_constraints": {"fps": 24}},
                    manifest=manifest,
                    fps=24,
                    project_dir=project,
                )

    def test_movie_msr_enrichment_uses_individual_manifest_images_for_vision(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        class VisionLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt_with_images(self, system_prompt, prompt, image_paths):
                self.calls.append((system_prompt, prompt, image_paths))
                return json.dumps({
                    "references": [
                        {"id": "mara", "type": "actor", "description": "Mara wears a graphite coat over a narrow red scarf, with a precise black bob"},
                        {"id": "ivo", "type": "actor", "description": "Ivo has swept silver hair, a white dinner jacket, and a dark carved cane"},
                        {"id": "archive", "type": "location", "description": "The archive has towering brass shelves, green lamps, and a wet black marble floor"},
                    ],
                    "relays": [{"index": 0, "prompt": "Mara steps between Ivo and the ledger, tightening one hand around its cover as Ivo raises his cane. The camera tracks sideways at waist height while green lamps flicker across the wet floor and both actors hold wary eye contact."}],
                })

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie = project / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True)
            paths = [refs / "mara.png", refs / "ivo.png", refs / "archive.png"]
            for path in paths:
                path.write_bytes(b"image")
            (movie / "bible.json").write_text(json.dumps({
                "actors": [{"id": "mara"}, {"id": "ivo"}], "locations": [{"id": "archive"}],
                "runtime_constraints": {"fps": 24},
            }), encoding="utf-8")
            (movie / "render_plan.json").write_text(json.dumps({"shots": [{
                "shot_id": "shot_7", "duration_seconds": 2, "description": "A confrontation",
                "reference_ids": {"actors": ["mara", "ivo"], "location": "archive"},
            }]}), encoding="utf-8")
            (refs / "manifest.json").write_text(json.dumps({
                "actors": [
                    {"id": "mara", "name": "Mara", "visual_description": "fallback Mara", "msr_sheet_path": "movie/references/mara.png"},
                    {"id": "ivo", "name": "Ivo", "visual_description": "fallback Ivo", "msr_sheet_path": "movie/references/ivo.png"},
                ],
                "locations": [{"id": "archive", "name": "Archive", "visual_description": "fallback Archive", "msr_sheet_path": "movie/references/archive.png"}],
            }), encoding="utf-8")
            llm = VisionLLM()
            statuses = []

            output = enrich_movie_render_plan_with_msr_prompts(
                project_dir=project, llm=llm, on_analysis_status=lambda shot, references: statuses.append((shot, references))
            )

            shot = json.loads(output.read_text(encoding="utf-8"))["shots"][0]
            self.assertEqual(paths, llm.calls[0][2])
            self.assertIn("graphite coat", shot["ltx"]["msr_global_prompt"])
            self.assertIn("swept silver hair", shot["ltx"]["msr_global_prompt"])
            self.assertIn("towering brass shelves", shot["ltx"]["msr_global_prompt"])
            relay = shot["ltx"]["msr_prompt_relay"][0]
            self.assertIn("steps between Ivo", relay["prompt"])
            self.assertNotIn("Target Description", relay["prompt"])
            self.assertNotIn("graphite coat", relay["prompt"])
            self.assertEqual((0, 47), (relay["frame_start"], relay["frame_end"]))
            self.assertEqual("shot_7", statuses[0][0])

    def test_movie_msr_partial_missing_images_keep_full_labeled_fallback(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        class VisionLLM:
            calls = []

            def complete_prompt_with_images(self, system_prompt, prompt, image_paths):
                self.calls.append(image_paths)
                return "{}"

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie = project / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True)
            (refs / "ivo.png").write_bytes(b"actor")
            (refs / "archive.png").write_bytes(b"location")
            (movie / "bible.json").write_text(json.dumps({"runtime_constraints": {"fps": 24}}), encoding="utf-8")
            (movie / "render_plan.json").write_text(json.dumps({"shots": [{
                "shot_id": "shot_partial", "duration_seconds": 1,
                "reference_ids": {"actors": ["mara", "ivo"], "location": "archive"},
            }]}), encoding="utf-8")
            (refs / "manifest.json").write_text(json.dumps({
                "actors": [
                    {"id": "mara", "name": "Mara", "visual_description": "Mara fallback", "msr_sheet_path": "movie/references/missing.png"},
                    {"id": "ivo", "name": "Ivo", "visual_description": "Ivo fallback", "msr_sheet_path": "movie/references/ivo.png"},
                ],
                "locations": [{"id": "archive", "name": "Archive", "visual_description": "Archive fallback", "msr_sheet_path": "movie/references/archive.png"}],
            }), encoding="utf-8")
            llm = VisionLLM()

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project, llm=llm)

            global_prompt = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]["msr_global_prompt"]
            self.assertIn("Reference image 1 (Mara): Mara fallback", global_prompt)
            self.assertIn("Reference image 2 (Ivo): Ivo fallback", global_prompt)
            self.assertIn("Reference image 3 (Scene): Archive fallback", global_prompt)
            self.assertEqual([], llm.calls)

    def test_movie_msr_transport_exception_is_logged_as_vision_unavailable(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        class FailingVisionLLM:
            def complete_prompt_with_images(self, system_prompt, prompt, image_paths):
                raise RuntimeError("transport failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie = project / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True)
            (refs / "mara.png").write_bytes(b"actor")
            (movie / "bible.json").write_text(json.dumps({"runtime_constraints": {"fps": 24}}), encoding="utf-8")
            (movie / "render_plan.json").write_text(json.dumps({"shots": [{
                "shot_id": "shot_transport", "duration_seconds": 1,
                "reference_ids": {"actors": ["mara"], "location": ""},
            }]}), encoding="utf-8")
            (refs / "manifest.json").write_text(json.dumps({
                "actors": [{"id": "mara", "name": "Mara", "visual_description": "Mara", "msr_sheet_path": "movie/references/mara.png"}],
                "locations": [],
            }), encoding="utf-8")

            with self.assertLogs("feverslop.application.movie_msr_enrichment", level="WARNING") as logs:
                enrich_movie_render_plan_with_msr_prompts(project_dir=project, llm=FailingVisionLLM())

            self.assertTrue(any("reason=vision unavailable" in message for message in logs.output))
            self.assertFalse(any("reason=invalid response" in message for message in logs.output))

    def test_movie_msr_vision_dialogue_relay_requests_spoken_lip_sync(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        class DialogueVisionLLM:
            prompt = ""
            system_prompt = ""

            def complete_prompt_with_images(self, system, prompt, _paths):
                self.system_prompt = system
                self.prompt = prompt
                return json.dumps({
                    "references": [{"id": "mara", "type": "actor", "description": "Mara has dark hair and a silver coat"}],
                    "relays": [{"index": 0, "prompt": "Mara speaks the line with precise lip sync and wary expression as the camera moves closer."}],
                })

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie = project / "movie"
            refs = movie / "references"
            refs.mkdir(parents=True)
            (refs / "mara.png").write_bytes(b"actor")
            (movie / "bible.json").write_text(json.dumps({"runtime_constraints": {"fps": 24}}))
            (movie / "render_plan.json").write_text(json.dumps({"shots": [{
                "shot_id": "shot_dialogue", "duration_seconds": 1,
                "dialogue": "Mara: It remembers me.",
                "reference_ids": {"actors": ["mara"], "location": ""},
            }]}))
            (refs / "manifest.json").write_text(json.dumps({
                "actors": [{"id": "mara", "name": "Mara", "msr_sheet_path": "movie/references/mara.png"}],
                "locations": [],
            }))
            llm = DialogueVisionLLM()

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project, llm=llm)
            relay = json.loads(output.read_text())["shots"][0]["ltx"]["msr_prompt_relay"][0]["prompt"]

            self.assertEqual("dialogue", json.loads(llm.prompt)["relay_segments"][0]["state"])
            self.assertIn("speaks", relay)
            self.assertIn("lip sync", relay)
            self.assertNotIn("mouth closed", relay)
            self.assertIn('state "dialogue"', llm.system_prompt)
            self.assertIn("speaks the provided dialogue with precise lip sync", llm.system_prompt)
            self.assertIn("instrumental and other non-vocal states keep mouths closed", llm.system_prompt.lower())
            self.assertNotIn("non-singing relays keep mouths closed", llm.system_prompt)


    def test_movie_msr_enrichment_enforces_dialogue_language_from_bible(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Archiv",
                        "premise": "Mara oeffnet ein verbotenes Buch.",
                        "story_arch": {"title": "Archiv", "premise": "Mara oeffnet ein verbotenes Buch.", "beats": ["opening"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24, "dialogue_language": "German"},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Archiv",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara opens the ledger",
                                "duration_seconds": 4,
                                "camera": "slow dolly",
                                "acting": "controlled fear",
                                "action": "opens the ledger",
                                "dialogue": "MARA: Es erinnert sich an mich.",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist", "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room", "msr_sheet_path": "movie/references/locations/archive/views/hero.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)

            enriched = json.loads(output.read_text(encoding="utf-8"))
            prompt = enriched["shots"][0]["ltx"]["original_style_i2v_prompt"]
            relay = enriched["shots"][0]["ltx"]["msr_prompt_relay"][0]
            self.assertNotIn("Dialogue language: German", prompt)
            self.assertNotIn("spoken in German", prompt)
            self.assertNotIn("Dialogue language: German", relay["prompt"])
            self.assertIn('Mara says in German: "Es erinnert sich an mich."', relay["prompt"])
            self.assertNotIn('"MARA:', relay["prompt"])
            global_prompt = enriched["shots"][0]["ltx"]["msr_global_prompt"]
            self.assertIn("Reference image 1 (Mara): stern archivist.", global_prompt)
            self.assertIn("Reference image 2 (Scene): quiet archive room.", global_prompt)
            self.assertNotIn("Dialogue language: German", global_prompt)
            self.assertNotIn("Audio contract:", global_prompt)
            self.assertNotIn("Style:", global_prompt)
            self.assertNotIn("MARA: Es erinnert sich an mich.", global_prompt)

    def test_movie_msr_enrichment_formats_radio_dialogue_as_diegetic_audio(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Station",
                        "premise": "Elara hears herself on the radio.",
                        "story_arch": {"title": "Station", "premise": "Elara hears herself on the radio.", "beats": ["warning"]},
                        "actors": [{"id": "elara", "name": "Elara", "visual_description": "weathered arctic technician in a frost-covered parka"}],
                        "locations": [{"id": "radio_room", "name": "Radio Room", "visual_description": "cramped warm room filled with glowing vacuum tubes and an old analog radio transmitter"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24, "dialogue_language": "German"},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Station",
                        "shots": [
                            {
                                "shot_id": "shot_0007",
                                "description": "Close up of Elara freezing as she hears a voice through the static",
                                "duration_seconds": 4,
                                "camera": "extreme close up on Elara's eyes",
                                "acting": "Elara's eyes widen in terror; she stares at the radio in disbelief",
                                "action": "The radio plays a recording of Elara's own voice screaming",
                                "dialogue": "(Radio Voice) Lauf weg! Er ist hier! Er kommt!",
                                "reference_ids": {"actors": ["elara"], "location": "radio_room"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "elara", "name": "Elara", "visual_description": "weathered arctic technician in a frost-covered parka", "msr_sheet_path": "actor.png"}],
                        "locations": [{"id": "radio_room", "name": "Radio Room", "visual_description": "cramped warm room filled with glowing vacuum tubes and an old analog radio transmitter", "msr_sheet_path": "location.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)

            ltx = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]
            local_prompt = ltx["msr_prompt_relay"][0]["prompt"]
            global_prompt = ltx["msr_global_prompt"]
            self.assertIn('The radio plays a recording of Elara\'s own voice screaming in German: "Lauf weg! Er ist hier! Er kommt!"', local_prompt)
            self.assertNotIn("Radio Voice", local_prompt)
            self.assertNotIn("Visible actor", local_prompt)
            self.assertNotIn("Elara says", local_prompt)
            self.assertNotIn("Elara asks", local_prompt)
            self.assertNotIn("Audio contract:", local_prompt)
            self.assertNotIn("Dialogue language:", local_prompt)
            self.assertEqual(
                "Reference image 1 (Elara): weathered arctic technician in a frost-covered parka. "
                "Reference image 2 (Scene): cramped warm room filled with glowing vacuum tubes and an old analog radio transmitter.",
                global_prompt,
            )

    def test_movie_msr_enrichment_blocks_dialogue_when_shot_has_no_dialogue(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Silent Shot",
                        "premise": "Mara opens a forbidden book.",
                        "story_arch": {"title": "Silent Shot", "premise": "Mara opens a forbidden book.", "beats": ["opening"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Silent Shot",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara silently opens the ledger",
                                "duration_seconds": 4,
                                "camera": "slow dolly",
                                "acting": "controlled fear",
                                "action": "opens the ledger without speaking",
                                "dialogue": "",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "stern archivist", "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room", "msr_sheet_path": "movie/references/locations/archive/views/hero.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)

            enriched = json.loads(output.read_text(encoding="utf-8"))
            prompt = enriched["shots"][0]["ltx"]["original_style_i2v_prompt"]
            relay = enriched["shots"][0]["ltx"]["msr_prompt_relay"][0]
            self.assertNotIn("No spoken dialogue", prompt)
            self.assertNotIn("Do not invent spoken lines", prompt)
            self.assertNotIn("No spoken dialogue", relay["prompt"])
            self.assertNotIn("Audio contract:", relay["prompt"])
            self.assertNotIn("Dialogue language:", relay["prompt"])
            self.assertNotIn("Style:", relay["prompt"])
            global_prompt = enriched["shots"][0]["ltx"]["msr_global_prompt"]
            self.assertIn("Reference image 1 (Mara): stern archivist.", global_prompt)
            self.assertIn("Reference image 2 (Scene): quiet archive room.", global_prompt)
            self.assertNotIn("No spoken dialogue", global_prompt)
            self.assertNotIn("No off-screen voice", global_prompt)
            self.assertNotIn("extra voice", global_prompt)

    def test_movie_msr_enrichment_drops_screenplay_dumps_from_continuity(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        screenplay_dump = (
            "EXT. SOMME VALLEY - DAY (1916): German soldiers move through fog; "
            "INT. TRENCH LINE - CONTINUOUS: KARL: Where is God? HANS: Not here."
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Mud",
                        "premise": "Soldiers lose faith in the trenches.",
                        "story_arch": {"title": "Mud", "premise": "Soldiers lose faith in the trenches.", "beats": ["opening"]},
                        "actors": [{"id": "hans", "name": "Hans", "visual_description": "mud-covered soldier"}],
                        "locations": [{"id": "trench", "name": "Trench Line", "visual_description": "muddy trench"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Mud",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Hans looks across the trench",
                                "duration_seconds": 4,
                                "camera": "static low angle",
                                "acting": "haunted silence",
                                "action": "stares into the shell smoke",
                                "dialogue": "",
                                "continuity_notes": f"{screenplay_dump}; Hans keeps the torn gray coat",
                                "reference_ids": {"actors": ["hans"], "location": "trench"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "continuity_plan.json").write_text(
                json.dumps(
                    {
                        "scene_continuity": {
                            "shot_0001": {
                                "incoming": [screenplay_dump],
                                "required_carryovers": [screenplay_dump, "Hans keeps the torn gray coat"],
                                "allowed_changes": ["smoke thickens"],
                                "outgoing": [screenplay_dump],
                            }
                        },
                        "narrative_chain": [
                            {
                                "shot_id": "shot_0001",
                                "story_state_before": screenplay_dump,
                                "story_state_after": "Hans understands the trench will not answer him.",
                                "cause_from_previous": screenplay_dump,
                                "narrative_purpose": "Show Hans losing faith.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "hans", "name": "Hans", "visual_description": "mud-covered soldier", "msr_sheet_path": "movie/references/actors/hans/msr_sheet.png"}],
                        "locations": [{"id": "trench", "name": "Trench Line", "visual_description": "muddy trench", "msr_sheet_path": "movie/references/locations/trench/views/hero.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)

            enriched = json.loads(output.read_text(encoding="utf-8"))
            continuity_notes = enriched["shots"][0]["continuity_notes"]
            prompt = enriched["shots"][0]["ltx"]["original_style_i2v_prompt"]
            relay_prompt = enriched["shots"][0]["ltx"]["msr_prompt_relay"][0]["prompt"]
            self.assertIn("Hans keeps the torn gray coat", continuity_notes)
            for value in (continuity_notes, prompt, relay_prompt):
                self.assertNotIn("EXT. SOMME VALLEY", value)
                self.assertNotIn("INT. TRENCH LINE", value)
                self.assertNotIn("KARL: Where is God?", value)
            for value in (prompt, relay_prompt):
                self.assertNotIn("Hans keeps the torn gray coat", value)
                self.assertNotIn("Continuity:", value)
                self.assertNotIn("CONTINUITY CONTRACT", value)

    def test_movie_msr_enrichment_uses_only_current_shot_facts_in_video_prompt(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Mud",
                        "premise": "Soldiers lose faith in the trenches.",
                        "story_arch": {"title": "Mud", "premise": "Soldiers lose faith in the trenches.", "beats": ["opening", "reply"]},
                        "actors": [
                            {"id": "hans", "name": "Hans", "visual_description": "mud-covered soldier"},
                            {"id": "karl", "name": "Karl", "visual_description": "trembling soldier"},
                        ],
                        "locations": [{"id": "trench", "name": "Trench Line", "visual_description": "muddy trench"}],
                        "continuity": [],
                        "style_constraints": ["desaturated trench realism"],
                        "runtime_constraints": {"fps": 24, "dialogue_language": "German"},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Mud",
                        "shots": [
                            {
                                "shot_id": "shot_0002",
                                "description": "Hans turns toward Karl in the trench",
                                "duration_seconds": 4,
                                "camera": "tight handheld close-up",
                                "acting": "controlled exhaustion",
                                "action": "Hans answers without raising his voice",
                                "dialogue": "Hans: Halt den Mund, Karl.",
                                "continuity_notes": "Karl was trembling beside the rifle; Hans keeps the torn gray coat",
                                "reference_ids": {"actors": ["hans", "karl"], "location": "trench"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "continuity_plan.json").write_text(
                json.dumps(
                    {
                        "scene_continuity": {
                            "shot_0002": {
                                "incoming": ["Karl whispered that he cannot feel the ground"],
                                "required_carryovers": ["Hans keeps the torn gray coat"],
                                "allowed_changes": ["Hans answers without raising his voice"],
                                "outgoing": ["Karl falls silent after Hans answers"],
                            }
                        },
                        "narrative_chain": [
                            {
                                "shot_id": "shot_0002",
                                "story_state_before": "Karl has just confessed fear in the previous beat.",
                                "story_state_after": "Hans shuts down the fear before it spreads.",
                                "cause_from_previous": "Karl whispered that he cannot feel the ground.",
                                "narrative_purpose": "Show Hans suppressing panic.",
                                "sets_up_next": "Friedrich enters with a memory of home.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [
                            {"id": "hans", "name": "Hans", "visual_description": "mud-covered soldier", "msr_sheet_path": "movie/references/actors/hans/msr_sheet.png"},
                            {"id": "karl", "name": "Karl", "visual_description": "trembling soldier", "msr_sheet_path": "movie/references/actors/karl/msr_sheet.png"},
                        ],
                        "locations": [{"id": "trench", "name": "Trench Line", "visual_description": "muddy trench", "msr_sheet_path": "movie/references/locations/trench/views/hero.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)

            enriched = json.loads(output.read_text(encoding="utf-8"))
            ltx = enriched["shots"][0]["ltx"]
            prompt = ltx["original_style_i2v_prompt"]
            relay_prompt = ltx["msr_prompt_relay"][0]["prompt"]
            for value in (prompt, relay_prompt):
                self.assertIn("Hans turns toward Karl in the trench", value)
                self.assertIn("Hans answers without raising his voice", value)
                self.assertIn('Hans says in German: "Halt den Mund, Karl."', value)
                self.assertNotIn('"Hans:', value)
                self.assertNotIn("Dialogue language: German", value)
                self.assertNotIn("Audio contract:", value)
                self.assertNotIn("Style:", value)
                self.assertNotIn("Actors:", value)
                self.assertNotIn("Location:", value)
                self.assertNotIn("Action:", value)
                self.assertNotIn("singing", value.lower())
                self.assertNotIn("chanting", value.lower())
                self.assertNotIn("background music", value.lower())
                self.assertNotIn("Continuity:", value)
                self.assertNotIn("CONTINUITY CONTRACT", value)
                self.assertNotIn("Karl whispered that he cannot feel the ground", value)
                self.assertNotIn("Friedrich enters with a memory of home", value)
            self.assertIn("Reference image 1 (Hans): mud-covered soldier.", ltx["msr_global_prompt"])
            self.assertIn("Reference image 2 (Karl): trembling soldier.", ltx["msr_global_prompt"])
            self.assertIn("Reference image 3 (Scene): muddy trench.", ltx["msr_global_prompt"])
            self.assertNotIn("Dialogue language: German", ltx["msr_global_prompt"])
            self.assertNotIn("Audio contract:", ltx["msr_global_prompt"])
            self.assertNotIn("No non-diegetic music", ltx["msr_global_prompt"])
            self.assertNotIn("quiet place", ltx["msr_global_prompt"])
            self.assertNotIn("ambient sounds", ltx["msr_global_prompt"])
            self.assertNotIn("Style:", ltx["msr_global_prompt"])
            self.assertNotIn("CONTINUITY CONTRACT", ltx["msr_global_prompt"])

    def test_movie_msr_global_prompt_excludes_audio_hints_for_indoor_locations(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Room",
                        "premise": "Mara waits indoors.",
                        "story_arch": {"title": "Room", "premise": "Mara waits indoors.", "beats": ["wait"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist"}],
                        "locations": [{"id": "archive", "name": "Archive Room", "visual_description": "small indoor archive room"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Room",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara waits beside the door",
                                "duration_seconds": 4,
                                "action": "Mara listens without moving",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist", "msr_sheet_path": "actor.png"}],
                        "locations": [{"id": "archive", "name": "Archive Room", "visual_description": "small indoor archive room", "msr_sheet_path": "location.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            global_prompt = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]["msr_global_prompt"]

            self.assertIn("Reference image 1 (Mara): quiet archivist.", global_prompt)
            self.assertIn("Reference image 2 (Scene): small indoor archive room.", global_prompt)
            self.assertNotIn("in a quiet room", global_prompt)
            self.assertNotIn("ambient sounds", global_prompt)

    def test_movie_msr_global_prompt_ignores_structured_style_constraints(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Room",
                        "premise": "Mara waits indoors.",
                        "story_arch": {"title": "Room", "premise": "Mara waits indoors.", "beats": ["wait"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist"}],
                        "locations": [{"id": "archive", "name": "Archive Room", "visual_description": "small indoor archive room"}],
                        "continuity": [],
                        "style_constraints": [{"word_count_min": 40}, "desaturated realism"],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Room",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara waits beside the door",
                                "duration_seconds": 4,
                                "action": "Mara listens without moving",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist", "msr_sheet_path": "actor.png"}],
                        "locations": [{"id": "archive", "name": "Archive Room", "visual_description": "small indoor archive room", "msr_sheet_path": "location.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            global_prompt = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]["msr_global_prompt"]

            self.assertIn("Reference image 1 (Mara): quiet archivist.", global_prompt)
            self.assertIn("Reference image 2 (Scene): small indoor archive room.", global_prompt)
            self.assertNotIn("Style:", global_prompt)
            self.assertNotIn("desaturated realism", global_prompt)
            self.assertNotIn("word_count_min", global_prompt)

    def test_movie_msr_enrichment_drops_planning_dumps_from_local_prompt_fields(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Room",
                        "premise": "Mara waits indoors.",
                        "story_arch": {"title": "Room", "premise": "Mara waits indoors.", "beats": ["wait"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist"}],
                        "locations": [{"id": "archive", "name": "Archive Room", "visual_description": "small indoor archive room"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Room",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara waits beside the door. story_idea: dumped screenplay",
                                "duration_seconds": 4,
                                "action": "Mara listens without moving. prompt_guidance: {\"word_count_min\": 40}",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist", "msr_sheet_path": "actor.png"}],
                        "locations": [{"id": "archive", "name": "Archive Room", "visual_description": "small indoor archive room", "msr_sheet_path": "location.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            prompt = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]["msr_prompt_relay"][0]["prompt"]

            self.assertIn("Mara waits beside the door", prompt)
            self.assertIn("Mara listens without moving", prompt)
            self.assertNotIn("Action:", prompt)
            self.assertNotIn("Actors:", prompt)
            self.assertNotIn("Location:", prompt)
            self.assertNotIn("story_idea", prompt)
            self.assertNotIn("prompt_guidance", prompt)
            self.assertNotIn("word_count_min", prompt)

    def test_movie_msr_prompts_use_clean_reference_format_and_avoid_action_duplication(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Station",
                        "premise": "Lena hears knocking.",
                        "story_arch": {"title": "Station", "premise": "Lena hears knocking.", "beats": ["knock"]},
                        "actors": [{"id": "tech", "name": "Technikerin", "visual_description": "Lena, frost-covered field technician in a red parka"}],
                        "locations": [{"id": "corridor", "name": "KORRIDOR DER STATION - NACHT", "visual_description": "narrow frozen station corridor with flickering emergency lights"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24, "dialogue_language": "German"},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Station",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Hinter einer Metalltür klopft etwas langsam von innen.",
                                "duration_seconds": 4,
                                "action": "Hinter einer Metalltür klopft etwas langsam von innen.",
                                "camera": "controlled interior dolly",
                                "acting": "silent physical reaction",
                                "dialogue": "",
                                "reference_ids": {"actors": ["tech"], "location": "corridor"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [
                            {
                                "id": "tech",
                                "name": "Technikerin",
                                "role": "character",
                                "visual_description": "Technikerin, character, Lena, frost-covered field technician in a red parka",
                                "image_prompt": "Full-body cinematic character reference sheet for Technikerin. Four vertical panels.",
                                "msr_sheet_path": "actor.png",
                            }
                        ],
                        "locations": [
                            {
                                "id": "corridor",
                                "name": "KORRIDOR DER STATION - NACHT",
                                "visual_description": "KORRIDOR DER STATION - NACHT, narrow frozen station corridor with flickering emergency lights",
                                "msr_sheet_path": "location.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            ltx = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]
            global_prompt = ltx["msr_global_prompt"]
            local_prompt = ltx["msr_prompt_relay"][0]["prompt"]

            self.assertIn("Reference image 1 (Technikerin): Lena, frost-covered field technician in a red parka.", global_prompt)
            self.assertIn("Reference image 2 (Scene): narrow frozen station corridor with flickering emergency lights.", global_prompt)
            self.assertNotIn("Use reference image", global_prompt)
            self.assertNotIn("Background reference", global_prompt)
            self.assertNotIn("character, Lena", global_prompt)
            self.assertEqual(1, local_prompt.count("Hinter einer Metalltür klopft etwas langsam von innen"))

    def test_movie_msr_reference_prompt_keeps_placeholder_references_without_story_defined_text(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Station",
                        "premise": "Lena hears knocking.",
                        "story_arch": {"title": "Station", "premise": "Lena hears knocking.", "beats": ["knock"]},
                        "actors": [{"id": "tech", "name": "Technikerin", "visual_description": "Technikerin, story-defined cinematic character with consistent face, hair, body shape, wardrobe, and posture"}],
                        "locations": [{"id": "corridor", "name": "KORRIDOR DER STATION - NACHT", "visual_description": "KORRIDOR DER STATION - NACHT, story-defined cinematic location with consistent production design, geography, lighting, and atmosphere"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Station",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Hinter einer Metalltür klopft etwas langsam von innen.",
                                "duration_seconds": 4,
                                "reference_ids": {"actors": ["tech"], "location": "corridor"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "tech", "name": "Technikerin", "role": "character", "visual_description": "Technikerin, story-defined cinematic character with", "msr_sheet_path": "actor.png"}],
                        "locations": [{"id": "corridor", "name": "KORRIDOR DER STATION - NACHT", "visual_description": "KORRIDOR DER STATION - NACHT, story-defined cinematic location with consistent production design, geography, lighting, and atmosphere", "msr_sheet_path": "location.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            global_prompt = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]["msr_global_prompt"]

            self.assertIn("Reference image 1 (Technikerin): Technikerin.", global_prompt)
            self.assertIn("Reference image 2 (Scene): KORRIDOR DER STATION - NACHT.", global_prompt)
            self.assertNotIn("story-defined", global_prompt)
            self.assertNotIn("consistent production design", global_prompt)

    def test_movie_msr_workflow_prompt_relay_node_excludes_continuity_contract(self):
        from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            actor_sheet = project / "movie" / "references" / "actors" / "hans" / "msr_sheet.png"
            location_sheet = project / "movie" / "references" / "locations" / "trench" / "views" / "hero.png"
            actor_sheet.parent.mkdir(parents=True)
            location_sheet.parent.mkdir(parents=True)
            Image.new("RGB", (10, 10), color=(255, 0, 0)).save(actor_sheet)
            Image.new("RGB", (10, 10), color=(0, 0, 255)).save(location_sheet)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Mud",
                        "premise": "Hans answers Karl in the trench.",
                        "story_arch": {"title": "Mud", "premise": "Hans answers Karl in the trench.", "beats": ["reply"]},
                        "actors": [{"id": "hans", "name": "Hans", "visual_description": "mud-covered soldier"}],
                        "locations": [{"id": "trench", "name": "Trench Line", "visual_description": "muddy trench"}],
                        "continuity": [],
                        "style_constraints": ["desaturated trench realism"],
                        "runtime_constraints": {"fps": 24, "dialogue_language": "German"},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Mud",
                        "fps": 24,
                        "shots": [
                            {
                                "shot_id": "shot_0002",
                                "description": "Hans turns toward Karl in the trench",
                                "duration_seconds": 4,
                                "camera": "tight handheld close-up",
                                "acting": "controlled exhaustion",
                                "action": "Hans answers without raising his voice",
                                "dialogue": "Hans: Halt den Mund, Karl.",
                                "continuity_notes": "Karl whispered that he cannot feel the ground; Hans keeps the torn gray coat",
                                "reference_ids": {"actors": ["hans"], "location": "trench"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "continuity_plan.json").write_text(
                json.dumps(
                    {
                        "scene_continuity": {
                            "shot_0002": {
                                "incoming": ["Karl whispered that he cannot feel the ground"],
                                "required_carryovers": ["Hans keeps the torn gray coat"],
                                "outgoing": ["Karl falls silent after Hans answers"],
                            }
                        },
                        "narrative_chain": [
                            {
                                "shot_id": "shot_0002",
                                "story_state_before": "Karl has just confessed fear.",
                                "sets_up_next": "Friedrich enters with a memory of home.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "hans", "name": "Hans", "visual_description": "mud-covered soldier", "msr_sheet_path": actor_sheet.relative_to(project).as_posix()}],
                        "locations": [{"id": "trench", "name": "Trench Line", "visual_description": "muddy trench", "msr_sheet_path": location_sheet.relative_to(project).as_posix()}],
                    }
                ),
                encoding="utf-8",
            )
            workflow_path = project / "workflow.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                        "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                        "27": {"inputs": {"global_prompt": "", "local_prompts": "", "segment_lengths": ""}, "_meta": {"title": "#PROMPT_RELAY"}},
                        "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    }
                ),
                encoding="utf-8",
            )

            render_plan_msr = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            plan = json.loads(render_plan_msr.read_text(encoding="utf-8"))
            scene = ComfyUIMovieVisualAdapter(client=object(), workflow_path=workflow_path)._movie_scenes(plan, project_dir=project)[0]
            backend = ComfyUIMSRVideoRenderBackend(
                client=object(),
                workflow_path=workflow_path,
                output_dir=project / "output",
                project_dir=project,
                asset_uploader=NativeAudioAssetUploader(),
            )

            patched = backend.build_workflow(
                scene,
                prompt=scene["ltx"]["original_style_i2v_prompt"],
                rolling={
                    "render_frame_count": 171,
                    "trim_front_frames": 50,
                    "tail_loss_frames": 25,
                },
            )

            relay_inputs = patched["27"]["inputs"]
            self.assertIn("Reference image 1 (Hans): mud-covered soldier.", relay_inputs["global_prompt"])
            self.assertIn("Reference image 2 (Scene): muddy trench.", relay_inputs["global_prompt"])
            self.assertIn("Hans turns toward Karl in the trench", relay_inputs["local_prompts"])
            self.assertIn('Hans says in German: "Halt den Mund, Karl."', relay_inputs["local_prompts"])
            self.assertNotIn('"Hans:', relay_inputs["local_prompts"])
            self.assertEqual(relay_inputs["local_prompts"], plan["shots"][0]["ltx"]["msr_prompt_relay"][0]["prompt"])
            self.assertEqual("170", relay_inputs["segment_lengths"])
            self.assertNotIn("Cinematic atmosphere gathers", relay_inputs["local_prompts"])
            self.assertNotIn("carries the last motion", relay_inputs["local_prompts"])
            self.assertNotIn("singing", relay_inputs["local_prompts"].lower())
            self.assertNotIn("chanting", relay_inputs["local_prompts"].lower())
            self.assertNotIn("background music", relay_inputs["local_prompts"].lower())
            self.assertNotIn("Continuity:", relay_inputs["local_prompts"])
            self.assertNotIn("CONTINUITY CONTRACT", relay_inputs["local_prompts"])
            self.assertNotIn("Actors:", relay_inputs["local_prompts"])
            self.assertNotIn("Location:", relay_inputs["local_prompts"])
            self.assertNotIn("Action:", relay_inputs["local_prompts"])
            self.assertNotIn("Audio contract:", relay_inputs["global_prompt"])
            self.assertNotIn("Dialogue language:", relay_inputs["global_prompt"])
            self.assertNotIn("Style:", relay_inputs["global_prompt"])
            self.assertNotIn("Karl whispered that he cannot feel the ground", relay_inputs["local_prompts"])
            self.assertNotIn("Friedrich enters with a memory of home", relay_inputs["local_prompts"])

    def test_movie_msr_silent_shot_prompt_avoids_music_and_vocal_trigger_words(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            movie_dir.mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Silent",
                        "premise": "Mara waits in a quiet archive.",
                        "story_arch": {"title": "Silent", "premise": "Mara waits in a quiet archive.", "beats": ["wait"]},
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "dusty archive"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24, "dialogue_language": "English"},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Silent",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara listens to dust falling through the archive light",
                                "duration_seconds": 4,
                                "action": "Mara stands still and breathes quietly",
                                "reference_ids": {"actors": ["mara"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "continuity_plan.json").write_text("{}", encoding="utf-8")
            (movie_dir / "references" / "manifest.json").parent.mkdir(parents=True)
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "visual_description": "quiet archivist", "msr_sheet_path": "actor.png"}],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "dusty archive", "msr_sheet_path": "location.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)
            ltx = json.loads(output.read_text(encoding="utf-8"))["shots"][0]["ltx"]
            prompt = ltx["msr_prompt_relay"][0]["prompt"]

            self.assertNotIn("Audio contract:", prompt)
            self.assertNotIn("Audio contract:", ltx["msr_global_prompt"])
            self.assertNotIn("No non-diegetic music", ltx["msr_global_prompt"])
            self.assertNotIn("Dialogue language:", prompt)
            self.assertNotIn("Style:", prompt)
            self.assertNotIn("music", prompt.lower())
            self.assertNotIn("score", prompt.lower())
            self.assertNotIn("soundtrack", prompt.lower())
            self.assertNotIn("singing", prompt.lower())
            self.assertNotIn("chanting", prompt.lower())

    def test_movie_continuity_fallback_drops_screenplay_dumps_from_carryovers(self):
        from feverslop.application.movie import build_movie_continuity_fallback
        from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieContinuityRule, MovieLocation, StoryArch

        screenplay_dump = (
            "EXT. SOMME VALLEY - DAY (1916): German soldiers move through fog; "
            "INT. TRENCH LINE - CONTINUOUS: KARL: Where is God? HANS: Not here."
        )
        bible = MovieBible(
            title="Mud",
            premise="Soldiers lose faith in the trenches.",
            story_arch=StoryArch(title="Mud", premise="Soldiers lose faith in the trenches.", beats=("opening",)),
            actors=(MovieActor(id="hans", name="Hans", visual_description="mud-covered soldier"),),
            locations=(MovieLocation(id="trench", name="Trench Line", visual_description="muddy trench"),),
            continuity=(
                MovieContinuityRule(id="script", description=screenplay_dump),
                MovieContinuityRule(id="coat", description="Hans keeps the torn gray coat"),
            ),
            style_constraints=(),
            runtime_constraints={},
        )
        shots = (
            CinematicShot(
                shot_id="shot_0001",
                description="Hans looks across the trench",
                duration_seconds=4,
                camera="static low angle",
                action="stares into smoke",
                expression="haunted silence",
                location="Trench Line",
                actor_ids=("hans",),
                location_id="trench",
            ),
        )

        plan = build_movie_continuity_fallback(bible=bible, shots=shots)
        carryovers = plan.scene_continuity["shot_0001"].required_carryovers

        self.assertNotIn(screenplay_dump, carryovers)
        self.assertIn("Hans keeps the torn gray coat", carryovers)

    def test_movie_msr_enrichment_writes_multi_actor_global_reference_prompt(self):
        from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie_dir = project / "movie"
            (movie_dir / "references").mkdir(parents=True)
            (movie_dir / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Duel",
                        "premise": "Mara and Ivo confront each other.",
                        "story_arch": {"title": "Duel", "premise": "Mara and Ivo confront each other.", "beats": ["confrontation"]},
                        "actors": [
                            {"id": "mara", "name": "Mara", "role": "archivist", "visual_description": "charcoal coat and cropped black hair"},
                            {"id": "ivo", "name": "Ivo", "role": "rival", "visual_description": "white suit and silver cane"},
                        ],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room"}],
                        "continuity": [],
                        "style_constraints": [],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Duel",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "description": "Mara and Ivo face each other across the archive table",
                                "duration_seconds": 4,
                                "camera": "slow lateral track",
                                "acting": "restrained suspicion",
                                "action": "both characters hold their ground",
                                "reference_ids": {"actors": ["mara", "ivo"], "location": "archive"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (movie_dir / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [
                            {"id": "mara", "name": "Mara", "role": "archivist", "visual_description": "charcoal coat and cropped black hair", "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png"},
                            {"id": "ivo", "name": "Ivo", "role": "rival", "visual_description": "white suit and silver cane", "msr_sheet_path": "movie/references/actors/ivo/msr_sheet.png"},
                        ],
                        "locations": [{"id": "archive", "name": "Archive", "visual_description": "quiet archive room", "msr_sheet_path": "movie/references/locations/archive/views/hero.png"}],
                    }
                ),
                encoding="utf-8",
            )

            output = enrich_movie_render_plan_with_msr_prompts(project_dir=project)

            enriched = json.loads(output.read_text(encoding="utf-8"))
            ltx = enriched["shots"][0]["ltx"]
            self.assertIn("Reference image 1 (Mara): charcoal coat and cropped black hair.", ltx["msr_global_prompt"])
            self.assertIn("charcoal coat", ltx["msr_global_prompt"])
            self.assertIn("Reference image 2 (Ivo): white suit and silver cane.", ltx["msr_global_prompt"])
            self.assertIn("white suit", ltx["msr_global_prompt"])
            self.assertIn("Reference image 3 (Scene): quiet archive room.", ltx["msr_global_prompt"])
            self.assertIn("Visible cast: Mara (`mara`) and Ivo (`ivo`)", ltx["original_style_i2v_prompt"])

    def test_movie_orchestrator_scaffolds_story_arch_and_render_plan(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
                artifact_writer=LocalMovieArtifactWriter(),
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

            def generate_movie_bible(self, **_kwargs):
                return {}

            def plan_shots_from_bible(self, **_kwargs):
                return self.plan_shots()

            def generate_movie_continuity_plan(self, **_kwargs):
                return {}

            def generate_movie_story_design(self, **_kwargs):
                return {}

            def generate_movie_screenplay(self, **_kwargs):
                return {}

            def generate_movie_narrative_plan(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=Planner(), projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
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
            self.assertIn("Seductive Succubus", succubus["visual_description"])
            self.assertNotIn("Full-body cinematic character reference sheet", succubus["visual_description"])
            self.assertNotIn("Four vertical panels", succubus["visual_description"])
            self.assertNotIn("drawn from the story premise", succubus["prompt"])
            self.assertNotIn("visual identity", succubus["prompt"].lower())
            self.assertNotIn("come, rest your weary soul", succubus["prompt"].lower())
            self.assertNotIn("The Desolate Void", succubus["prompt"])
            self.assertIn("Four vertical panels in one image", succubus["prompt"])
            self.assertIn("plain white seamless studio background", succubus["prompt"])

    def test_movie_scaffold_actor_prompts_prefer_solo_shots_over_shared_montage(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.domain.movie import CinematicShot, StoryArch

        class Planner:
            def generate_story_arch(self, **_kwargs):
                return StoryArch(title="Void", premise="A man fights a succubus and goat demon.", beats=("beat",))

            def plan_shots(self, **_kwargs):
                return (
                    CinematicShot(
                        shot_id="shot_0001",
                        description="A low-angle shot of the goat demon, towering and muscular, with massive curved horns.",
                        duration_seconds=4,
                        camera="low angle",
                        action="The goat demon bellows, shaking the dreamscape.",
                        expression="Menacing, primal",
                        location="Dreamscape",
                        actor_ids=("the_goat_demon",),
                        location_id="dreamscape",
                    ),
                    CinematicShot(
                        shot_id="shot_0002",
                        description="A split-screen composition showing the succubus on one side and the demon on the other.",
                        duration_seconds=4,
                        camera="split-screen",
                        action="The two entities converge on the man from opposite sides.",
                        expression="Seductive vs. Ferocious",
                        location="Dreamscape",
                        actor_ids=("the_succubus", "the_goat_demon"),
                        location_id="dreamscape",
                    ),
                )

            def generate_movie_bible(self, **_kwargs):
                return {}

            def plan_shots_from_bible(self, **_kwargs):
                return self.plan_shots()

            def generate_movie_continuity_plan(self, **_kwargs):
                return {}

            def generate_movie_story_design(self, **_kwargs):
                return {}

            def generate_movie_screenplay(self, **_kwargs):
                return {}

            def generate_movie_narrative_plan(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(planner=Planner(), projects_root=Path(temp_dir), artifact_writer=LocalMovieArtifactWriter()).execute(
                MovieInput(
                    name="Void",
                    source_type="short_story",
                    story_text="A man fights a succubus and goat demon inside a dreamscape.",
                    desired_length=8,
                )
            )

            manifest = json.loads(result.reference_manifest_path.read_text())
            goat = next(actor for actor in manifest["actors"] if actor["id"] == "the_goat_demon")
            self.assertIn("The Goat Demon", goat["visual_description"])
            self.assertNotIn("Four vertical panels", goat["visual_description"])
            self.assertNotIn("massive curved horns", goat["visual_description"])
            self.assertNotIn("split-screen", goat["prompt"].lower())
            self.assertNotIn("succubus", goat["prompt"].lower())
            self.assertNotIn("low-angle", goat["prompt"].lower())
            self.assertNotIn("shot", goat["prompt"].lower())
            self.assertNotIn("bellows", goat["prompt"].lower())
            self.assertNotIn("visual identity", goat["prompt"].lower())
            self.assertIn("Four vertical panels in one image", goat["prompt"])

    def test_movie_actor_reference_prompts_drop_closeup_and_motion_cues(self):
        from feverslop.application.movie import build_movie_actor_reference_prompt, build_movie_actor_visual_description

        visual_description = build_movie_actor_visual_description(
            "Extreme close-up of the man's eye fluttering as the screen fades; "
            "The man's eyes roll back as he enters a trance-like sleep; "
            "Unsettled, drifting",
        )
        prompt = build_movie_actor_reference_prompt("The Man", visual_description)

        self.assertNotIn("close-up", visual_description.lower())
        self.assertNotIn("eye fluttering", visual_description.lower())
        self.assertNotIn("eyes roll back", visual_description.lower())
        self.assertNotIn("screen fades", visual_description.lower())
        self.assertIn("Unsettled, drifting", visual_description)
        self.assertNotIn("Four vertical panels", visual_description)
        self.assertIn("Four vertical panels in one image", prompt)

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
                    artifact_writer=LocalMovieArtifactWriter(),
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
                artifact_writer=LocalMovieArtifactWriter(),
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

    def test_screenplay_scaffold_accepts_markdown_scene_headings_and_parentheticals(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        screenplay = """
        **TITLE: THE MUD AND THE SILENCE**

        **SCENE 1**
        **EXT. SOMME VALLEY - DAY (1916)**

        Grey mist and churned earth fill the valley.

        **SCENE 2**
        **INT. TRENCH LINE - CONTINUOUS**

        HANS (22, eyes hollow) leans against a muddy wall. KARL cleans a rifle.

        KARL
        (whispering)
        *Ich kann den Boden nicht mehr spüren. Es ist nur noch Matsch.*
        (I can't feel the ground anymore. It's just mud.)

        HANS
        *Halt den Mund, Karl. Die Stille ist gefährlicher als der Lärm.*
        (Shut up, Karl. The silence is more dangerous than the noise.)
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
                artifact_writer=LocalMovieArtifactWriter(),
            ).execute(
                MovieInput(
                    name="Door Below",
                    source_type="screenplay",
                    story_text=screenplay,
                    desired_length=20,
                    mode="scaffold",
                    config={"dialogue_language": "german"},
                )
            )

            screenplay_artifact = json.loads(result.screenplay_path.read_text(encoding="utf-8"))
            render_plan = json.loads(result.render_plan_path.read_text(encoding="utf-8"))

            self.assertEqual(2, len(screenplay_artifact["scenes"]))
            self.assertEqual("EXT. SOMME VALLEY - DAY (1916)", screenplay_artifact["scenes"][0]["heading"])
            self.assertEqual("INT. TRENCH LINE - CONTINUOUS", screenplay_artifact["scenes"][1]["heading"])
            self.assertIn("KARL: Ich kann den Boden nicht mehr spüren. Es ist nur noch Matsch.", screenplay_artifact["scenes"][1]["dialogue"])
            self.assertIn("HANS: Halt den Mund, Karl. Die Stille ist gefährlicher als der Lärm.", screenplay_artifact["scenes"][1]["dialogue"])
            self.assertNotIn("whispering", screenplay_artifact["scenes"][1]["dialogue"])
            self.assertNotIn("I can't feel the ground", screenplay_artifact["scenes"][1]["dialogue"])
            self.assertEqual("TRENCH LINE - CONTINUOUS", render_plan["shots"][1]["location"])
            self.assertIn("Ich kann den Boden nicht mehr spüren", render_plan["shots"][1]["dialogue"])

    def test_movie_two_ref_edit_workflow_exposes_required_anchors(self):
        workflow = json.loads(Path("workflows/image_edit_flux2_klein_2ref_v1.json").read_text(encoding="utf-8-sig"))
        titles = {str(node.get("_meta", {}).get("title") or "") for node in workflow.values()}

        self.assertIn("#BASE_IMAGE", titles)
        self.assertIn("#CHARACTER_REF", titles)
        self.assertIn("#PROMPT_POSITIVE", titles)
        self.assertIn("#PROMPT_NEGATIVE", titles)
        self.assertIn("#SAVE_IMAGE", titles)

    def test_movie_two_ref_edit_backend_patches_plate_character_and_prompt(self):
        from feverslop.adapters.movie_edit_image_backend import MovieTwoRefEditImageBackend

        class FakeClient:
            def __init__(self):
                self.uploads = []
                self.queued = None

            def upload_image(self, path, *, subfolder, file_type, overwrite):
                self.uploads.append(Path(path).name)
                return {"name": Path(path).name, "subfolder": subfolder, "type": file_type}

            def comfy_path_from_upload(self, upload):
                return f"{upload['subfolder']}/{upload['name']}"

            def queue_prompt(self, workflow):
                self.queued = workflow
                return "prompt-1"

            def wait_for_completion(self, prompt_id):
                return {"prompt_id": prompt_id}

            def extract_output_images(self, history):
                return [{"filename": "out.png", "subfolder": "", "type": "output"}]

            def download_view_file(self, *, filename, subfolder, file_type, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"png")
                return output_path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow_path = root / "edit.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"class_type": "LoadImage", "inputs": {"image": ""}, "_meta": {"title": "#BASE_IMAGE"}},
                        "2": {"class_type": "LoadImage", "inputs": {"image": ""}, "_meta": {"title": "#CHARACTER_REF"}},
                        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "#PROMPT_POSITIVE"}},
                        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "#PROMPT_NEGATIVE"}},
                        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_IMAGE"}},
                    }
                ),
                encoding="utf-8",
            )
            plate = root / "plate.png"
            character = root / "character.png"
            plate.write_bytes(b"plate")
            character.write_bytes(b"character")
            client = FakeClient()

            output = MovieTwoRefEditImageBackend(client=client, workflow_path=workflow_path).render_edit(
                scene_number=1,
                prompt="Add only Leo.",
                plate_image=plate,
                character_image=character,
                output_dir=root / "out",
                pass_number=2,
            )

        self.assertEqual(root / "out" / "scene_0001_pass_02.png", output)
        self.assertEqual(["plate.png", "character.png"], client.uploads)
        self.assertEqual("feverslop/movie_edit/plate.png", client.queued["1"]["inputs"]["image"])
        self.assertEqual("feverslop/movie_edit/character.png", client.queued["2"]["inputs"]["image"])
        self.assertEqual("Add only Leo.", client.queued["3"]["inputs"]["text"])
        self.assertEqual("movie_edit/scene_0001_pass_02", client.queued["5"]["inputs"]["filename_prefix"])

    def test_screenplay_scaffold_preserves_multilingual_dialogue_without_translations(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        screenplay = """
        EXT. STONE CLEARING - DUSK

        MORWENNA
        (in archaic French)
        Tu tardes.
        (You're late.)

        LEO
        Where am I?
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
                artifact_writer=LocalMovieArtifactWriter(),
            ).execute(
                MovieInput(
                    name="Blackwood",
                    source_type="screenplay",
                    story_text=screenplay,
                    desired_length=12,
                    mode="scaffold",
                )
            )

            screenplay_artifact = json.loads(result.screenplay_path.read_text(encoding="utf-8"))
            dialogue = screenplay_artifact["scenes"][0]["dialogue"]

        self.assertIn("MORWENNA: Tu tardes.", dialogue)
        self.assertIn("LEO: Where am I?", dialogue)
        self.assertNotIn("You're late", dialogue)
        self.assertNotIn("in archaic French", dialogue)

    def test_movie_visual_plan_derives_scene_views_and_shot_blocking(self):
        from feverslop.application.movie_visual_plan import build_movie_visual_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            (project / "movie").mkdir(parents=True)
            (project / "movie" / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Blackwood",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "scene": 1,
                                "description": "Leo enters the stone clearing.",
                                "camera": "wide exterior establishing move",
                                "action": "Leo sees Morwenna sorting roots at the stone.",
                                "location": "STONE CLEARING - DUSK",
                                "location_id": "stone_clearing",
                                "actor_ids": ["leo", "morwenna"],
                                "dialogue": "MORWENNA: Tu tardes.",
                                "duration_seconds": 5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (project / "movie" / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Blackwood",
                        "actors": [
                            {"id": "leo", "name": "Leo", "visual_description": "modern hiker in expensive outdoor gear"},
                            {"id": "morwenna", "name": "Morwenna", "visual_description": "ancient witch with iron-gray braided hair"},
                        ],
                        "locations": [
                            {"id": "stone_clearing", "name": "Stone Clearing", "visual_description": "mossy forest clearing with a flat ritual stone"},
                        ],
                        "runtime_constraints": {"fps": 24},
                    }
                ),
                encoding="utf-8",
            )
            (project / "movie" / "references").mkdir()
            (project / "movie" / "references" / "manifest.json").write_text(
                json.dumps(
                    {
                        "actors": [
                            {"id": "leo", "msr_sheet_path": "movie/references/actors/leo/msr_sheet.png"},
                            {"id": "morwenna", "sheet_path": "movie/references/actors/morwenna/sheet.png"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = build_movie_visual_plan(project_dir=project)
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual("Blackwood", data["title"])
        self.assertEqual("stone_clearing", data["scene_views"][0]["location_id"])
        self.assertIn(data["shots"][0]["view_id"], {view["view_id"] for view in data["scene_views"]})
        self.assertEqual(["leo", "morwenna"], data["shots"][0]["selected_actor_ids"])
        self.assertIn("Visible cast: `leo`, `morwenna`", data["shots"][0]["video_prompt"])
        self.assertIn("Scene-only background plate", data["shots"][0]["base_plate_prompt"])
        self.assertEqual(["leo", "morwenna"], [item["actor_id"] for item in data["shots"][0]["edit_passes"]])
        self.assertIn("Add only leo", data["shots"][0]["edit_passes"][0]["prompt"])
        self.assertEqual("movie/references/actors/leo/msr_sheet.png", data["shots"][0]["edit_passes"][0]["reference_image_path"])
        self.assertEqual("movie/references/actors/morwenna/sheet.png", data["shots"][0]["edit_passes"][1]["reference_image_path"])

    def test_movie_visual_plan_edit_prompts_anchor_characters_to_floor_and_screen_zones(self):
        from feverslop.application.movie_visual_plan import build_movie_visual_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            (project / "movie").mkdir(parents=True)
            (project / "movie" / "render_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Blackwood",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "scene": 1,
                                "description": "Morwenna waits inside the hut.",
                                "camera": "wide interior view",
                                "action": "Morwenna stands near the hearth while Leo enters.",
                                "location": "STONE HUT - DAY",
                                "location_id": "stone_hut",
                                "actor_ids": ["morwenna", "leo"],
                                "duration_seconds": 5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (project / "movie" / "bible.json").write_text(
                json.dumps(
                    {
                        "title": "Blackwood",
                        "actors": [{"id": "morwenna", "name": "Morwenna"}, {"id": "leo", "name": "Leo"}],
                        "locations": [{"id": "stone_hut", "name": "Stone Hut", "visual_description": "stone interior with hearth, baskets, jars, and tools"}],
                    }
                ),
                encoding="utf-8",
            )

            output = build_movie_visual_plan(project_dir=project)
            data = json.loads(output.read_text(encoding="utf-8"))

        first_pass = data["shots"][0]["edit_passes"][0]
        second_pass = data["shots"][0]["edit_passes"][1]
        self.assertEqual("center-right foreground floor area", first_pass["placement_zone"])
        self.assertEqual("center-left foreground floor area", second_pass["placement_zone"])
        self.assertIn("Place morwenna in the center-right foreground floor area", first_pass["prompt"])
        self.assertIn("feet must touch the visible floor plane", first_pass["prompt"])
        self.assertIn("Do not place the character on shelves, baskets, barrels, pots, tools, furniture, walls, or inside containers", first_pass["prompt"])
        self.assertIn("full-body standing human scale", first_pass["prompt"])

    def test_movie_i2v_render_plan_matches_classic_render_contract(self):
        from feverslop.application.movie_i2v_render_plan import write_movie_i2v_render_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            (project / "movie").mkdir(parents=True)
            (project / "movie" / "visual_plan.json").write_text(
                json.dumps(
                    {
                        "title": "Blackwood",
                        "shots": [
                            {
                                "shot_id": "shot_0001",
                                "scene": 1,
                                "duration_seconds": 5,
                                "view_id": "stone_clearing_wide",
                                "selected_actor_ids": ["leo"],
                                "base_plate_prompt": "Scene-only background plate.",
                                "edit_passes": [{"pass": 1, "actor_id": "leo", "prompt": "Add only leo from Image 2 into Image 1."}],
                                "video_prompt": "Use the supplied startframe. Leo walks slowly.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = write_movie_i2v_render_plan(project_dir=project)
            plan = json.loads(output.read_text(encoding="utf-8"))

        self.assertIsInstance(plan, list)
        self.assertEqual(1, plan[0]["scene"])
        self.assertEqual("Scene-only background plate.", plan[0]["z_image"]["prompt"])
        self.assertEqual("Use the supplied startframe. Leo walks slowly.", plan[0]["ltx"]["original_style_i2v_prompt"])
        self.assertEqual("leo", plan[0]["movie"]["edit_passes"][0]["actor_id"])
        self.assertEqual(120, plan[0]["frame_count"])

    def test_movie_i2v_edit_adapter_renders_base_edit_passes_then_video(self):
        from feverslop.adapters.movie_i2v_visual import ComfyUIMovieI2VEditVisualAdapter

        class FakeArtifactStore:
            def __init__(self, scenes):
                self.scenes = scenes

            def read_render_plan(self, path):
                return self.scenes

        class FakeBaseBackend:
            def __init__(self):
                self.requests = []

            def render_image(self, request):
                self.requests.append(request)
                path = request.output_dir / f"scene_{request.scene_number:04}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"base")
                return path

        class FakeEditBackend:
            def __init__(self):
                self.calls = []

            def render_edit(self, **kwargs):
                self.calls.append(kwargs)
                path = Path(kwargs["output_dir"]) / f"scene_{kwargs['scene_number']:04}_pass_{kwargs['pass_number']:02}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"edit")
                return path

        class FakeVideoUseCase:
            def __init__(self):
                self.request = None

            def execute(self, request):
                self.request = request
                final = request.output_dir / "final" / "scene_0001.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"clip")
                return [final]

        class FakePostprocessor:
            def write_concat_list(self, rendered, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("\n".join(str(path) for path in rendered), encoding="utf-8")
                return output_path

            def concat_clips(self, concat_list, final_output, *, video_only=False, reencode=True):
                final_output.parent.mkdir(parents=True, exist_ok=True)
                final_output.write_bytes(b"movie")
                return final_output

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            character = project / "movie" / "references" / "actors" / "leo" / "hero.png"
            character.parent.mkdir(parents=True)
            character.write_bytes(b"hero")
            scene = {
                "scene": 1,
                "width": 1280,
                "height": 704,
                "z_image": {"prompt": "base prompt"},
                "movie": {
                    "edit_passes": [
                        {
                            "pass": 1,
                            "actor_id": "leo",
                            "reference_image_path": "movie/references/actors/leo/hero.png",
                            "prompt": "Add only leo.",
                        }
                    ]
                },
            }
            base = FakeBaseBackend()
            edit = FakeEditBackend()
            video = FakeVideoUseCase()
            adapter = ComfyUIMovieI2VEditVisualAdapter(
                base_image_backend=base,
                edit_backend=edit,
                artifact_store=FakeArtifactStore([scene]),
                video_use_case=video,
                workflow_path=Path("base.json"),
                edit_workflow_path=Path("edit.json"),
                i2v_workflow_path=Path("i2v.json"),
                postprocessor=FakePostprocessor(),
            )

            final = adapter.render_movie(project_dir=project, render_plan_path=project / "movie" / "render_plan_i2v.json")

            self.assertEqual(project / "output" / "movie" / f"{project.name}.mp4", final)
            self.assertEqual(project / "output" / "movie" / "storyboard" / "base", base.requests[0].output_dir)
            self.assertEqual(project / "output" / "movie" / "storyboard" / "base" / "scene_0001.png", edit.calls[0]["plate_image"])
            self.assertEqual(character, edit.calls[0]["character_image"])
            self.assertTrue((project / "output" / "movie" / "storyboard" / "final" / "scene_0001.png").exists())
            self.assertEqual(project / "output" / "movie" / "storyboard" / "final", video.request.storyboard_dir)

    def test_movie_i2v_edit_adapter_reports_startframe_step_progress(self):
        from feverslop.adapters.movie_i2v_visual import ComfyUIMovieI2VEditVisualAdapter

        class FakeArtifactStore:
            def read_render_plan(self, _path):
                return [
                    {
                        "scene": 1,
                        "z_image": {"prompt": "base one"},
                        "movie": {
                            "edit_passes": [
                                {"pass": 1, "actor_id": "leo", "reference_image_path": "movie/references/actors/leo/hero.png", "prompt": "Add leo."},
                                {"pass": 2, "actor_id": "morwenna", "reference_image_path": "movie/references/actors/morwenna/hero.png", "prompt": "Add morwenna."},
                            ]
                        },
                    },
                    {
                        "scene": 2,
                        "z_image": {"prompt": "base two"},
                        "movie": {
                            "edit_passes": [
                                {"pass": 1, "actor_id": "leo", "reference_image_path": "movie/references/actors/leo/hero.png", "prompt": "Add leo."},
                            ]
                        },
                    },
                ]

        class FakeBaseBackend:
            def render_image(self, request):
                path = request.output_dir / f"scene_{request.scene_number:04}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"base")
                return path

        class FakeEditBackend:
            def render_edit(self, **kwargs):
                path = Path(kwargs["output_dir"]) / f"scene_{kwargs['scene_number']:04}_pass_{kwargs['pass_number']:02}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"edit")
                return path

        class FakeVideoUseCase:
            def execute(self, request):
                final = request.output_dir / "final" / "scene_0001.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_bytes(b"clip")
                return [final]

        class FakePostprocessor:
            def write_concat_list(self, rendered, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("\n".join(str(path) for path in rendered), encoding="utf-8")
                return output_path

            def concat_clips(self, concat_list, final_output, *, video_only=False, reencode=True):
                final_output.parent.mkdir(parents=True, exist_ok=True)
                final_output.write_bytes(b"movie")
                return final_output

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            for actor_id in ("leo", "morwenna"):
                character = project / "movie" / "references" / "actors" / actor_id / "hero.png"
                character.parent.mkdir(parents=True, exist_ok=True)
                character.write_bytes(b"hero")
            events = []
            adapter = ComfyUIMovieI2VEditVisualAdapter(
                base_image_backend=FakeBaseBackend(),
                edit_backend=FakeEditBackend(),
                artifact_store=FakeArtifactStore(),
                video_use_case=FakeVideoUseCase(),
                workflow_path=Path("base.json"),
                edit_workflow_path=Path("edit.json"),
                i2v_workflow_path=Path("i2v.json"),
                postprocessor=FakePostprocessor(),
            )

            adapter.render_movie(
                project_dir=project,
                render_plan_path=project / "movie" / "render_plan_i2v.json",
                on_startframe_step=lambda event: events.append(event),
            )

        self.assertEqual(
            [
                ("base", 1, 5, 1, ""),
                ("edit", 2, 5, 1, "leo"),
                ("edit", 3, 5, 1, "morwenna"),
                ("base", 4, 5, 2, ""),
                ("edit", 5, 5, 2, "leo"),
            ],
            [(event["kind"], event["completed"], event["total"], event["scene"], event.get("actor_id", "")) for event in events],
        )

    def test_movie_storyboard_page_lists_visual_plan_shots(self):
        from feverslop.tools.movie_storyboard_page import generate_movie_storyboard_page

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            (project / "movie").mkdir(parents=True)
            (project / "output" / "movie" / "storyboard" / "final").mkdir(parents=True)
            (project / "output" / "movie" / "storyboard" / "final" / "scene_0001.png").write_bytes(b"png")
            (project / "movie" / "visual_plan.json").write_text(
                json.dumps(
                    {
                        "shots": [
                            {
                                "scene": 1,
                                "shot_id": "shot_0001",
                                "view_id": "stone_hut_hearth",
                                "selected_actor_ids": ["leo", "morwenna"],
                                "video_prompt": "Leo stops breathing as Morwenna turns.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = generate_movie_storyboard_page(project_dir=project)
            html = output.read_text(encoding="utf-8")

        self.assertIn("stone_hut_hearth", html)
        self.assertIn("leo, morwenna", html)
        self.assertIn("scene_0001.png", html)

    def test_short_story_scaffold_still_generates_screenplay_from_unformatted_idea(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
                artifact_writer=LocalMovieArtifactWriter(),
            ).execute(
                MovieInput(
                    name="Unformatted Idea",
                    source_type="short_story",
                    story_text="A deserter follows a broken compass through a silent battlefield and finds a buried radio.",
                    desired_length=24,
                    mode="scaffold",
                )
            )

            screenplay_artifact = json.loads(result.screenplay_path.read_text(encoding="utf-8"))

            self.assertEqual("short_story", screenplay_artifact["source_type"])
            self.assertGreaterEqual(len(screenplay_artifact["scenes"]), 1)
            self.assertIn("deserter", screenplay_artifact["scenes"][0]["action"].lower())

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
                artifact_writer=LocalMovieArtifactWriter(),
            ).execute(
                MovieInput(
                    name="Door Below",
                    source_type="screenplay",
                    story_text=screenplay,
                    desired_length=12,
                    mode="scaffold",
                )
            )

            bible = json.loads(result.bible_path.read_text())
            manifest = json.loads(result.reference_manifest_path.read_text())

            self.assertEqual(["mara"], [actor["id"] for actor in bible["actors"]])
            self.assertNotIn("main_character", [actor["id"] for actor in bible["actors"]])
            self.assertEqual("Mara", bible["actors"][0]["visual_description"])
            self.assertNotIn("story-defined", bible["actors"][0]["visual_description"])
            self.assertIn("ABANDONED STATION", bible["locations"][0]["visual_description"])
            self.assertNotIn("Mara", bible["locations"][0]["visual_description"])
            self.assertNotIn("story-defined", bible["locations"][0]["visual_description"])
            self.assertEqual("Mara", manifest["actors"][0]["name"])
            self.assertIn("Mara", manifest["actors"][0]["prompt"])
            self.assertEqual("ABANDONED STATION - NIGHT", manifest["locations"][0]["name"])
            self.assertIn("ABANDONED STATION", manifest["locations"][0]["prompt"])

    def test_llm_movie_bible_uses_screenplay_cues_over_generic_placeholders(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import StoryArch

        class BadBibleLLM:
            def complete_prompt(self, *_args, **_kwargs):
                return json.dumps(
                    {
                        "actors": [
                            {
                                "id": "technikerin",
                                "name": "Technikerin",
                                "role": "character",
                                "visual_description": "Technikerin, story-defined cinematic character with consistent face, hair, body shape, wardrobe, and posture",
                            },
                            {
                                "id": "main_character",
                                "name": "Main Character",
                                "role": "character",
                                "visual_description": "Main Character, story-defined cinematic character with consistent face, hair, body shape, wardrobe, and posture",
                            },
                        ],
                        "locations": [
                            {
                                "id": "primary_location",
                                "name": "Primary Location",
                                "visual_description": "Primary Location, story-defined cinematic location with consistent production design, geography, lighting, and atmosphere",
                            }
                        ],
                    }
                )

        story_arch = StoryArch(
            title="Station",
            premise="A technician hears a knocking door.",
            beats=("A technician hears a knocking door in an orbital corridor.",),
        )
        screenplay = "INT. KORRIDOR DER STATION - NACHT\n\nTECHNIKERIN\nWer ist da drin?\n\nHinter einer Metalltuer klopft etwas langsam von innen."

        bible = LLMMoviePlanner(BadBibleLLM()).generate_movie_bible(
            title="Station",
            source_type="screenplay",
            story_text=screenplay,
            desired_length=20,
            story_arch=story_arch,
            config={},
        )

        self.assertEqual(["technikerin"], [actor.id for actor in bible.actors])
        self.assertEqual("Technikerin", bible.actors[0].visual_description)
        self.assertEqual(["korridor_der_station_nacht"], [location.id for location in bible.locations])
        self.assertIn("KORRIDOR DER STATION - NACHT", bible.locations[0].visual_description)
        self.assertIn("Metalltuer klopft", bible.locations[0].visual_description)

    def test_movie_bible_loader_cleans_generic_location_placeholders(self):
        from feverslop.application.movie import movie_bible_from_dict

        bible = movie_bible_from_dict(
            {
                "title": "Station",
                "premise": "A technician hears a knocking door.",
                "story_arch": {"title": "Station", "premise": "A technician hears a knocking door.", "beats": []},
                "actors": [{"id": "technikerin", "name": "Technikerin", "visual_description": "Technikerin"}],
                "locations": [
                    {
                        "id": "korridor_der_station_nacht",
                        "name": "KORRIDOR DER STATION - NACHT",
                        "visual_description": "KORRIDOR DER STATION - NACHT, story-defined cinematic location with consistent production design, geography, lighting, and atmosphere",
                    }
                ],
            }
        )

        self.assertEqual("KORRIDOR DER STATION - NACHT", bible.locations[0].visual_description)

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

    def test_movie_planner_does_not_repeat_screenplay_dialogue_when_splitting_long_scene(self):
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        screenplay = """
        EXT. ARCTIC STATION - NIGHT

        Lena reaches the buried station door. Red warning lights pulse through snow. She raises the radio to her ear. The storm swallows the horizon.

        TECHNICIAN: Is anyone there?
        """
        planner = DeterministicMoviePlanner()
        story_arch = planner.generate_story_arch(
            title="Ice",
            source_type="screenplay",
            story_text=screenplay,
            desired_length=36,
        )

        shots = planner.plan_shots(
            story_arch=story_arch,
            desired_length=36,
            width=640,
            height=480,
            min_duration=4,
            max_duration=10,
        )

        self.assertGreater(len(shots), 1)
        self.assertEqual(1, sum(1 for shot in shots if "Is anyone there" in shot.dialogue))
        self.assertEqual("", shots[1].dialogue)
        self.assertGreater(len({shot.action for shot in shots}), 1)
        self.assertNotEqual(shots[0].action, shots[1].action)

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

    def test_movie_workflow_patcher_replaces_ltx_audio_loader_chain_with_empty_latent(self):
        from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

        workflow = {
            "1": {"class_type": "LoadAudio", "_meta": {"title": "#LOAD_AUDIO"}, "inputs": {"audio": "song.wav"}},
            "2": {"class_type": "TrimAudioDuration", "_meta": {"title": "#TRIM_AUDIO"}, "inputs": {"audio": ["1", 0], "duration": 3.0}},
            "5": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "ltxv-audio.safetensors"}},
            "22": {"class_type": "PrimitiveInt", "_meta": {"title": "#FRAMES"}, "inputs": {"value": 224}},
            "25": {"class_type": "PrimitiveInt", "_meta": {"title": "#FRAMERATE"}, "inputs": {"value": 24}},
            "20": {"class_type": "VideoLatent", "inputs": {}},
            "30": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["20", 0], "audio_latent": ["36", 0]}},
            "34": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["2", 0], "audio_vae": ["5", 0]}},
            "35": {"class_type": "SolidMask", "inputs": {"value": 0}},
            "36": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["34", 0], "mask": ["35", 0]}},
            "40": {"class_type": "Sampler", "inputs": {"latent_image": ["30", 0]}},
            "52": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["40", 1]}},
            "56": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["40", 0], "audio": ["52", 0]}},
        }

        patched = MovieWorkflowPatcher().strip_audio_inputs(workflow)

        self.assertNotIn("1", patched)
        self.assertNotIn("2", patched)
        self.assertNotIn("34", patched)
        self.assertNotIn("35", patched)
        self.assertNotIn("36", patched)
        self.assertIn("30", patched)
        replacement_id = str(patched["30"]["inputs"]["audio_latent"][0])
        self.assertEqual("LTXVEmptyLatentAudio", patched[replacement_id]["class_type"])
        self.assertEqual(["22", 0], patched[replacement_id]["inputs"]["frames_number"])
        self.assertEqual(["25", 0], patched[replacement_id]["inputs"]["frame_rate"])
        self.assertEqual(["5", 0], patched[replacement_id]["inputs"]["audio_vae"])
        self.assertEqual(["30", 0], patched["40"]["inputs"]["latent_image"])
        self.assertEqual(["52", 0], patched["56"]["inputs"]["audio"])

    def test_movie_workflow_patcher_repairs_empty_audio_latent_vae_link(self):
        from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

        workflow = {
            "2": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},
            "5": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "LTX23_audio_vae_bf16.safetensors"}},
            "22": {"class_type": "PrimitiveInt", "_meta": {"title": "#FRAMES"}, "inputs": {"value": 224}},
            "25": {"class_type": "PrimitiveInt", "_meta": {"title": "#FRAMERATE"}, "inputs": {"value": 24}},
            "30": {"class_type": "LTXVConcatAVLatent", "inputs": {"audio_latent": ["61", 0]}},
            "52": {"class_type": "LTXVAudioVAEDecode", "inputs": {"audio_vae": ["5", 0], "samples": ["51", 1]}},
            "61": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": ["22", 0],
                    "frame_rate": ["25", 0],
                    "batch_size": 1,
                    "audio_vae": ["2", 0],
                },
            },
        }

        patched = MovieWorkflowPatcher().strip_audio_inputs(workflow)

        self.assertEqual(["5", 0], patched["61"]["inputs"]["audio_vae"])

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

    def test_comfyui_movie_ingredients_adapter_disables_audio_upload(self):
        from feverslop.adapters.comfyui_ingredients_video_backend import ComfyUIIngredientsVideoRenderBackend
        from feverslop.adapters.movie_ingredients_visual import ComfyUIMovieIngredientsVisualAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            render_plan_path = temp / "movie" / "render_plan_ingredients.json"
            render_plan_path.parent.mkdir(parents=True, exist_ok=True)
            render_plan_path.write_text(
                json.dumps({
                    "title": "Ingredient Movie",
                    "fps": 24,
                    "scenes": [{
                        "scene": 1,
                        "duration_seconds": 2.0,
                        "ingredients_scene_sheet": str(sheet),
                        "ltx": {"ingredients_target_prompt": "target"},
                    }],
                }),
                encoding="utf-8",
            )
            workflow_path = temp / "ingredients.json"
            workflow_path.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#INGREDIENTS"}, "class_type": "LoadImage"},
                    "2": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT_POSITIVE"}, "class_type": "CLIPTextEncode"},
                    "3": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                    "4": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                    "5": {"inputs": {"value": 49}, "_meta": {"title": "#FRAMES"}},
                    "6": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMERATE"}},
                }),
                encoding="utf-8",
            )
            queue = FakeMovieRenderQueue()
            asset_uploader = NativeAudioAssetUploader()
            postprocessor = FakeMoviePostprocessor()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=object(),
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                asset_uploader=asset_uploader,
                render_queue=queue,
                postprocessor=postprocessor,
                postprocess=False,
            )

            final = ComfyUIMovieIngredientsVisualAdapter(backend=backend).render_movie(
                project_dir=temp,
                render_plan_path=render_plan_path,
            )

            self.assertEqual(temp / "output" / "movie" / "ingredient-movie.mp4", final)
            self.assertEqual([], asset_uploader.audio_calls)

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
                    "duration_seconds": 4,
                    "shots": [
                        _movie_shot(temp),
                        {**_movie_shot(temp), "shot_id": "shot_0002", "description": "The door opens wider."},
                    ],
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
            self.assertEqual([1, 2], [call[1] for call in queue.calls])
            self.assertEqual(
                [temp / "output" / "movie" / "ltx_msr" / "scene_0001.mp4", temp / "output" / "movie" / "ltx_msr" / "scene_0002.mp4"],
                postprocessor.concat_lists[0][0],
            )
            self.assertEqual([(postprocessor.concat_lists[0][1], final, False, True)], postprocessor.concat_calls)
            self.assertTrue((temp / "output" / "movie" / "ltx_msr_debug" / "scene_0001_workflow.json").exists())
            self.assertTrue((temp / "output" / "movie" / "ltx_msr_debug" / "scene_0002_workflow.json").exists())

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

    def test_comfyui_movie_adapter_rerenders_selected_scenes_and_rebuilds_final(self):
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
                    "shots": [
                        _movie_shot(temp),
                        {**_movie_shot(temp), "shot_id": "shot_0002", "description": "The door answers."},
                    ],
                }),
                encoding="utf-8",
            )
            existing_scene = temp / "output" / "movie" / "ltx_msr" / "scene_0001.mp4"
            existing_scene.parent.mkdir(parents=True)
            existing_scene.write_bytes(b"old scene 1")
            queue = FakeMovieRenderQueue()
            postprocessor = FakeMoviePostprocessor()

            ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                render_queue=queue,
                asset_uploader=NativeAudioAssetUploader(),
                postprocessor=postprocessor,
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path, selected_scenes=[2])

            self.assertEqual([2], [call[1] for call in queue.calls])
            self.assertEqual(
                [existing_scene, temp / "output" / "movie" / "ltx_msr" / "scene_0002.mp4"],
                postprocessor.concat_lists[0][0],
            )

    def test_comfyui_movie_adapter_uses_last_frame_as_startframe_for_continuous_transition(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "movie_msr.json"
            workflow_path.write_text(json.dumps(_movie_msr_workflow()), encoding="utf-8")
            i2v_workflow_path = temp / "movie_msr_i2v.json"
            i2v_workflow = {
                **_movie_msr_workflow(),
                "11": {"class_type": "LoadImage", "_meta": {"title": "#STARTFRAME"}, "inputs": {"image": ""}},
            }
            i2v_workflow_path.write_text(json.dumps(i2v_workflow), encoding="utf-8")
            render_plan_path = temp / "movie" / "render_plan.json"
            render_plan_path.parent.mkdir()
            render_plan_path.write_text(
                json.dumps({
                    "title": "Door Below",
                    "resolution": {"width": 1280, "height": 704},
                    "shots": [
                        _movie_shot(temp),
                        {
                            **_movie_shot(temp),
                            "shot_id": "shot_0002",
                            "description": "The door opens wider.",
                            "transition_from_previous": "continuous",
                        },
                    ],
                }),
                encoding="utf-8",
            )
            queue = FakeMovieRenderQueue()
            asset_uploader = NativeAudioAssetUploader()
            postprocessor = FakeMoviePostprocessor()

            ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                i2v_workflow_path=i2v_workflow_path,
                render_queue=queue,
                asset_uploader=asset_uploader,
                postprocessor=postprocessor,
                continuity_keyframes="last-to-start",
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path)

            expected_startframe = temp / "output" / "movie" / "keyframes" / "scene_0001_to_0002_start.png"
            [(source, temporary_frame)] = postprocessor.last_frame_extracts
            self.assertEqual(
                temp / "output" / "movie" / "ltx_msr" / "scene_0001.mp4",
                source,
            )
            self.assertEqual(expected_startframe.parent, temporary_frame.parent)
            self.assertTrue(
                temporary_frame.name.startswith(
                    ".scene_0001_to_0002_start-"
                )
            )
            self.assertTrue(expected_startframe.is_file())
            scene_1_workflow = queue.calls[0][0]
            self.assertFalse(any("#STARTFRAME" == node.get("_meta", {}).get("title") for node in scene_1_workflow.values()))
            scene_2_workflow = queue.calls[1][0]
            self.assertEqual("scene_0001_to_0002_start.png", scene_2_workflow["11"]["inputs"]["image"])

    def test_comfyui_movie_adapter_builds_i2v_continuity_handoff_without_front_preroll(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "movie_msr.json"
            workflow_path.write_text(json.dumps(_movie_msr_workflow()), encoding="utf-8")
            i2v_workflow_path = temp / "movie_msr_i2v.json"
            i2v_workflow = {
                **_movie_msr_workflow(),
                "4": {
                    "class_type": "PromptRelayEncode",
                    "_meta": {"title": "#PROMPT_RELAY"},
                    "inputs": {"global_prompt": "", "local_prompts": "", "segment_lengths": ""},
                },
                "11": {"class_type": "LoadImage", "_meta": {"title": "#STARTFRAME"}, "inputs": {"image": ""}},
                "12": {
                    "class_type": "LTXAddVideoICLoRAGuide",
                    "_meta": {"title": "#MSR_GUIDE"},
                    "inputs": {"frame_idx": 0, "strength": 1},
                },
            }
            i2v_workflow_path.write_text(json.dumps(i2v_workflow), encoding="utf-8")
            render_plan_path = temp / "movie" / "render_plan.json"
            render_plan_path.parent.mkdir()
            render_plan_path.write_text(
                json.dumps({
                    "title": "Door Below",
                    "resolution": {"width": 1280, "height": 704},
                    "shots": [
                        {
                            **_movie_shot(temp),
                            "shot_id": "shot_0001",
                            "description": "Mara waits at the archive threshold.",
                            "ltx": {
                                "msr_global_prompt": "Reference image 1: Mara. Background reference: Archive.",
                                "msr_prompt_relay": [
                                    {
                                        "frame_start": 0,
                                        "frame_end": 47,
                                        "prompt": "Mara waits at the archive threshold as fog drifts outward.",
                                    }
                                ],
                            },
                        },
                        {
                            **_movie_shot(temp),
                            "shot_id": "shot_0002",
                            "description": "Mara steps through the archive door.",
                            "transition_from_previous": "continuous",
                            "ltx": {
                                "msr_global_prompt": "Reference image 1: Mara. Background reference: Archive.",
                                "msr_prompt_relay": [
                                    {
                                        "frame_start": 0,
                                        "frame_end": 47,
                                        "prompt": "Mara steps through the archive door.",
                                    }
                                ],
                            },
                        },
                    ],
                }),
                encoding="utf-8",
            )
            queue = FakeMovieRenderQueue()
            postprocessor = FakeMoviePostprocessor()

            ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                i2v_workflow_path=i2v_workflow_path,
                render_queue=queue,
                asset_uploader=NativeAudioAssetUploader(),
                postprocessor=postprocessor,
                continuity_keyframes="last-to-start",
                continuity_handoff_factory=_continuity_handoff_factory,
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path)

            self.assertEqual(0, postprocessor.trim_specs[1].trim_front_frames)
            scene_2_workflow = queue.calls[1][0]
            relay_inputs = scene_2_workflow["4"]["inputs"]
            self.assertEqual(18, scene_2_workflow["12"]["inputs"]["frame_idx"])
            self.assertEqual(17, scene_2_workflow["3"]["inputs"]["frame_count"])
            self.assertTrue(relay_inputs["segment_lengths"].startswith("18,"))
            self.assertTrue(relay_inputs["local_prompts"].startswith("Mara waits at the archive threshold"))
            self.assertIn("\n|Mara steps through the archive door.", relay_inputs["local_prompts"])

    def test_comfyui_movie_adapter_does_not_extract_startframe_when_feature_is_off(self):
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
                    "shots": [
                        _movie_shot(temp),
                        {
                            **_movie_shot(temp),
                            "shot_id": "shot_0002",
                            "description": "The door opens wider.",
                            "transition_from_previous": "continuous",
                        },
                    ],
                }),
                encoding="utf-8",
            )
            queue = FakeMovieRenderQueue()
            postprocessor = FakeMoviePostprocessor()

            ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                render_queue=queue,
                asset_uploader=NativeAudioAssetUploader(),
                postprocessor=postprocessor,
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path)

            self.assertEqual([], postprocessor.last_frame_extracts)
            self.assertFalse(any("#STARTFRAME" == node.get("_meta", {}).get("title") for node in queue.calls[1][0].values()))

    def test_comfyui_movie_adapter_requires_previous_clip_for_selected_continuous_scene(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "movie_msr_i2v.json"
            workflow = {
                **_movie_msr_workflow(),
                "11": {"class_type": "LoadImage", "_meta": {"title": "#STARTFRAME"}, "inputs": {"image": ""}},
            }
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            render_plan_path = temp / "movie" / "render_plan.json"
            render_plan_path.parent.mkdir()
            render_plan_path.write_text(
                json.dumps({
                    "title": "Door Below",
                    "resolution": {"width": 1280, "height": 704},
                    "shots": [
                        _movie_shot(temp),
                        {
                            **_movie_shot(temp),
                            "shot_id": "shot_0002",
                            "description": "The door answers.",
                            "transition_from_previous": "continuous",
                        },
                    ],
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "previous movie scene clip"):
                ComfyUIMovieVisualAdapter(
                    client=object(),
                    workflow_path=workflow_path,
                    render_queue=FakeMovieRenderQueue(),
                    asset_uploader=NativeAudioAssetUploader(),
                    postprocessor=FakeMoviePostprocessor(),
                    continuity_keyframes="last-to-start",
                    continuity_handoff_factory=_continuity_handoff_factory,
                ).render_movie(project_dir=temp, render_plan_path=render_plan_path, selected_scenes=[2])

    def test_selected_movie_predecessor_can_be_rerendered_before_handoff(self):
        from feverslop.adapters.movie_visual import (
            _validate_selected_continuity_dependencies,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            scenes = [
                {
                    "scene": 1,
                    "references": {
                        "actor_msr_paths": [actor.as_posix()],
                        "location_msr_path": location.as_posix(),
                    },
                    "transition_from_previous": "cut",
                },
                {
                    "scene": 2,
                    "references": {
                        "actor_msr_paths": [actor.as_posix()],
                        "location_msr_path": location.as_posix(),
                    },
                    "transition_from_previous": "continuous",
                },
            ]

            _validate_selected_continuity_dependencies(
                scenes,
                output_dir=temp,
                selected={1, 2},
            )

    def test_comfyui_movie_adapter_concat_only_reuses_existing_scene_clips(self):
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
                    "shots": [
                        _movie_shot(temp),
                        {**_movie_shot(temp), "shot_id": "shot_0002", "description": "The door answers."},
                    ],
                }),
                encoding="utf-8",
            )
            scene_1 = temp / "output" / "movie" / "ltx_msr" / "scene_0001.mp4"
            scene_2 = temp / "output" / "movie" / "ltx_msr" / "scene_0002.mp4"
            scene_1.parent.mkdir(parents=True)
            scene_1.write_bytes(b"scene 1")
            scene_2.write_bytes(b"scene 2")
            queue = FakeMovieRenderQueue()
            postprocessor = FakeMoviePostprocessor()

            ComfyUIMovieVisualAdapter(
                client=object(),
                workflow_path=workflow_path,
                render_queue=queue,
                asset_uploader=NativeAudioAssetUploader(),
                postprocessor=postprocessor,
            ).render_movie(project_dir=temp, render_plan_path=render_plan_path, concat_only=True)

            self.assertEqual([], queue.calls)
            self.assertEqual([scene_1, scene_2], postprocessor.concat_lists[0][0])

    def test_comfyui_movie_adapter_concat_only_fails_when_scene_clip_is_missing(self):
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
                    "shots": [_movie_shot(temp)],
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing rendered movie scene clip"):
                ComfyUIMovieVisualAdapter(
                    client=object(),
                    workflow_path=workflow_path,
                    render_queue=FakeMovieRenderQueue(),
                    asset_uploader=NativeAudioAssetUploader(),
                    postprocessor=FakeMoviePostprocessor(),
                ).render_movie(project_dir=temp, render_plan_path=render_plan_path, concat_only=True)

    def test_movie_job_uses_comfyui_adapter_when_configured(self):
        from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
        from feverslop.studio.job_service import build_movie_visual_adapter

        adapter = build_movie_visual_adapter(Path("project"), Path("workflow.json"), movie_config={"render_backend": "comfyui"})

        self.assertIsInstance(adapter, ComfyUIMovieVisualAdapter)

    def test_movie_runtime_config_requires_i2v_workflow_for_last_frame_keyframes(self):
        from feverslop.studio.job_service import movie_runtime_config

        with self.assertRaisesRegex(ValueError, "msr-i2v-startframe"):
            movie_runtime_config({"continuity_keyframes": "last-to-start"})

        config = movie_runtime_config({
            "movie_video_workflow": "msr-i2v-startframe",
            "continuity_keyframes": "last-to-start",
        })

        self.assertEqual("last-to-start", config["continuity_keyframes"])

    def test_movie_runtime_config_uses_v6_ingredients_workflow_by_default(self):
        from feverslop.studio.job_service import movie_runtime_config

        config = movie_runtime_config({"movie_video_workflow": "ingredients"})

        self.assertEqual(
            "workflows/video_ltxv_ingredients_2stage_v6.json",
            config["ingredients_workflow"],
        )

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

    def test_movie_workflow_patcher_is_available_without_studio_imports(self):
        from feverslop.composition.movie_workflow import patch_movie_msr_workflow

        output = patch_movie_msr_workflow()

        self.assertIsInstance(output, dict)
        self.assertFalse(any(node.get("class_type") in {"LoadAudio", "TrimAudioDuration"} for node in output.values()))

    def test_movie_workflow_patcher_patches_msr_i2v_startframe_anchor(self):
        from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

        workflow = {
            "1": {"class_type": "LoadImage", "_meta": {"title": "#MSR_ACTOR_1"}, "inputs": {"image": ""}},
            "2": {"class_type": "LoadImage", "_meta": {"title": "#MSR_BACKGROUND"}, "inputs": {"image": ""}},
            "3": {"class_type": "LTXICLoRALoaderModelOnly", "_meta": {"title": "#MSR_LORA"}, "inputs": {"lora_name": "old.safetensors"}},
            "4": {"class_type": "LiconMSR", "_meta": {"title": "#MSR_FRAME_COUNT"}, "inputs": {"frame_count": 17}},
            "5": {"class_type": "PromptRelayEncode", "_meta": {"title": "#PROMPT_RELAY"}, "inputs": {"global_prompt": "", "local_prompts": ""}},
            "6": {"class_type": "LoadImage", "_meta": {"title": "#STARTFRAME"}, "inputs": {"image": ""}},
        }

        patched = MovieWorkflowPatcher().patch_msr_i2v_startframe(
            workflow,
            startframe_image_name="scene_0001_start.png",
            msr_lora_name="LTX-2.3-Licon-MSR-V1.safetensors",
            msr_frame_count=25,
        )

        self.assertEqual("scene_0001_start.png", patched["6"]["inputs"]["image"])
        self.assertEqual("LTX-2.3-Licon-MSR-V1.safetensors", patched["3"]["inputs"]["lora_name"])
        self.assertEqual(25, patched["4"]["inputs"]["frame_count"])

    def test_movie_workflow_patcher_requires_startframe_anchor_for_msr_i2v(self):
        from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

        with self.assertRaisesRegex(ValueError, "#STARTFRAME"):
            MovieWorkflowPatcher().patch_msr_i2v_startframe(
                {"1": {"class_type": "LoadImage", "_meta": {"title": "#MSR_ACTOR_1"}, "inputs": {"image": ""}}},
                startframe_image_name="scene_0001_start.png",
            )

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
        self.assertIn("Write every non-dialogue prose field in English", llm.calls[1][1])
        self.assertIn("Only the dialogue field may use", llm.calls[1][1])

    def test_llm_movie_planner_prompts_require_english_visual_prose_except_dialogue(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import StoryArch

        class CapturingLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, system_prompt=None):
                self.calls.append(prompt)
                if "movie bible" in prompt.lower():
                    return json.dumps({"actors": [], "locations": []})
                if "dramaturgical story design" in prompt.lower():
                    return json.dumps({})
                if "canonical structured screenplay" in prompt.lower():
                    return json.dumps({})
                if "narrative memory plan" in prompt.lower():
                    return json.dumps({})
                if "continuity plan" in prompt.lower():
                    return json.dumps({})
                if "render plan" in prompt.lower():
                    return json.dumps({"shots": []})
                return json.dumps({"premise": "A technician enters a lighthouse.", "beats": ["The light fails."]})

        planner = LLMMoviePlanner(CapturingLLM())
        story_arch = StoryArch(title="Light", premise="A technician enters a lighthouse.", beats=("The light fails.",))
        bible = planner.generate_movie_bible(
            title="Light",
            source_type="short_story",
            story_text="Eine Technikerin repariert einen Leuchtturm.",
            desired_length=30,
            story_arch=story_arch,
            config={"dialogue_language": "German"},
        )
        planner.generate_movie_story_design(
            title="Light",
            source_type="short_story",
            story_text="Eine Technikerin repariert einen Leuchtturm.",
            desired_length=30,
            bible=bible,
            story_arch=story_arch,
            config={"dialogue_language": "German"},
        )
        planner.generate_movie_screenplay(
            title="Light",
            source_type="short_story",
            story_text="Eine Technikerin repariert einen Leuchtturm.",
            desired_length=30,
            bible=bible,
            story_arch=story_arch,
            story_design=None,
            config={"dialogue_language": "German"},
        )
        planner.plan_shots_from_bible(bible=bible, screenplay=None, desired_length=30, width=640, height=480)

        prompts = "\n".join(planner.llm.calls)
        self.assertIn("Name every actor_ids entry in action", prompts)
        self.assertIn("spatial relationship", prompts)
        self.assertIn("collective noun", prompts)

        prompts = "\n\n".join(planner.llm.calls)
        self.assertIn("actor visual_description", prompts)
        self.assertIn("Write all story-design prose in English", prompts)
        self.assertIn("Write every non-dialogue screenplay field in English", prompts)
        self.assertIn("Write every non-dialogue prose field in English", prompts)
        self.assertIn("Only dialogue may use German", prompts)

    def test_shot_plan_prompt_INCLUDES_screenplay(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieBible, MovieActor, MovieLocation, MovieContinuityRule, StoryArch, MovieScreenplayArtifact, MovieScreenplayScene

        class CapturingLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, system_prompt=None):
                self.calls.append(prompt)
                return json.dumps({
                    "shots": [{
                        "shot_id": "shot_0001",
                        "description": "test",
                        "duration_seconds": 30,
                        "camera": "static",
                        "action": "test",
                        "acting": "calm",
                        "location": "Room",
                        "dialogue": "A: hello",
                        "actor_ids": ["actor_a"],
                        "location_id": "room",
                        "continuity_notes": "",
                        "transition_from_previous": "cut",
                    }],
                })

        llm = CapturingLLM()
        planner = LLMMoviePlanner(llm)
        bible = MovieBible(
            title="Test",
            premise="A test",
            story_arch=StoryArch(title="Test", premise="A test", beats=("beat 1",)),
            actors=(MovieActor(id="actor_a", name="A", role="lead", visual_description="A"),),
            locations=(MovieLocation(id="room", name="Room", visual_description="Room"),),
            continuity=(MovieContinuityRule(id="vis", description="visual"),),
            style_constraints=(),
            runtime_constraints={},
        )
        screenplay = MovieScreenplayArtifact(
            title="Test",
            source_type="short_story",
            dialogue_language="English",
            scenes=(
                MovieScreenplayScene(
                    scene_id="scene_0001",
                    heading="INT. ROOM - DAY",
                    summary="A says hello",
                    action="A stands in the room",
                    dialogue="A: Hello, world.",
                    actor_ids=("actor_a",),
                    location_id="room",
                ),
            ),
        )
        planner.plan_shots_from_bible(
            bible=bible,
            screenplay=screenplay,
            desired_length=30,
            width=640,
            height=480,
        )
        prompt = "\n".join(llm.calls)
        self.assertIn("SCREENPLAY", prompt)
        self.assertIn("Hello, world", prompt)
        self.assertIn("Map screenplay scene dialogue", prompt)

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
            self.assertEqual("movie/references/actors/main_character/views/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertEqual("movie/references/locations/primary_location/views/hero.png", manifest["locations"][0]["msr_sheet_path"])
            self.assertGreaterEqual(len(backend.requests), 2)
            self.assertTrue(
                backend.requests[0].prompt.startswith("Full-body cinematic character reference sheet for Mara.")
            )
            self.assertEqual("gothic protagonist", manifest["actors"][0]["visual_description"])
            self.assertEqual(1, backend.requests[0].prompt.count("Full-body cinematic character reference sheet"))
            self.assertIn("1st panel head-and-shoulders closeup", backend.requests[0].prompt)
            self.assertIn("plain white seamless studio background", backend.requests[0].prompt)
            self.assertIn("no extra characters", backend.requests[0].prompt)

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

    def test_movie_reference_sync_rebuilds_mixed_actor_prompts_before_rendering(self):
        from feverslop.studio.job_service import sync_movie_manifest_with_render_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "movie" / "references").mkdir(parents=True)
            (root / "movie" / "render_plan.json").write_text(
                json.dumps({
                    "shots": [
                        {
                            "scene": 1,
                            "description": "The goat demon lunges from the shadows.",
                            "action": "The goat demon bellows, shaking the dreamscape.",
                            "expression": "Menacing, primal",
                            "location": "Dreamscape",
                            "reference_ids": {"actors": ["the_goat_demon"], "location": "dreamscape"},
                        },
                        {
                            "scene": 2,
                            "description": "A low-angle shot of the goat demon, towering and muscular, with massive curved horns.",
                            "action": "The goat demon bellows, shaking the dreamscape.",
                            "expression": "Menacing, primal",
                            "location": "Dreamscape",
                            "reference_ids": {"actors": ["the_goat_demon"], "location": "dreamscape"},
                        },
                        {
                            "scene": 3,
                            "description": "A split-screen composition shows the succubus and demon closing in on the man.",
                            "action": "The two entities converge on the man from opposite sides.",
                            "expression": "Seductive vs. Ferocious",
                            "location": "Dreamscape",
                            "reference_ids": {"actors": ["the_succubus", "the_goat_demon"], "location": "dreamscape"},
                        },
                    ]
                }),
                encoding="utf-8",
            )
            mixed_prompt = (
                "Full-body cinematic character reference sheet for The Goat Demon. "
                "Visual identity inferred from scenes: The goat demon lunges; "
                "A split-screen composition shows the succubus and demon closing in on the man. "
                "Consistent face, hair, body shape, wardrobe, posture, neutral expression, clean studio background, no text."
            )
            (root / "movie" / "references" / "manifest.json").write_text(
                json.dumps({
                    "actors": [
                        {
                            "id": "the_goat_demon",
                            "name": "The Goat Demon",
                            "prompt": mixed_prompt,
                            "visual_description": mixed_prompt,
                            "image_prompt": mixed_prompt,
                            "msr_sheet_path": "movie/references/actors/the_goat_demon/msr_sheet.png",
                        }
                    ],
                    "locations": [{"id": "dreamscape", "name": "Dreamscape", "prompt": "Dreamscape", "msr_sheet_path": "x.png"}],
                }),
                encoding="utf-8",
            )

            sync_movie_manifest_with_render_plan(root)

            manifest = json.loads((root / "movie" / "references" / "manifest.json").read_text())
            goat = next(actor for actor in manifest["actors"] if actor["id"] == "the_goat_demon")
            self.assertIn("massive curved horns", goat["visual_description"])
            self.assertNotIn("Four vertical panels", goat["visual_description"])
            self.assertNotIn("split-screen", goat["prompt"].lower())
            self.assertNotIn("succubus", goat["prompt"].lower())
            self.assertNotIn("jump cut", goat["prompt"].lower())
            self.assertNotIn("lunges", goat["prompt"].lower())
            self.assertNotIn("bellows", goat["prompt"].lower())
            self.assertNotIn("visual identity", goat["prompt"].lower())
            self.assertIn("Four vertical panels in one image", goat["prompt"])
            self.assertEqual("", goat["msr_sheet_path"])

    def test_api_creates_movie_project_with_scaffold_artifacts(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_planner = FakeMoviePlanner()
            with patch("feverslop.studio.project_repository.build_movie_planner", return_value=fake_planner):
                client = NativeStudioHarness(temp_dir)

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
                        "dialogue_language": "German",
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
            self.assertEqual("German", config["dialogue_language"])
            self.assertIn("steering", config)
            self.assertEqual("llm", project["metadata"]["movie"]["planner_backend"])
            self.assertEqual("German", project["metadata"]["movie"]["dialogue_language"])
            self.assertEqual("comfyui", project["metadata"]["movie"]["reference_backend"])
            self.assertEqual("comfyui", project["metadata"]["movie"]["render_backend"])
            self.assertEqual(1, len(fake_planner.story_calls))
            self.assertEqual(1, len(fake_planner.shot_calls))

    def test_api_rejects_invalid_screenplay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)

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
            client = NativeStudioHarness(temp_dir)
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
            self.assertEqual("movie/references/actors/main_character/views/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertEqual("movie/references/locations/primary_location/views/hero.png", manifest["locations"][0]["msr_sheet_path"])
            self.assertEqual("local", manifest["generator_backend"])
            self.assertFalse((Path(temp_dir) / "door-below" / "movie" / "workflows").exists())

    def test_api_movie_full_auto_i2v_edit_writes_storyboard_and_job_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)
            created = client.post(
                "/api/projects",
                json={
                    "project_type": "movie",
                    "name": "Witch Story",
                    "source_type": "screenplay",
                    "story_text": "EXT. BLACKWOOD FOREST - DAY\n\nMORWENNA\nTu tardes.",
                    "desired_length": 12,
                    "movie_mode": "full_auto",
                    "movie_planner_backend": "deterministic",
                    "movie_reference_backend": "local",
                    "movie_render_backend": "local",
                    "movie_video_workflow": "i2v-edit",
                },
            )
            self.assertEqual(200, created.status_code, created.text)

            job = client.post("/api/projects/witch-story/jobs", json={"action": "movie-full-auto"})

            self.assertEqual(200, job.status_code, job.text)
            job_id = job.json()["id"]
            for _ in range(50):
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] == "succeeded":
                    break
                time.sleep(0.01)

            project_dir = Path(temp_dir) / "witch-story"
            metadata = json.loads((project_dir / ".studio" / "project.json").read_text())

            self.assertEqual("succeeded", status["status"])
            self.assertEqual("workflows/image_edit_flux2_klein_2ref_v1.json", metadata["movie"]["edit_workflow"])
            self.assertTrue((project_dir / "movie" / "visual_plan.json").exists())
            self.assertTrue((project_dir / "movie" / "render_plan_i2v.json").exists())
            self.assertTrue((project_dir / "output" / "movie" / "storyboard" / "index.html").exists())
            self.assertTrue((project_dir / "output" / "movie" / "witch-story.mp4").exists())
            logs = "\n".join(status["logs"])
            self.assertIn("Movie visual plan", logs)
            self.assertIn("Movie I2V render plan", logs)
            self.assertIn("Storyboard review page", logs)

    def test_api_movie_full_auto_startframe_director_writes_contracts_and_job_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)
            created = client.post(
                "/api/projects",
                json={
                    "project_type": "movie",
                    "name": "Director Story",
                    "source_type": "screenplay",
                    "story_text": "EXT. ARCHIVE DOOR - DAY\n\nMARA\nIt opens.",
                    "desired_length": 12,
                    "movie_mode": "full_auto",
                    "movie_planner_backend": "deterministic",
                    "movie_reference_backend": "local",
                    "movie_render_backend": "local",
                    "movie_video_workflow": "startframe-director",
                },
            )
            self.assertEqual(200, created.status_code, created.text)

            job = client.post("/api/projects/director-story/jobs", json={"action": "movie-full-auto"})

            self.assertEqual(200, job.status_code, job.text)
            job_id = job.json()["id"]
            for _ in range(50):
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] == "succeeded":
                    break
                time.sleep(0.01)

            project_dir = Path(temp_dir) / "director-story"
            self.assertEqual("succeeded", status["status"])
            self.assertTrue((project_dir / "movie" / "identity_ledger.json").exists())
            self.assertTrue((project_dir / "movie" / "startframe_plan.json").exists())
            self.assertTrue((project_dir / "movie" / "startframe_director_prompts.json").exists())
            self.assertTrue((project_dir / "movie" / "startframe_validation.json").exists())
            self.assertTrue((project_dir / "output" / "movie" / "storyboard" / "final" / "scene_0001.png").exists())
            self.assertTrue((project_dir / "output" / "movie" / "director-story.mp4").exists())
            logs = "\n".join(status["logs"])
            self.assertIn("Movie identity ledger", logs)
            self.assertIn("Movie startframe plan", logs)
            self.assertIn("Movie director prompts", logs)

    def test_movie_full_auto_regenerates_missing_planning_artifacts_for_legacy_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)
            created = client.post(
                "/api/projects",
                json={
                    "project_type": "movie",
                    "name": "Legacy Door",
                    "source_type": "short_story",
                    "story_text": "A locksmith finds a glowing door below an abandoned station.",
                    "desired_length": 30,
                    "movie_mode": "scaffold",
                    "movie_planner_backend": "deterministic",
                    "movie_reference_backend": "local",
                    "movie_render_backend": "local",
                },
            )
            self.assertEqual(200, created.status_code, created.text)
            project_dir = Path(temp_dir) / "legacy-door"
            for relative in ("screenplay.json", "screenplay.md", "narrative_plan.json", "scene_cards.json", "shot_cards.json"):
                path = project_dir / "movie" / relative
                if path.exists():
                    path.unlink()

            job = client.post("/api/projects/legacy-door/jobs", json={"action": "movie-full-auto"})

            self.assertEqual(200, job.status_code, job.text)
            job_id = job.json()["id"]
            for _ in range(50):
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] == "succeeded":
                    break
                time.sleep(0.01)
            self.assertEqual("succeeded", status["status"])
            self.assertTrue((project_dir / "movie" / "screenplay.json").exists())
            self.assertTrue((project_dir / "movie" / "narrative_plan.json").exists())
            self.assertTrue((project_dir / "movie" / "scene_cards.json").exists())
            self.assertTrue((project_dir / "movie" / "shot_cards.json").exists())

    def test_api_starts_movie_reference_job_and_updates_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)
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
            self.assertEqual("movie/references/actors/main_character/views/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertEqual("movie/references/locations/primary_location/views/hero.png", manifest["locations"][0]["msr_sheet_path"])

    def test_api_starts_movie_render_job_from_existing_msr_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)
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
                    "movie_render_backend": "local",
                },
            )
            self.assertEqual(200, created.status_code, created.text)
            project_dir = Path(temp_dir) / "door-below"
            manifest = json.loads((project_dir / "movie" / "references" / "manifest.json").read_text())
            manifest["actors"][0]["msr_sheet_path"] = "movie/references/actors/main_character/msr_sheet.png"
            manifest["locations"][0]["msr_sheet_path"] = "movie/references/locations/primary_location/views/hero.png"
            manifest["generator_backend"] = "local"
            (project_dir / "movie" / "references" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            from feverslop.application.movie_msr_enrichment import enrich_movie_render_plan_with_msr_prompts

            enrich_movie_render_plan_with_msr_prompts(project_dir=project_dir)

            job = client.post("/api/projects/door-below/jobs", json={"action": "movie-render"})

            self.assertEqual(200, job.status_code, job.text)
            job_id = job.json()["id"]
            for _ in range(50):
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] == "succeeded":
                    break
                time.sleep(0.01)
            self.assertEqual("succeeded", status["status"])
            self.assertTrue((project_dir / "output" / "movie" / "door-below.mp4").exists())
            self.assertEqual("movie-render", status["action"])
            self.assertTrue(any("Rendered clip" in line and "1/" in line for line in status["logs"]))
            render_step = next(step for step in status["steps"] if step["name"] == "LTX MSR native-audio render")
            self.assertEqual(100, render_step["progress"])

    def test_movie_full_auto_rejects_non_movie_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = NativeStudioHarness(temp_dir)
            created = client.post(
                "/api/projects",
                json={"project_type": "standard_music_video", "name": "Song"},
            )
            self.assertEqual(200, created.status_code, created.text)

            job = client.post("/api/projects/song/jobs", json={"action": "movie-full-auto"})

            self.assertEqual(400, job.status_code)
            self.assertIn("movie", job.text.lower())

    def test_screenplay_prompt_requires_dialogue(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieActor, MovieBible, MovieContinuityRule, MovieLocation, StoryArch

        class CapturingLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, system_prompt=None):
                self.calls.append(prompt)
                return json.dumps({
                    "title": "Test",
                    "source_type": "short_story",
                    "dialogue_language": "English",
                    "scenes": [{
                        "scene_id": "scene_0001",
                        "heading": "INT. ROOM - DAY",
                        "summary": "test",
                        "action": "test",
                        "dialogue": "A: hello",
                        "actor_ids": ["actor_a"],
                        "location_id": "room",
                        "source_span": "",
                        "dramatic_purpose": "",
                        "conflict": "",
                        "emotional_turn": "",
                        "subtext": "",
                        "dialogue_function": "",
                    }],
                })

        planner = LLMMoviePlanner(CapturingLLM())
        bible = MovieBible(
            title="Test",
            premise="A test",
            story_arch=StoryArch(title="Test", premise="A test", beats=("beat 1",)),
            actors=(MovieActor(id="actor_a", name="A", role="lead", visual_description="A"),),
            locations=(MovieLocation(id="room", name="Room", visual_description="Room"),),
            continuity=(MovieContinuityRule(id="vis", description="visual"),),
            style_constraints=(),
            runtime_constraints={},
        )
        planner.generate_movie_screenplay(
            title="Test",
            source_type="short_story",
            story_text="A short story about A.",
            desired_length=30,
            bible=bible,
            story_arch=bible.story_arch,
            story_design=None,
            config={},
        )
        prompt = "\n".join(planner.llm.calls).lower()
        self.assertIn("dialogue is mandatory", prompt)
        self.assertIn("every scene with two or more actors", prompt)
        self.assertIn("voiceover", prompt)


class TestLocationMergeHelpers(unittest.TestCase):
    """Tests for location fuzzy matching, deduplication, and visual description cleanup."""

    def test_location_base_name_strips_time_of_day(self):
        from feverslop.adapters.movie_planning import _location_base_name

        self.assertEqual("BLACKWOOD FOREST", _location_base_name("EXT. BLACKWOOD FOREST - DAY"))
        self.assertEqual("Blackwood Forest", _location_base_name("Blackwood Forest - NIGHT"))
        self.assertEqual("Blackwood Forest", _location_base_name("Blackwood Forest - DUSK"))
        self.assertEqual("Blackwood Forest", _location_base_name("Blackwood Forest - Dawn"))
        self.assertEqual("Blackwood Forest", _location_base_name("Blackwood Forest - MOMENTS LATER"))
        self.assertEqual("Blackwood Forest", _location_base_name("Blackwood Forest - LATE AFTERNOON"))
        self.assertEqual("Blackwood Forest", _location_base_name("Blackwood Forest - CONTINUOUS"))

    def test_location_base_name_strips_parentheticals(self):
        from feverslop.adapters.movie_planning import _location_base_name

        self.assertEqual("Hut", _location_base_name("Hut (2024)"))
        self.assertEqual("Hut", _location_base_name("Hut (Interior)"))
        self.assertEqual("Clearing", _location_base_name("Clearing - DAY (2024)"))

    def test_location_base_name_strips_int_ext(self):
        from feverslop.adapters.movie_planning import _location_base_name

        self.assertEqual("HUT", _location_base_name("INT. HUT"))
        self.assertEqual("HUT", _location_base_name("EXT. HUT - NIGHT"))
        self.assertEqual("HUT", _location_base_name("INT/EXT. HUT"))

    def test_location_base_name_complex(self):
        from feverslop.adapters.movie_planning import _location_base_name

        self.assertEqual("BLACKWOOD FOREST", _location_base_name("EXT. BLACKWOOD FOREST - DAY (2024)"))
        self.assertEqual("STONE HUT", _location_base_name("INT. STONE HUT - NIGHT (2024)"))
        self.assertEqual("Clearing near the river", _location_base_name("Clearing near the river - LATE AFTERNOON"))

    def test_location_id_matches_exact(self):
        from feverslop.adapters.movie_planning import _location_id_matches

        self.assertTrue(_location_id_matches("blackwood_forest", "blackwood_forest", name_a="Blackwood Forest", name_b="Blackwood Forest"))

    def test_location_id_matches_fuzzy(self):
        from feverslop.adapters.movie_planning import _location_id_matches

        self.assertTrue(_location_id_matches(
            "blackwood_forest_day_2024", "blackwood_forest",
            name_a="EXT. BLACKWOOD FOREST - DAY (2024)",
            name_b="Blackwood Forest"
        ))

    def test_location_id_matches_different(self):
        from feverslop.adapters.movie_planning import _location_id_matches

        self.assertFalse(_location_id_matches(
            "hut", "clearing",
            name_a="INT. HUT - NIGHT",
            name_b="Clearing"
        ))

    def test_location_visual_description_basic(self):
        from feverslop.adapters.movie_planning import _location_visual_description

        desc = _location_visual_description("EXT. FOREST - DAY", "Trees tower overhead. The wind howls through the branches.", character_names=("Leo",))
        self.assertIn("FOREST", desc)
        self.assertIn("trees", desc.lower())

    def test_location_visual_description_strips_character_actions(self):
        from feverslop.adapters.movie_planning import _location_visual_description

        desc = _location_visual_description("EXT. HUT - NIGHT", "Rain pours down. LEO clutches his chest. He runs away. Lightning strikes nearby.", character_names=("LEO",))
        self.assertNotIn("LEO", desc)
        self.assertNotIn("Leo", desc)
        self.assertNotIn("clutches", desc)
        self.assertNotIn("runs away", desc)

    def test_location_visual_description_strips_pronoun_actions(self):
        from feverslop.adapters.movie_planning import _location_visual_description

        desc = _location_visual_description("EXT. CLEARING", "The ground is muddy. He walks forward. She turns around. A bird flies overhead.", character_names=())
        self.assertNotIn("He walks", desc)
        self.assertNotIn("She turns", desc)
        self.assertIn("bird", desc.lower())

    def test_location_visual_description_strips_camera_directions(self):
        from feverslop.adapters.movie_planning import _location_visual_description

        desc = _location_visual_description("INT. ROOM", "Camera pans across the desk. CUT TO wide shot. The room is dimly lit.", character_names=())
        self.assertNotIn("Camera", desc)
        self.assertNotIn("CUT TO", desc)

    def test_location_visual_description_strips_dialogue_verbs(self):
        from feverslop.adapters.movie_planning import _location_visual_description

        desc = _location_visual_description("EXT. STREET", "The street is empty. LEO says nothing. The streetlights flicker.", character_names=("LEO",))
        self.assertNotIn("says", desc)

    def test_is_character_action_pronoun(self):
        from feverslop.adapters.movie_planning import _is_character_action

        self.assertTrue(_is_character_action("He walks forward.", character_names=()))
        self.assertTrue(_is_character_action("She turns around.", character_names=()))
        self.assertTrue(_is_character_action("They leave the room.", character_names=()))
        self.assertTrue(_is_character_action("It breaks.", character_names=()))

    def test_is_character_action_possessive(self):
        from feverslop.adapters.movie_planning import _is_character_action

        self.assertTrue(_is_character_action("His hand shakes.", character_names=()))
        self.assertTrue(_is_character_action("Her eyes widen.", character_names=()))
        self.assertTrue(_is_character_action("Their footsteps echo.", character_names=()))

    def test_is_character_action_name(self):
        from feverslop.adapters.movie_planning import _is_character_action

        self.assertTrue(_is_character_action("LEO clutches his chest.", character_names=("LEO",)))
        self.assertTrue(_is_character_action("Leo runs forward.", character_names=("LEO",)))
        self.assertFalse(_is_character_action("The old tree cracks.", character_names=("LEO",)))

    def test_is_character_action_camera(self):
        from feverslop.adapters.movie_planning import _is_character_action

        self.assertTrue(_is_character_action("Camera pans left.", character_names=()))
        self.assertTrue(_is_character_action("CUT TO wide shot.", character_names=()))
        self.assertTrue(_is_character_action("FADE TO BLACK.", character_names=()))
        self.assertTrue(_is_character_action("DISSOLVE TO interior.", character_names=()))
        self.assertTrue(_is_character_action("WE SEE the landscape.", character_names=()))

    def test_is_character_action_vo_os(self):
        from feverslop.adapters.movie_planning import _is_character_action

        self.assertTrue(_is_character_action("LEO (V.O.) narrates.", character_names=("LEO",)))
        self.assertTrue(_is_character_action("LEO (O.S.) calls out.", character_names=("LEO",)))

    def test_is_character_action_environment_false(self):
        from feverslop.adapters.movie_planning import _is_character_action

        self.assertFalse(_is_character_action("The wind howls through the trees.", character_names=("LEO",)))
        self.assertFalse(_is_character_action("Rain pours down on the streets.", character_names=()))
        self.assertFalse(_is_character_action("The moon rises over the forest.", character_names=()))

    def test_location_base_collides_exact(self):
        from feverslop.adapters.movie_planning import _location_base_collides

        self.assertTrue(_location_base_collides("hut", {"hut"}))
        self.assertFalse(_location_base_collides("clearing", {"hut"}))

    def test_location_base_collides_substring(self):
        from feverslop.adapters.movie_planning import _location_base_collides

        self.assertTrue(_location_base_collides("hut", {"stone_hut"}))
        self.assertTrue(_location_base_collides("stone_hut", {"hut"}))
        self.assertFalse(_location_base_collides("hut", {"cabin"}))

    def test_merge_screenplay_fuzzy_matches_locations(self):
        from feverslop.adapters.movie_planning import _merge_screenplay_references
        from feverslop.domain.movie import MovieLocation

        llm_locations = [
            MovieLocation(id="blackwood_forest", name="Blackwood Forest", visual_description="A dense ancient forest with towering oaks."),
            MovieLocation(id="hut", name="Hut", visual_description="A stone hut on a hill."),
        ]
        screenplay_locations = [
            MovieLocation(id="ext_blackwood_forest_day_2024", name="EXT. BLACKWOOD FOREST - DAY (2024)", visual_description="Blackwood Forest. Day (2024). Trees tower overhead."),
        ]
        merged = _merge_screenplay_references(screenplay_locations, llm_locations)
        # merged: fuzzy-matched screenplay->LLM (keeps LLM description) + unmatched LLM hut
        self.assertEqual(len(merged), 2)
        # The fuzzy-matched item preserves screenplay's ID but gets LLM's description
        matched_ids = {m.id for m in merged}
        self.assertIn("blackwood_forest", matched_ids)
        self.assertIn("hut", matched_ids)
        merged_forest = next(m for m in merged if m.id == "blackwood_forest")
        self.assertEqual(merged_forest.visual_description, "A dense ancient forest with towering oaks.")

    def test_merge_screenplay_fallback_keeps_screenplay_description(self):
        from feverslop.adapters.movie_planning import _merge_screenplay_references
        from feverslop.domain.movie import MovieLocation

        llm_locations = [
            MovieLocation(id="clearing", name="Clearing", visual_description="Clearing."),
        ]
        screenplay_locations = [
            MovieLocation(id="clearing_near_the_river", name="Clearing near the river", visual_description="Clearing near the river. Water flows nearby."),
        ]
        merged = _merge_screenplay_references(screenplay_locations, llm_locations)
        # Different base IDs: screenplay kept as-is, LLM "clearing" appended as unmatched
        self.assertEqual(len(merged), 2)
        screenplay_item = next(m for m in merged if m.id == "clearing_near_the_river")
        self.assertEqual(screenplay_item.visual_description, "Clearing near the river. Water flows nearby.")

    def test_merge_screenplay_fallback_promotes_screenplay_when_llm_description_is_just_name(self):
        from feverslop.adapters.movie_planning import _merge_screenplay_references
        from feverslop.domain.movie import MovieLocation

        llm_locations = [
            MovieLocation(id="hut", name="Hut", visual_description="Hut"),
        ]
        screenplay_locations = [
            MovieLocation(id="hut", name="Hut", visual_description="Stone hut on a hill. Moss on the walls."),
        ]
        merged = _merge_screenplay_references(screenplay_locations, llm_locations)
        self.assertEqual(len(merged), 1)
        # When LLM description == LLM name, fallback to screenplay description
        self.assertEqual(merged[0].visual_description, "Stone hut on a hill. Moss on the walls.")

    def test_merge_screenplay_no_llm_uses_screenplay_only(self):
        from feverslop.adapters.movie_planning import _merge_screenplay_references
        from feverslop.domain.movie import MovieLocation

        screenplay_locations = [
            MovieLocation(id="mysterious_gate", name="Mysterious Gate", visual_description="Mysterious Gate. Old iron bars."),
        ]
        merged = _merge_screenplay_references(screenplay_locations, [])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].id, "mysterious_gate")


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


def _continuity_handoff_factory(
    postprocessor,
    project_dir: Path,
    selected_rerender: bool,
):
    from feverslop.adapters.postprocessor_frame_extractor import (
        PostprocessorFrameExtractor,
    )
    from feverslop.application.continuity_handoff import (
        ContinuityHandoffUseCase,
    )

    return ContinuityHandoffUseCase(
        PostprocessorFrameExtractor(
            postprocessor,
            project_dir=project_dir,
            selected_rerender=selected_rerender,
        )
    )


    if __name__ == "__main__":
        unittest.main()


class TestRefineLocationPrompts(unittest.TestCase):
    """Tests for LLM-based location description refinement."""

    def test_refine_locations_prompt_includes_location_data(self):
        from feverslop.adapters.movie_planning import _refine_location_prompts_prompt
        from feverslop.domain.movie import MovieLocation

        locations = (
            MovieLocation(id="garden", name="GARDEN", visual_description="GARDEN"),
            MovieLocation(id="stone_hut", name="Stone Hut", visual_description="Stone Hut"),
        )
        prompt = _refine_location_prompts_prompt(locations, "EXT. GARDEN - DAY\nA wild herb garden.")

        self.assertIn('"id": "garden"', prompt)
        self.assertIn('"id": "stone_hut"', prompt)
        self.assertIn("GARDEN", prompt)
        self.assertIn("physical environment", prompt.lower())
        self.assertIn("no people", prompt)
        self.assertIn("no text", prompt)

    def test_refine_locations_prompt_rules_are_present(self):
        from feverslop.adapters.movie_planning import _refine_location_prompts_prompt
        from feverslop.domain.movie import MovieLocation

        locations = (MovieLocation(id="x", name="X", visual_description="X"),)
        prompt = _refine_location_prompts_prompt(locations, "test")

        self.assertIn("visual_description", prompt)
        self.assertIn("image_prompt", prompt)
        self.assertIn("Remove all character names", prompt)
        self.assertIn("Wide establishing view", prompt)
        self.assertIn("production design", prompt)

    def test_refine_locations_success(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieLocation

        class FakeLLM:
            def complete_prompt(self, prompt, *, system_prompt):
                return '{"locations": [{"id": "garden", "visual_description": "Overgrown herb garden with stone paths", "image_prompt": "Overgrown herb garden with stone paths. Wide establishing view, production design, lighting, atmosphere, no people, no text."}]}'

        planner = LLMMoviePlanner(FakeLLM())
        locations = (
            MovieLocation(id="garden", name="GARDEN", visual_description="GARDEN"),
        )
        result = planner.refine_locations(locations, source_text="EXT. GARDEN - DAY")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "garden")
        self.assertIn("herb garden", result[0].visual_description.lower())
        self.assertIn("no people", result[0].image_prompt.lower())

    def test_refine_locations_removes_reference_sheet_wording_from_image_prompt(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieLocation

        class FakeLLM:
            def complete_prompt(self, prompt, *, system_prompt):
                return '{"locations": [{"id": "garden", "visual_description": "Overgrown garden", "image_prompt": "Cinematic environment reference sheet for Garden. Wide establishing view."}]}'

        result = LLMMoviePlanner(FakeLLM()).refine_locations(
            (MovieLocation(id="garden", name="Garden", visual_description="Garden"),),
            source_text="EXT. GARDEN - DAY",
        )

        self.assertNotIn("reference sheet", result[0].image_prompt.lower())
        self.assertIn("wide establishing view", result[0].image_prompt.lower())

    def test_refine_locations_fallback_on_parse_error(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieLocation

        class FakeLLM:
            def complete_prompt(self, prompt, *, system_prompt):
                raise ConnectionError("llm down")

        planner = LLMMoviePlanner(FakeLLM())
        locations = (
            MovieLocation(id="garden", name="GARDEN", visual_description="GARDEN"),
        )
        result = planner.refine_locations(locations, source_text="test")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "garden")
        self.assertEqual(result[0].visual_description, "GARDEN")

    def test_refine_locations_fallback_on_invalid_json(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieLocation

        class FakeLLM:
            def complete_prompt(self, prompt, *, system_prompt):
                return "not json at all"

        planner = LLMMoviePlanner(FakeLLM())
        locations = (
            MovieLocation(id="garden", name="GARDEN", visual_description="GARDEN"),
        )
        result = planner.refine_locations(locations, source_text="test")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "garden")
        self.assertEqual(result[0].visual_description, "GARDEN")

    def test_refine_locations_skips_missing_ids(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import MovieLocation

        class FakeLLM:
            def complete_prompt(self, prompt, *, system_prompt):
                return '{"locations": [{"id": "other", "visual_description": "Something", "image_prompt": "Something"}]}'

        planner = LLMMoviePlanner(FakeLLM())
        locations = (
            MovieLocation(id="garden", name="GARDEN", visual_description="GARDEN"),
        )
        result = planner.refine_locations(locations, source_text="test")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "garden")
        self.assertEqual(result[0].visual_description, "GARDEN")
        self.assertEqual(result[0].image_prompt, "")

    def test_refine_locations_wired_into_generate_movie_bible(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import StoryArch

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, *, system_prompt):
                self.calls.append(system_prompt)
                if "film development producer" in system_prompt:
                    return '{"title": "Test", "premise": "Test", "actors": [{"id": "a", "name": "A", "role": "r", "visual_description": "A"}], "locations": [{"id": "loc", "name": "LOC", "visual_description": "LOC"}]}'
                elif "production designer" in system_prompt:
                    return '{"locations": [{"id": "loc", "visual_description": "Refined environment", "image_prompt": "Refined environment. Wide establishing view, production design, lighting, atmosphere, no people, no text."}]}'
                return "{}"

        llm = FakeLLM()
        planner = LLMMoviePlanner(llm)
        bible = planner.generate_movie_bible(
            title="Test",
            source_type="short_story",
            story_text="Test story",
            desired_length=12,
            story_arch=StoryArch(title="Test", premise="Test", beats=("beat",)),
            config={"refine_location_prompts": True},
        )
        self.assertEqual(len(bible.locations), 1)
        self.assertEqual(bible.locations[0].id, "loc")
        self.assertIn("Refined environment", bible.locations[0].visual_description)
        self.assertIn("no people", bible.locations[0].image_prompt)
        self.assertEqual(2, len(llm.calls))

    def test_refine_locations_not_wired_without_config(self):
        from feverslop.adapters.movie_planning import LLMMoviePlanner
        from feverslop.domain.movie import StoryArch

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete_prompt(self, prompt, *, system_prompt):
                self.calls.append(system_prompt)
                return '{"title": "Test", "premise": "Test", "actors": [{"id": "a", "name": "A", "role": "r", "visual_description": "A"}], "locations": [{"id": "loc", "name": "LOC", "visual_description": "LOC"}]}'

        llm = FakeLLM()
        planner = LLMMoviePlanner(llm)
        planner.generate_movie_bible(
            title="Test",
            source_type="short_story",
            story_text="Test story",
            desired_length=12,
            story_arch=StoryArch(title="Test", premise="Test", beats=("beat",)),
            config={},
        )
        self.assertEqual(1, len(llm.calls))
        self.assertNotIn("production designer", llm.calls[0])

    def test_reference_manifest_prefers_image_prompt(self):
        from feverslop.application.movie import _reference_manifest
        from feverslop.domain.movie import MovieActor, MovieBible, MovieLocation, MovieProject, StoryArch

        bible = MovieBible(
            title="Test",
            premise="Test",
            story_arch=StoryArch(title="Test", premise="Test", beats=("beat",)),
            actors=(MovieActor(id="a", name="A", role="r", visual_description="A"),),
            locations=(MovieLocation(id="loc", name="LOC", visual_description="LOC", image_prompt="Refined. Wide establishing view, production design, lighting, atmosphere, no people, no text."),),
            continuity=(),
            style_constraints=(),
            runtime_constraints={},
        )
        project = MovieProject(
            slug="test",
            name="Test",
            bible=bible,
            story_arch=StoryArch(title="Test", premise="Test", beats=("beat",)),
            shots=(),
            duration_seconds=12,
            width=1280,
            height=704,
            mode="scaffold",
        )
        manifest = _reference_manifest(project)
        loc = manifest["locations"][0]
        self.assertEqual("LOC", loc["visual_description"])
        self.assertIn("Refined", loc["image_prompt"])
        self.assertIn("no people", loc["image_prompt"])

    def test_reference_manifest_falls_back_to_visual_description(self):
        from feverslop.application.movie import _reference_manifest
        from feverslop.domain.movie import MovieActor, MovieBible, MovieLocation, MovieProject, StoryArch

        bible = MovieBible(
            title="Test",
            premise="Test",
            story_arch=StoryArch(title="Test", premise="Test", beats=("beat",)),
            actors=(MovieActor(id="a", name="A", role="r", visual_description="A"),),
            locations=(MovieLocation(id="loc", name="LOC", visual_description="LOC", image_prompt=""),),
            continuity=(),
            style_constraints=(),
            runtime_constraints={},
        )
        project = MovieProject(
            slug="test",
            name="Test",
            bible=bible,
            story_arch=StoryArch(title="Test", premise="Test", beats=("beat",)),
            shots=(),
            duration_seconds=12,
            width=1280,
            height=704,
            mode="scaffold",
        )
        manifest = _reference_manifest(project)
        loc = manifest["locations"][0]
        self.assertEqual("LOC", loc["image_prompt"])
        self.assertEqual("LOC", loc["prompt"])

    def test_manifest_location_prefers_image_prompt(self):
        from feverslop.application.movie_artifacts import _manifest_location

        result = _manifest_location(
            {"id": "loc", "name": "LOC", "visual_description": "LOC", "image_prompt": "Refined env."},
            {},
        )
        self.assertEqual("LOC", result["visual_description"])
        self.assertEqual("Refined env.", result["image_prompt"])
        self.assertEqual("Refined env.", result["prompt"])

    def test_manifest_location_falls_back(self):
        from feverslop.application.movie_artifacts import _manifest_location

        result = _manifest_location(
            {"id": "loc", "name": "LOC", "visual_description": "LOC"},
            {},
        )
        self.assertEqual("LOC", result["visual_description"])
        self.assertEqual("LOC", result["image_prompt"])
