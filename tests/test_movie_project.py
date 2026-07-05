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

    def concat_clips(self, concat_list, output_file, video_only=False, reencode=False):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"movie")
        self.concat_calls.append((Path(concat_list), output_file, video_only, reencode))
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
    def test_movie_scaffold_persists_screenplay_memory_and_shot_cards(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=DeterministicMoviePlanner(), projects_root=Path(temp_dir)).execute(
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
            screenplay = json.loads((root / "movie" / "screenplay.json").read_text(encoding="utf-8"))
            narrative = json.loads((root / "movie" / "narrative_plan.json").read_text(encoding="utf-8"))
            scene_cards = json.loads((root / "movie" / "scene_cards.json").read_text(encoding="utf-8"))
            shot_cards = json.loads((root / "movie" / "shot_cards.json").read_text(encoding="utf-8"))
            plan = json.loads((root / "movie" / "render_plan.json").read_text(encoding="utf-8"))

            self.assertEqual("movie/screenplay.json", plan["movie_screenplay_path"])
            self.assertEqual("movie/shot_cards.json", plan["movie_shot_cards_path"])
            self.assertEqual("English", screenplay["dialogue_language"])
            self.assertEqual(["mara"], screenplay["scenes"][0]["actor_ids"])
            self.assertEqual("archive", screenplay["scenes"][0]["location_id"])
            self.assertIn("It remembers me.", screenplay["scenes"][0]["dialogue"])
            self.assertEqual("scene_0001", narrative["sequences"][0]["scene_ids"][0])
            self.assertEqual("shot_0001", scene_cards["scene_cards"][0]["shot_ids"][0])
            self.assertEqual("shot_0001", shot_cards["shot_cards"][0]["shot_id"])
            self.assertIn("start_frame_brief", shot_cards["shot_cards"][0])
            self.assertIn("end_frame_brief", shot_cards["shot_cards"][0])
            self.assertNotIn("INT. ARCHIVE", shot_cards["memory_pack"]["current_shot"]["description"])

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

        with tempfile.TemporaryDirectory() as temp_dir:
            ScaffoldMovieUseCase(planner=BiblePlanner(), projects_root=Path(temp_dir)).execute(
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
            ScaffoldMovieUseCase(planner=DeterministicMoviePlanner(), projects_root=Path(temp_dir)).execute(
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
            self.assertIn("MARA: It remembers me.", prompt)
            self.assertEqual("same charcoal coat", enriched["shots"][0]["continuity_notes"])
            self.assertNotIn("Continuity:", prompt)
            self.assertNotIn("CONTINUITY CONTRACT", prompt)
            self.assertNotIn("Continuity:", relay["prompt"])
            self.assertNotIn("CONTINUITY CONTRACT", relay["prompt"])
            self.assertNotIn("Full-body cinematic character reference sheet", prompt)
            self.assertNotIn("Four vertical panels", prompt)
            self.assertEqual(0, relay["frame_start"])
            self.assertEqual(95, relay["frame_end"])
            self.assertNotIn("start_frame", relay)
            self.assertNotIn("end_frame", relay)

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
            self.assertIn("Dialogue language: German", prompt)
            self.assertIn("spoken in German", prompt)
            self.assertIn("Dialogue language: German", relay["prompt"])

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
            self.assertIn("No spoken dialogue", prompt)
            self.assertIn("Do not invent spoken lines", prompt)
            self.assertIn("No spoken dialogue", relay["prompt"])

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
                self.assertIn("Hans: Halt den Mund, Karl.", value)
                self.assertIn("Dialogue language: German", value)
                self.assertNotIn("Continuity:", value)
                self.assertNotIn("CONTINUITY CONTRACT", value)
                self.assertNotIn("Karl whispered that he cannot feel the ground", value)
                self.assertNotIn("Friedrich enters with a memory of home", value)
            self.assertIn("Reference image 1: Hans", ltx["msr_global_prompt"])
            self.assertIn("Reference image 2: Karl", ltx["msr_global_prompt"])
            self.assertIn("Background reference: Trench Line", ltx["msr_global_prompt"])
            self.assertNotIn("CONTINUITY CONTRACT", ltx["msr_global_prompt"])

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
            actor_sheet.write_bytes(b"actor")
            location_sheet.write_bytes(b"location")
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

            patched = backend.build_workflow(scene, prompt=scene["ltx"]["original_style_i2v_prompt"])

            relay_inputs = patched["27"]["inputs"]
            self.assertIn("Reference image 1: Hans", relay_inputs["global_prompt"])
            self.assertIn("Background reference: Trench Line", relay_inputs["global_prompt"])
            self.assertIn("Hans turns toward Karl in the trench", relay_inputs["local_prompts"])
            self.assertIn("Hans: Halt den Mund, Karl.", relay_inputs["local_prompts"])
            self.assertNotIn("Continuity:", relay_inputs["local_prompts"])
            self.assertNotIn("CONTINUITY CONTRACT", relay_inputs["local_prompts"])
            self.assertNotIn("Karl whispered that he cannot feel the ground", relay_inputs["local_prompts"])
            self.assertNotIn("Friedrich enters with a memory of home", relay_inputs["local_prompts"])

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
            self.assertIn("Reference image 1: Mara", ltx["msr_global_prompt"])
            self.assertIn("charcoal coat", ltx["msr_global_prompt"])
            self.assertIn("Reference image 2: Ivo", ltx["msr_global_prompt"])
            self.assertIn("white suit", ltx["msr_global_prompt"])
            self.assertIn("Background reference: Archive", ltx["msr_global_prompt"])

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

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(planner=Planner(), projects_root=Path(temp_dir)).execute(
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
            ).execute(
                MovieInput(
                    name="Mud Silence",
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

    def test_short_story_scaffold_still_generates_screenplay_from_unformatted_idea(self):
        from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
        from feverslop.adapters.movie_planning import DeterministicMoviePlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ScaffoldMovieUseCase(
                planner=DeterministicMoviePlanner(),
                projects_root=Path(temp_dir),
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
            self.assertEqual("ABANDONED STATION - NIGHT", manifest["locations"][0]["name"])
            self.assertIn("ABANDONED STATION - NIGHT", manifest["locations"][0]["prompt"])

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

    def test_movie_full_auto_regenerates_missing_planning_artifacts_for_legacy_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(temp_dir))
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

    def test_api_starts_movie_render_job_from_existing_msr_plan(self):
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
