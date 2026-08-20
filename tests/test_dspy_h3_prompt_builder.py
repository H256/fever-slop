import unittest
from importlib.resources import files
import tempfile
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    _format_relay_shots,
    _format_performance_timing,
    _normalize_relay_segments,
    _repair_audio_references,
    _scene_references,
)
from feverslop.adapters.movie_minimax_visual import _h3_movie_prompt
from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer
from feverslop.prompting.dspy_h3_generator import VideoPromptGenerator
from feverslop.prompting.dspy_h3_models import (
    MusicIntent,
    PlannedShot,
    PlannedSubject,
    PromptPlan,
    PromptMode,
    ReferenceAsset,
    ReferenceKind,
    ReferenceLimits,
    ReferenceUsage,
    ReferenceVideoPrompt,
    ResolvedPromptPlan,
    ResolvedReference,
    RetentionAnalysis,
    SubjectDefinition,
    VideoPromptRequest,
)
from feverslop.prompting.dspy_h3_generator_core import VideoPromptGenerator as CoreVideoPromptGenerator
from feverslop.prompting.dspy_h3_signatures import build_dspy_signatures


class FakeGeneratedPrompt:
    rendered_prompt = "subject_definitions: <Subject 1>\ndetailed_description: test"


class FakeGenerator:
    def __init__(self, result=None):
        self.requests = []
        self.result = result or FakeGeneratedPrompt()

    def __call__(self, request):
        self.requests.append(request)
        return self.result


class IncompleteAudioPrompt:
    rendered_prompt = """subject_definitions:
<Subject 1> is a singer.

summary: [reference generation] A singer performs.

retention_analysis:
<Subject 1>: fully_preserved - The singer remains recognizable.

detailed_description: <Subject 1> sings in a close-up.

overall_soundscape: A quiet room tone.

non_diegetic_music: N/A"""


class DspyH3PromptBuilderTests(unittest.TestCase):
    def test_formats_instrument_specific_beat_contact_guidance(self):
        prompt = _format_performance_timing({
            "references": {"actor_reference_descriptions": [
                {"name": "Drummer", "role": "Percussionist"},
            ]},
            "performance_timing": {
                "bpm": 120,
                "beats": [
                    {"time_seconds": 0.5, "downbeat": True, "impact": 0.8},
                    {"time_seconds": 1.0, "downbeat": False, "impact": 0.4},
                ],
            },
        })

        self.assertIn("BPM 120", prompt)
        self.assertIn("downbeats at 0.50s", prompt)
        self.assertIn("stick contact exactly on each listed beat", prompt)
        self.assertIn("rebound", prompt)
    def test_audio_repair_replaces_annotated_duplicates_with_canonical_copy_modes(self):
        prompt = """subject_definitions:
<Subject 1> (Drummer): A drummer. Source references: <Picture 1>.
<Audio 1> is an old vocal definition.
<Audio 2> is an old mix definition.

summary: [reference generation] <Subject 1> performs.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - stable.
<Audio 1> (appears in [Shot 1]): partially_copy - vocals.
<Audio 2> (appears in [Shot 1]): fully_copy - mix.
<Audio 1>: partially_copy - duplicate vocals.
<Audio 2>: partially_copy - contradictory duplicate mix.

detailed_description: <Subject 1> performs.

overall_soundscape: Music.

non_diegetic_music: N/A"""
        references = [
            {"label": "<Audio 1>", "kind": "audio", "role": "audio_reuse", "name": "vocals", "copy_mode": "partially_copy"},
            {"label": "<Audio 2>", "kind": "audio", "role": "audio_reuse", "name": "full_mix", "copy_mode": "fully_copy"},
        ]

        repaired = _repair_audio_references(prompt, references)

        retention = repaired.split("retention_analysis:", 1)[1].split("detailed_description:", 1)[0]
        self.assertEqual(1, retention.count("<Audio 1>"))
        self.assertEqual(1, retention.count("<Audio 2>"))
        self.assertIn("<Audio 1>: partially_copy", retention)
        self.assertIn("<Audio 2>: fully_copy", retention)

    def test_movie_minimax_adapter_uses_structured_dspy_r2v_prompt(self):
        from feverslop.adapters.movie_minimax_visual import _build_movie_h3_prompt

        class Builder:
            def build_h3_prompt(self, **kwargs):
                self.request = kwargs
                return {"prompt": "subject_definitions:\n<Subject 1> Leo\n\nsummary: Leo runs."}

        builder = Builder()
        prompt = _build_movie_h3_prompt(
            {
                "scene": 1,
                "description": "Leo runs through the forest.",
                "action": "Leo runs.",
                "camera": "Handheld tracking shot.",
                "references": {
                    "actor_msr_paths": ["movie/references/leo.png"],
                    "location_msr_path": "movie/references/forest.png",
                    "actor_ids": ["leo"],
                    "location_id": "forest",
                },
            },
            builder=builder,
            reference_root=Path("project"),
        )

        self.assertIn("subject_definitions:", prompt)
        self.assertEqual("ref", builder.request["mode"])
        self.assertEqual("movie", builder.request["video_type"])
        self.assertFalse(builder.request["append_relay_prompt"])
        self.assertIn("Leo runs through the forest", builder.request["concept"])

    def test_scene_references_pass_existing_visual_descriptions_to_dspy(self):
        references, _images = _scene_references(
            {
                "references": {
                    "actor_msr_paths": ["actor.png"],
                    "actor_ids": ["leo"],
                    "actor_reference_descriptions": [
                        {"id": "leo", "name": "Leo", "visual_description": "A weathered hiker."},
                    ],
                    "location_msr_path": "forest.png",
                    "location_id": "forest",
                    "location_reference_description": {
                        "id": "forest",
                        "name": "Ancient Forest",
                        "visual_description": "A dark ancient forest.",
                    },
                }
            },
            None,
            None,
        )

        self.assertEqual("A weathered hiker.", references[0]["description"])
        self.assertEqual("A dark ancient forest.", references[1]["description"])
        self.assertEqual("Leo", references[0]["name"])
        self.assertEqual("Ancient Forest", references[1]["name"])

    def test_scene_references_deduplicate_audio_paths_from_scene_and_global_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocal = root / "output" / "stems" / "vocals.wav"
            full_mix = root / "input" / "song.wav"
            references, _images = _scene_references(
                {
                    "references": {
                        "reference_audio_paths": [
                            "output/stems/vocals.wav",
                            "input/song.wav",
                        ],
                    }
                },
                {"vocals": vocal, "full_mix": full_mix},
                root,
            )

        audio_references = [reference for reference in references if reference["kind"] == "audio"]
        self.assertEqual(2, len(audio_references))
        self.assertEqual(["full_mix", "vocals"], [reference["name"] for reference in audio_references])

    def test_fully_instrumental_relay_excludes_vocal_stem_but_keeps_full_mix(self):
        references, _images = _scene_references(
            {
                "ltx": {"prompt_relay": [{
                    "frame_start": 0,
                    "frame_end": 120,
                    "state": "instrumental",
                    "prompt": "No vocal performance, mouth closed, no lip movement.",
                }]},
                "references": {
                    "reference_audio_paths": ["vocals.wav", "song.wav"],
                    "_stem_audio_tags": {
                        "vocals.wav": "audio_transfer - vocal singing lip-synced to the audio signal",
                        "song.wav": "full_mix - original song for beat and rhythm continuity",
                    },
                },
            },
            {"vocals": Path("vocals.wav"), "full_mix": Path("song.wav")},
            None,
        )

        audio = [reference for reference in references if reference["kind"] == "audio"]
        self.assertEqual(["full_mix"], [reference["name"] for reference in audio])
        self.assertEqual("fully_copy", audio[0]["copy_mode"])

    def test_scene_audio_labels_follow_role_stem_then_full_mix_order(self):
        references, _images = _scene_references(
            {
                "type": "instrumental",
                "ltx": {"prompt_relay": [{"state": "instrumental"}]},
                "references": {
                    "actor_reference_descriptions": [
                        {"name": "Drummer", "role": "Percussionist"},
                    ],
                    "reference_audio_paths": ["vocals.wav", "song.wav"],
                    "_stem_audio_tags": {
                        "vocals.wav": "audio_transfer - vocal singing lip-synced to the audio signal",
                        "song.wav": "full_mix - original song for beat and rhythm continuity",
                    },
                },
            },
            {
                "vocals": Path("vocals.wav"),
                "drums": Path("drums.wav"),
                "full_mix": Path("song.wav"),
            },
            None,
        )

        audio = [reference for reference in references if reference["kind"] == "audio"]
        self.assertEqual(
            [("<Audio 1>", "drums"), ("<Audio 2>", "full_mix")],
            [(reference["label"], reference["name"]) for reference in audio],
        )

    def test_unmanaged_audio_follows_managed_stems_to_match_backend_slots(self):
        references, _images = _scene_references(
            {
                "type": "vocals",
                "references": {
                    "actor_reference_descriptions": [
                        {"name": "Singer", "role": "Lead Singer"},
                    ],
                    "reference_audio_paths": ["ambience.wav"],
                },
            },
            {"vocals": Path("vocals.wav"), "full_mix": Path("song.wav")},
            None,
        )

        audio = [reference for reference in references if reference["kind"] == "audio"]
        self.assertEqual(
            ["vocals", "full_mix", "ambience"],
            [reference["name"] for reference in audio],
        )

    def test_local_picture_without_description_reaches_h3_analyzer_without_placeholder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            picture = root / "actor.png"
            picture.write_bytes(b"image")
            references, images = _scene_references(
                {
                    "references": {
                        "actor_msr_paths": ["actor.png"],
                        "actor_ids": ["leo"],
                        "actor_reference_descriptions": [{"id": "leo"}],
                    }
                },
                None,
                root,
            )

        self.assertEqual("", references[0]["description"])
        self.assertEqual([picture], images)

    def test_passes_general_steering_and_prompt_guidance_to_generator(self):
        generator = FakeGenerator()
        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={"segment_id": "seg-1", "duration_seconds": 2, "fps": 24},
            concept="A singer performs.",
            scene_details={},
            global_context={
                "story_idea": "A singer crosses a mountain.",
                "style": "Cinematic dark fantasy.",
                "subject": "The same singer throughout.",
                "steering": {"global": "Use only the configured locations."},
                "prompt_guidance": {"camera_motion": "Use deliberate tracking shots."},
            },
            mode="base",
        )

        notes = generator.requests[0]["notes"]
        self.assertIn("Use only the configured locations.", notes)
        self.assertIn("Use deliberate tracking shots.", notes)
        self.assertIn("A singer crosses a mountain.", notes)
    def test_minimax_movie_prompt_preserves_r2v_prompt_and_adds_relay_shots(self):
        prompt = _h3_movie_prompt({
            "h3": {"prompt": "Use <Picture 1> for the actor."},
            "duration_seconds": 6.4,
            "fps": 24,
            "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 154, "prompt": "The actor looks left."}]},
            "references": {"actor_msr_paths": ["actors/bard_woman/views/msr_sheet.png"]},
        })

        self.assertIn("<Picture 1>", prompt)
        self.assertIn("actors/bard_woman/views/msr_sheet.png", prompt)
        self.assertIn("[Shot 1, 0.00-6.40sec]", prompt)

    def test_normalizes_relay_frames_to_timed_shots(self):
        segment = {
            "duration_seconds": 6.4,
            "fps": 24,
            "ltx": {
                "prompt_relay": [
                    {"frame_start": 36, "frame_end": 153, "state": "vocals", "prompt": "The singer turns."},
                    {"frame_start": 153, "frame_end": 240, "state": "instrumental", "prompt": "The camera pulls back."},
                ]
            },
        }

        self.assertEqual(
            [
                {"shot": 1, "start_seconds": 1.5, "end_seconds": 6.375, "state": "vocals", "prompt": "The singer turns."},
                {"shot": 2, "start_seconds": 6.375, "end_seconds": 6.4, "state": "instrumental", "prompt": "The camera pulls back."},
            ],
            _normalize_relay_segments(segment),
        )

    def test_formats_relay_shots_with_minimax_syntax(self):
        shots = [
            {
                "shot": 1,
                "start_seconds": 1.5,
                "end_seconds": 6.4,
                "state": "vocals",
                "prompt": "The singer turns.",
                "source_prompt": "The singer cooks a rabbit over the fire.",
            },
        ]

        formatted = _format_relay_shots(shots)
        self.assertIn("[Shot 1, 1.50-6.40sec] (vocals) The singer turns.", formatted)
        self.assertIn("Required action and props to preserve: The singer cooks a rabbit over the fire.", formatted)

    def test_normalizes_image_like_relay_without_losing_source_action(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 4,
            "fps": 24,
            "ltx": {"prompt_relay": [{
                "frame_start": 0,
                "frame_end": 96,
                "prompt": "Cinematic close-up, warm firelight, the singer cooks a rabbit over the fire.",
                "source_prompt": "Cinematic close-up, warm firelight, the singer cooks a rabbit over the fire.",
            }]},
        })

        self.assertIn("cooks a rabbit", _format_relay_shots(shots))

    def test_passes_relay_segments_to_generator_and_appends_timed_shots(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator)

        result = builder.build_h3_prompt(
            segment={
                "segment_id": "seg-1",
                "type": "vocals",
                "duration_seconds": 6.4,
                "fps": 24,
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 154, "prompt": "The singer turns."}]},
                "references": {"actor_sheet_paths": ["actor.png"]},
            },
            concept="A singer performs.",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertNotIn("relay_segments_json", generator.requests[0])
        self.assertEqual(0.0, generator.requests[0]["relay_segments"][0]["start_seconds"])
        self.assertIn("[Shot 1, 0.00-6.40sec]", result["prompt"])
        self.assertIn("actor.png", " ".join(reference["source"] for reference in result["references"]))

    def test_repairs_missing_audio_reuse_references_in_all_required_sections(self):
        builder = DspyH3PromptBuilder(FakeGenerator(IncompleteAudioPrompt()))

        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "vocals"},
            concept="A singer performs.",
            scene_details={},
            global_context={},
            mode="ref",
            audio_paths={
                "vocals": Path("output/stems/vocals.wav"),
                "full_mix": Path("input/song.mp3"),
            },
        )

        prompt = result["prompt"]
        self.assertIn("<Audio 1> is the synchronized vocals audio reference and is reused for the scene.", prompt)
        self.assertIn("<Audio 2> is the synchronized full_mix audio reference and is reused for the scene.", prompt)
        self.assertIn("[reference generation + audio reuse]", prompt)
        self.assertIn("<Audio 1>: partially_copy", prompt)
        self.assertIn("<Audio 2>: fully_copy", prompt)
        self.assertIn("<Audio 1> and <Audio 2>", prompt)
        self.assertIn("overall_soundscape: A quiet room tone. The synchronized audio behavior follows <Audio 1> and <Audio 2>.", prompt)
        self.assertIn("non_diegetic_music: The synchronized audio references are scene inputs, not non-diegetic music: <Audio 1> and <Audio 2>.", prompt)

    def test_repairs_existing_audio_labels_with_wrong_role(self):
        weak = IncompleteAudioPrompt.rendered_prompt.replace(
            "<Subject 1> is a singer.",
            "<Subject 1> is a singer.\n<Audio 1> is a reference track.\n<Audio 2> is a reference track.",
        ).replace(
            "[reference generation]",
            "[reference generation + audio reuse]",
        ).replace(
            "<Subject 1>: fully_preserved - The singer remains recognizable.",
            "<Subject 1>: fully_preserved - The singer remains recognizable.\n<Audio 1>: reference - do not copy.\n<Audio 2>: reference - do not copy.",
        )
        generated = type("Generated", (), {"rendered_prompt": weak})()
        builder = DspyH3PromptBuilder(FakeGenerator(generated))

        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "vocals"},
            concept="A singer performs.",
            scene_details={},
            global_context={},
            audio_paths={"vocals": Path("output/stems/vocals.wav"), "full_mix": Path("input/song.mp3")},
        )

        prompt = result["prompt"]
        self.assertIn("<Audio 1> is the synchronized vocals audio reference and is reused for the scene.", prompt)
        self.assertIn("<Audio 2>: fully_copy - the synchronized full_mix audio is reused for this scene.", prompt)
        self.assertNotIn("is a reference track.", prompt)
        self.assertNotIn("do not copy.", prompt)

    def test_keeps_complete_audio_references_unchanged(self):
        complete = IncompleteAudioPrompt.rendered_prompt.replace(
            "<Subject 1> is a singer.",
            "<Subject 1> is a singer.\n<Audio 1> is the synchronized vocals audio reference and is reused for the scene.\n<Audio 2> is the synchronized full_mix audio reference and is reused for the scene.",
        ).replace(
            "[reference generation]",
            "[reference generation + audio reuse] A singer performs using <Audio 1> and <Audio 2>",
        ).replace(
            "<Subject 1>: fully_preserved - The singer remains recognizable.",
            "<Subject 1>: fully_preserved - The singer remains recognizable.\n<Audio 1>: partially_copy - retained.\n<Audio 2>: fully_copy - retained.",
        ).replace(
            "<Subject 1> sings in a close-up.",
            "<Subject 1> sings in a close-up with <Audio 1> and <Audio 2>.",
        ).replace(
            "A quiet room tone.",
            "A quiet room tone with <Audio 1> and <Audio 2>. The synchronized audio behavior follows <Audio 1> and <Audio 2>.",
        ).replace(
            "non_diegetic_music: N/A",
            "non_diegetic_music: The synchronized audio references are scene inputs, not non-diegetic music: <Audio 1> and <Audio 2>.",
        )
        generated = type("Generated", (), {"rendered_prompt": complete})()
        builder = DspyH3PromptBuilder(FakeGenerator(generated))

        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "vocals"},
            concept="A singer performs.",
            scene_details={},
            global_context={},
            mode="ref",
            audio_paths={"vocals": Path("output/stems/vocals.wav"), "full_mix": Path("input/song.mp3")},
        )

        self.assertEqual(complete, result["prompt"])

    def test_reference_prompt_accepts_audio_tags_in_non_diegetic_music(self):
        prompt = ReferenceVideoPrompt(
            subject_definitions=[],
            summary="A scene.",
            retention_analysis=[],
            detailed_description="A detailed scene.",
            overall_soundscape="The song is audible.",
            non_diegetic_music=(
                "N/A\n"
                "<Audio 1> (audio_transfer - vocal singing lip-synced to the audio signal)\n"
                "<Audio 2> (full_mix - original song for beat and rhythm continuity)"
            ),
        )

        self.assertIn("<Audio 1>", prompt.non_diegetic_music)

    def test_image_analysis_is_cached_for_repeated_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "reference.png"
            image_path.write_bytes(b"image")
            calls = []

            class Analysis:
                objective_description = "A reference image"
                visible_subjects = []
                environment = ""
                visual_style = ""
                composition = ""
                lighting = ""
                visible_text = []

            def predictor(**kwargs):
                calls.append(kwargs)
                return type("Prediction", (), {"analysis": Analysis()})()

            reference = ReferenceAsset(
                kind=ReferenceKind.PICTURE,
                source=str(image_path),
                role="subject",
            )
            analyzer = LocalImageAnalyzer(predictor)

            first = analyzer.analyze(reference)
            second = analyzer.analyze(reference)

        self.assertEqual("A reference image", first)
        self.assertEqual(first, second)
        self.assertEqual(1, len(calls))

    def test_generator_converts_openai_url_object_for_dspy(self):
        class UrlObject:
            def __str__(self):
                return "http://your-llm-server.local/v1"

        class Client:
            base_url = UrlObject()
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            max_tokens = 16384

        guides = files("feverslop.prompting.guides")
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertEqual("http://your-llm-server.local/v1", lm_factory.call_args.kwargs["api_base"])
        self.assertFalse(lm_factory.call_args.kwargs["cache"])

    def test_generator_passes_dspy_cache_setting_to_lm(self):
        class Client:
            base_url = "http://your-llm-server.local/v1"
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            max_tokens = 16384
            dspy_cache = True

        guides = files("feverslop.prompting.guides")
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertTrue(lm_factory.call_args.kwargs["cache"])

    def test_generator_passes_dspy_temperature_to_lm(self):
        class Client:
            base_url = "http://your-llm-server.local/v1"
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            dspy_temperature = 0.25
            max_tokens = 16384
            dspy_cache = False

        guides = files("feverslop.prompting.guides")
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertEqual(0.25, lm_factory.call_args.kwargs["temperature"])

    def test_reference_limits_use_plural_picture_field(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        generator.limits = ReferenceLimits()
        generator.image_analyzer = type("Analyzer", (), {"should_analyze": lambda *_: False})()

        resolved = generator._resolve_references([
            ReferenceAsset(kind=ReferenceKind.PICTURE, source="actor.png", description="actor"),
            ReferenceAsset(kind=ReferenceKind.PICTURE, source="location.png", description="location"),
        ])

        self.assertEqual([item.label for item in resolved], ["<Picture 1>", "<Picture 2>"])

    def test_persistent_unknown_planner_references_fail_after_three_attempts_with_all_labels(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        plan = PromptPlan(
            creative_intent="Invalid",
            subjects=[PlannedSubject(
                name="Singer",
                description="A singer",
                source_references=["<Picture 9>"],
            )],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 9>",
                purpose="sync",
                details="invalid",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="A shot",
                reference_labels=["<Video 9>"],
            )],
            overall_soundscape="A song",
            music_intent=MusicIntent.NONE,
        )
        calls = []

        def planner(**kwargs):
            calls.append(kwargs)
            return type("Prediction", (), {"plan": plan})()

        generator.planner = planner
        request = VideoPromptRequest(
            mode=PromptMode.R2V,
            user_prompt="A scene",
            duration_seconds=5.0,
        )
        references = [ResolvedReference(
            label="<Picture 1>",
            kind="picture",
            source="actor.png",
            role="subject",
            description="A singer",
        )]

        with self.assertRaisesRegex(
            ValueError,
            r"unknown=\['<Audio 9>', '<Picture 9>', '<Video 9>'\].*allowed=\['<Picture 1>'\]",
        ):
            generator._plan(request, references)
        self.assertEqual(3, len(calls))

    def test_retries_planner_after_unknown_reference(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        plans = [
            PromptPlan(
                creative_intent="Invalid attempt",
                subjects=[PlannedSubject(
                    name="Singer",
                    description="A singer",
                    source_references=["<Picture 9>"],
                )],
                overall_soundscape="A song",
                music_intent=MusicIntent.NONE,
            ),
            PromptPlan(
                creative_intent="Valid attempt",
                subjects=[PlannedSubject(
                    name="Singer",
                    description="A singer",
                    source_references=["<Picture 1>"],
                )],
                overall_soundscape="A song",
                music_intent=MusicIntent.NONE,
            ),
        ]
        calls = []

        def planner(**kwargs):
            calls.append(kwargs)
            return type("Prediction", (), {"plan": plans[len(calls) - 1]})()

        generator.planner = planner
        request = VideoPromptRequest(
            mode=PromptMode.R2V,
            user_prompt="A scene",
            duration_seconds=5.0,
        )
        references = [ResolvedReference(
            label="<Picture 1>",
            kind="picture",
            source="actor.png",
            role="subject",
            description="A singer",
        )]

        result = generator._plan(request, references)

        self.assertEqual("Valid attempt", result.creative_intent)
        self.assertEqual(2, len(calls))
        self.assertIn("<Picture 9>", calls[1]["notes"])
        self.assertIn("<Picture 1>", calls[1]["notes"])

    def test_planner_retries_when_a_loaded_picture_is_not_mapped_to_a_subject(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        invalid = PromptPlan(
            creative_intent="Missing location",
            subjects=[PlannedSubject(
                name="Drummer",
                description="A drummer",
                source_references=["<Picture 1>"],
            )],
            overall_soundscape="A song",
            music_intent=MusicIntent.NONE,
        )
        valid = PromptPlan(
            creative_intent="Mapped location",
            subjects=[
                PlannedSubject(name="Drummer", description="A drummer", source_references=["<Picture 1>"]),
                PlannedSubject(name="Stage", description="A black stage", source_references=["<Picture 2>"]),
            ],
            overall_soundscape="A song",
            music_intent=MusicIntent.NONE,
        )
        calls = []

        def planner(**kwargs):
            calls.append(kwargs)
            return type("Prediction", (), {"plan": invalid if len(calls) == 1 else valid})()

        generator.planner = planner
        request = VideoPromptRequest(mode=PromptMode.R2V, user_prompt="A scene", duration_seconds=5.0)
        references = [
            ResolvedReference(label="<Picture 1>", kind="picture", source="actor.png", role="subject", name="Drummer", description="A drummer"),
            ResolvedReference(label="<Picture 2>", kind="picture", source="stage.png", role="environment", name="Stage", description="A stage"),
        ]

        result = generator._plan(request, references)

        self.assertEqual(1, len(calls))
        self.assertEqual(["<Subject 1>", "<Subject 2>"], [subject.label for subject in result.subjects])

    def test_planner_reconstructs_persistently_unmapped_visuals_with_warning(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        invalid = PromptPlan(
            creative_intent="Performance",
            subjects=[],
            overall_soundscape="A song",
            music_intent=MusicIntent.NONE,
        )
        calls = []

        def planner(**kwargs):
            calls.append(kwargs)
            return type("Prediction", (), {"plan": invalid.model_copy(deep=True)})()

        generator.planner = planner
        request = VideoPromptRequest(mode=PromptMode.R2V, user_prompt="A scene", duration_seconds=5.0)
        references = [
            ResolvedReference(label="<Picture 1>", kind="picture", source="actor.png", role="subject", name="Lead Singer", description="Silver-haired singer"),
            ResolvedReference(label="<Picture 2>", kind="picture", source="reef.png", role="environment", name="The Azure Reef", description="Blue crystalline reef"),
        ]

        with self.assertLogs("feverslop.prompting.dspy_h3_generator_core", level="INFO") as captured:
            result = generator._plan(request, references)

        self.assertEqual(1, len(calls))
        self.assertEqual(
            [["<Picture 1>"], ["<Picture 2>"]],
            [subject.source_references for subject in result.subjects],
        )
        self.assertEqual(["Lead Singer", "The Azure Reef"], [subject.name for subject in result.subjects])
        self.assertTrue(any("normalized required subject mappings" in message for message in captured.output))

    def test_reference_renderer_retries_unknown_subject_with_mismatch_details(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        calls = []

        def renderer(**kwargs):
            calls.append(kwargs)
            subject = "<Subject 3>" if len(calls) == 1 else "<Subject 1>"
            return type("Output", (), {
                "summary": f"{subject} performs.",
                "retention_analysis": [RetentionAnalysis(
                    target_label="<Subject 1>", mode="fully_preserved", details="stable"
                )],
                "detailed_description": f"{subject} performs on beat.",
                "overall_soundscape": "Music.",
                "non_diegetic_music": None,
            })()

        generator.reference_renderer = renderer
        generator.reference_guide_path = "minimax-h3-references.md"
        plan = ResolvedPromptPlan(
            creative_intent="Performance",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Drummer", description="A drummer",
                source_references=["<Picture 1>"],
            )],
            overall_soundscape="Music.",
            music_intent=MusicIntent.NONE,
        )
        request = VideoPromptRequest(mode=PromptMode.R2V, user_prompt="A drummer", duration_seconds=5)
        refs = [ResolvedReference(
            label="<Picture 1>", kind="picture", source="actor.png",
            role="subject", description="A drummer",
        )]

        output = generator._render_reference(request, plan, refs)

        self.assertEqual(2, len(calls))
        self.assertIn("undefined_subjects=['<Subject 3>']", calls[1]["notes"])
        self.assertEqual("<Subject 1> performs.", output.summary)

    def test_reference_renderer_retries_active_singing_in_instrumental_relay(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        calls = []

        def renderer(**kwargs):
            calls.append(kwargs)
            description = (
                "<Subject 1> sings with perfect lip sync."
                if len(calls) == 1
                else "<Subject 1> keeps the mouth relaxed and closed, with no singing or lip sync."
            )
            return type("Output", (), {
                "summary": "<Subject 1> is shown.",
                "retention_analysis": [RetentionAnalysis(
                    target_label="<Subject 1>", mode="fully_preserved", details="stable"
                )],
                "detailed_description": description,
                "overall_soundscape": "Instrumental music.",
                "non_diegetic_music": None,
            })()

        generator.reference_renderer = renderer
        generator.reference_guide_path = "minimax-h3-references.md"
        plan = ResolvedPromptPlan(
            creative_intent="Instrumental shot",
            subjects=[SubjectDefinition(
                label="<Subject 1>", name="Singer", description="A singer",
                source_references=["<Picture 1>"],
            )],
            overall_soundscape="Instrumental music.",
            music_intent=MusicIntent.NONE,
        )
        request = VideoPromptRequest(
            mode=PromptMode.R2V,
            user_prompt="An instrumental shot",
            duration_seconds=5,
            relay_segments=[{
                "start_seconds": 0,
                "end_seconds": 5,
                "state": "instrumental",
                "prompt": "No vocal performance, mouth closed, no lip movement.",
            }],
        )
        refs = [ResolvedReference(
            label="<Picture 1>", kind="picture", source="actor.png",
            role="subject", description="A singer",
        )]

        output = generator._render_reference(request, plan, refs)

        self.assertEqual(2, len(calls))
        self.assertIn("active_vocal_language=True", calls[1]["notes"])
        self.assertIn("no singing or lip sync", output.detailed_description)

    def test_generator_components_have_dedicated_modules(self):
        self.assertEqual(VideoPromptGenerator.__module__, "feverslop.prompting.dspy_h3_generator")
        self.assertEqual(LocalImageAnalyzer.__module__, "feverslop.prompting.dspy_h3_analyzer")
        self.assertEqual(PromptMode.__module__, "feverslop.prompting.dspy_h3_models")
        self.assertTrue(callable(build_dspy_signatures))
        self.assertEqual(ReferenceAsset.__module__, "feverslop.prompting.dspy_h3_models")

    def test_dspy_signatures_resolve_nested_pydantic_output_types(self):
        analyze_image, *_ = build_dspy_signatures()

        self.assertIs(
            analyze_image.output_fields["analysis"].annotation,
            __import__("feverslop.prompting.dspy_h3_models", fromlist=["ImageAnalysis"]).ImageAnalysis,
        )

    def test_signatures_use_structured_inputs_instead_of_json_strings(self):
        _, build_plan, render_base, render_reference = build_dspy_signatures()

        self.assertNotIn("references_json", build_plan.input_fields)
        self.assertNotIn("plan_json", render_base.input_fields)
        self.assertNotIn("references_json", render_base.input_fields)
        self.assertNotIn("relay_segments_json", render_base.input_fields)
        self.assertNotIn("plan_json", render_reference.input_fields)
        self.assertNotIn("references_json", render_reference.input_fields)
        self.assertNotIn("relay_segments_json", render_reference.input_fields)

        for signature in (build_plan, render_base, render_reference):
            field = signature.input_fields["relay_segments"]
            self.assertFalse(field.is_required())
            self.assertEqual(field.default, [])

    def test_integrated_guides_are_bundled_with_prompting_package(self):
        guides = files("feverslop.prompting.guides")

        base = (guides / "minimax-h3-base.md").read_text(encoding="utf-8")
        reference = (guides / "minimax-h3-references.md").read_text(encoding="utf-8")

        self.assertIn("integrated_multimodal_description", base)
        self.assertIn("subject_definitions", reference)
        self.assertIn("retention_analysis", reference)

        self.assertTrue((guides / "krea-actor.md").is_file())
        self.assertTrue((guides / "krea-location.md").is_file())

    def test_scene_reference_roles_are_preserved_for_full_generator(self):
        references, _ = _scene_references(
            {
                "references": {
                    "actor_ids": ["actor"],
                    "actor_msr_paths": ["actor.png"],
                    "location_msr_path": "location.png",
                }
            },
            {"vocals": Path("vocals.wav")},
            None,
        )

        self.assertEqual(
            [reference["role"] for reference in references],
            ["subject", "environment", "audio_reuse"],
        )

    def test_scene_references_preserve_canonical_paths_and_dedupe_derived_refs(self):
        references, _ = _scene_references(
            {
                "references": {
                    "reference_image_paths": ["existing.png", "actor.png"],
                    "reference_video_paths": ["clip.mp4"],
                    "reference_audio_paths": ["existing.wav", "vocals.wav"],
                    "actor_ids": ["leo"],
                    "actor_msr_paths": ["actor.png"],
                    "location_id": "forest",
                    "location_msr_path": "forest.png",
                }
            },
            {"vocals": Path("vocals.wav"), "full_mix": Path("full_mix.wav")},
            None,
        )

        self.assertEqual(
            [(reference["kind"], reference["source"]) for reference in references],
            [
                ("picture", "actor.png"),
                ("picture", "forest.png"),
                ("picture", "existing.png"),
                ("video", "clip.mp4"),
                ("audio", "full_mix.wav"),
                ("audio", "existing.wav"),
                ("audio", "vocals.wav"),
            ],
        )

    def test_build_request_propagates_r2v_canonical_references(self):
        generator = FakeGenerator()
        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={
                "segment_id": "seg-1",
                "references": {
                    "reference_image_paths": ["existing.png"],
                    "reference_video_paths": ["clip.mp4"],
                    "reference_audio_paths": ["existing.wav"],
                },
            },
            concept="A scene",
            scene_details={},
            global_context={},
            mode="r2v",
            audio_paths={"vocals": Path("vocals.wav")},
        )

        request = generator.requests[0]
        self.assertEqual(
            [(reference["kind"], reference["source"]) for reference in request["references"]],
            [
                ("picture", "existing.png"),
                ("video", "clip.mp4"),
                ("audio", "vocals.wav"),
                ("audio", "existing.wav"),
            ],
        )

    def test_build_request_is_valid_canonical_h3_payload_for_all_modes(self):
        image_paths = {
            "t2v": ["style.png"],
            "i2v": ["first.png"],
            "fl2v": ["first.png", "last.png"],
            "l2v": ["last.png"],
            "r2v": ["reference.png"],
        }
        for mode, paths in image_paths.items():
            generator = FakeGenerator()
            DspyH3PromptBuilder(generator).build_h3_prompt(
                segment={
                    "segment_id": "seg-1",
                    "references": {
                        "reference_image_paths": paths,
                        "reference_video_paths": ["motion.mp4"],
                        "reference_audio_paths": ["scene.wav"],
                    },
                },
                concept="A scene",
                scene_details={},
                global_context={},
                mode=mode,
            )

            request = VideoPromptRequest.model_validate(generator.requests[0])
            self.assertTrue(all(reference.description for reference in request.references))
            roles = [reference.role.value for reference in request.references]
            if mode == "i2v":
                self.assertEqual(["first_frame"], roles[:1])
            elif mode == "fl2v":
                self.assertEqual(["first_frame", "last_frame"], roles[:2])
            elif mode == "l2v":
                self.assertEqual("last_frame", roles[0])

    def test_reports_progress_after_each_scene(self):
        progress = []
        statuses = []
        builder = DspyH3PromptBuilder(FakeGenerator())

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.build_all_h3_prompts(
                stage1_segments=[
                    {"segment_id": "seg-1", "type": "vocals"},
                    {"segment_id": "seg-2", "type": "instrumental"},
                ],
                concept_prompts={"seg-1": "one", "seg-2": "two"},
                scene_details={},
                global_context={},
                output_json_path=Path(temp_dir) / "h3.json",
                artifact_store=JsonArtifactStore(),
                progress_callback=lambda current, total: progress.append((current, total)),
                status_callback=lambda current, total, status: statuses.append((current, total, status)),
            )

        self.assertEqual([(1, 2), (2, 2)], progress)
        self.assertEqual(
            [(1, 2, "started"), (1, 2, "completed"), (2, 2, "started"), (2, 2, "completed")],
            statuses,
        )

    def test_maps_scene_references_and_audio_stems_to_generator_request(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator)

        result = builder.build_h3_prompt(
            segment={
                "segment_id": "seg-1",
                "type": "vocals",
                "lyrics": "Ein Lied",
                "references": {
                    "actor_ids": ["elara"],
                    "location_id": "tavern",
                    "actor_msr_paths": ["movie/references/elara.png"],
                    "location_msr_path": "movie/references/tavern.png",
                },
            },
            concept="A singer in a tavern",
            scene_details={"camera_motion": "slow push in"},
            global_context={"style": "cinematic", "story_idea": "loss"},
            mode="ref",
            audio_paths={"vocals": Path("output/stems/vocals.wav")},
        )

        request = generator.requests[0]
        references = request["references"]
        self.assertEqual([ref["source"] for ref in references[:2]], [
            "movie/references/elara.png",
            "movie/references/tavern.png",
        ])
        self.assertEqual([ref["label"] for ref in references[:2]], [
            "<Picture 1>",
            "<Picture 2>",
        ])
        self.assertEqual(references[2]["label"], "<Audio 1>")
        self.assertEqual(request["music_intent"], "none")
        self.assertEqual(result["prompt"], FakeGeneratedPrompt.rendered_prompt)

    def test_appends_opt_in_reference_contract_without_global_band_rules(self):
        builder = DspyH3PromptBuilder(FakeGenerator())
        result = builder.build_h3_prompt(
            segment={
                "segment_id": "concert-1",
                "reference_profile": "live_concert",
                "references": {
                    "actor_ids": ["singer", "drummer"],
                    "actor_msr_paths": ["singer.png", "drummer.png"],
                    "location_msr_path": "stage.png",
                    "actor_reference_descriptions": [
                        {"id": "singer", "name": "Singer", "role": "Lead singer"},
                        {"id": "drummer", "name": "Drummer", "role": "Drummer"},
                    ],
                    "prop_bindings": {
                        "Singer": ["microphone"],
                        "Drummer": ["drum kit"],
                    },
                },
            },
            concept="A band performs on stage.",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertIn("exactly one persistent physical individual", result["prompt"])
        self.assertIn("main festival stage", result["prompt"])
        self.assertIn("Singer remains bound to microphone", result["prompt"])
        self.assertIn("Drummer remains bound to drum kit", result["prompt"])

    def test_generic_profile_does_not_receive_live_concert_contract(self):
        result = DspyH3PromptBuilder(FakeGenerator()).build_h3_prompt(
            segment={
                "segment_id": "tavern-1",
                "reference_profile": "crowded_tavern",
                "references": {
                    "actor_ids": ["singer"],
                    "actor_msr_paths": ["singer.png"],
                    "location_msr_path": "tavern.png",
                    "actor_reference_descriptions": [
                        {"id": "singer", "name": "Singer", "role": "Lead singer"},
                    ],
                    "prop_bindings": {"Singer": ["microphone"]},
                },
            },
            concept="A singer performs in a crowded tavern.",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertIn("exactly one persistent physical individual", result["prompt"])
        self.assertNotIn("catwalk", result["prompt"].lower())
        self.assertNotIn("main festival stage", result["prompt"].lower())

    def test_does_not_force_music_mode_without_scene_audio(self):
        generator = FakeGenerator()

        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={"segment_id": "seg-1"},
            concept="A silent scene",
            scene_details={},
            global_context={},
        )

        self.assertNotIn("music_intent", generator.requests[0])

    def test_resolves_existing_picture_paths_only_for_generator(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator, reference_root=Path.cwd())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            picture = root / "output" / "actor.png"
            picture.parent.mkdir()
            picture.write_bytes(b"not-an-image")
            builder = DspyH3PromptBuilder(generator, reference_root=root)
            result = builder.build_h3_prompt(
                segment={
                    "segment_id": "seg-1",
                    "references": {"actor_msr_paths": ["output/actor.png"]},
                },
                concept="A singer",
                scene_details={},
                global_context={},
            )

        self.assertEqual(generator.requests[-1]["references"][0]["source"], str(picture))
        self.assertEqual(result["references"][0]["source"], "output/actor.png")

    def test_falls_back_to_existing_prompt_when_generator_fails(self):
        class BrokenGenerator:
            def __call__(self, request):
                raise RuntimeError("DSPy unavailable")

        builder = DspyH3PromptBuilder(BrokenGenerator())
        result = builder.build_h3_prompt(
            segment={"segment_id": "seg-1", "type": "instrumental"},
            concept="fallback scene",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertEqual(result["prompt"], "fallback scene")
        self.assertEqual(result["dspy_error"], "DSPy unavailable")

    def test_sanitizes_embedded_image_data_in_fallback_error(self):
        payload = "data:image/png;base64," + ("A" * 400)

        class BrokenGenerator:
            def __call__(self, request):
                raise RuntimeError(payload)

        result = DspyH3PromptBuilder(BrokenGenerator()).build_h3_prompt(
            segment={"segment_id": "seg-1"},
            concept="fallback scene",
            scene_details={},
            global_context={},
        )

        self.assertNotIn("data:image", result["dspy_error"])
        self.assertNotIn("A" * 100, result["dspy_error"])
        self.assertIn("embedded image omitted", result["dspy_error"])

    def test_production_mode_does_not_hide_dspy_failure(self):
        class BrokenGenerator:
            def __call__(self, request):
                raise RuntimeError("connection refused")

        with self.assertRaisesRegex(RuntimeError, "DSPy H3 generation failed: connection refused"):
            DspyH3PromptBuilder(BrokenGenerator(), allow_fallback=False).build_h3_prompt(
                segment={"segment_id": "seg-1"},
                concept="fallback scene",
                scene_details={},
                global_context={},
            )


if __name__ == "__main__":
    unittest.main()
