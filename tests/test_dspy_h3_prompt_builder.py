import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from typing import Any
from unittest.mock import patch

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.h3_prompt_checkpoints import H3PromptCheckpointStore
from feverslop.adapters.movie_minimax_visual import _h3_movie_prompt
from feverslop.prompting.dspy_h3_analyzer import LocalImageAnalyzer
from feverslop.prompting.dspy_h3_generator import VideoPromptGenerator
from feverslop.prompting.dspy_h3_generator_core import (
    VideoPromptGenerator as CoreVideoPromptGenerator,
)
from feverslop.prompting.dspy_h3_models import (
    CreativeFieldIssue,
    H3CreativePlan,
    H3CreativeShot,
    MusicIntent,
    PlannedShot,
    PlannedSubject,
    PromptMode,
    PromptJudgeResult,
    PromptPlan,
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
from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    _format_relay_shots,
    _normalize_relay_segments,
    _scene_references,
    _speaker_bindings_for_compile,
    _stamp_relay_speaker_binding,
)
from feverslop.prompting.scene_prompt_builder import normalize_scene_references
from feverslop.prompting.dspy_h3_signatures import build_dspy_signatures, build_h3_signature_bundle


class FakeGeneratedPrompt:
    rendered_prompt = "subject_definitions: <Subject 1>\ndetailed_description: test"
    judge = PromptJudgeResult(verdict="good")


class FakeGenerator:
    def __init__(self, result=None):
        self.requests = []
        self.result = result or FakeGeneratedPrompt()

    def __call__(self, request):
        self.requests.append(request)
        return self.result


class CallbackGenerator(FakeGenerator):
    def set_warning_callback(self, callback):
        self.warning_callback = callback


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
    def test_reconstructs_generic_speaker_bindings_for_stale_checkpoints(self):
        bindings = _speaker_bindings_for_compile(
            segment={"references": {
                "actor_ids": ["hero"],
                "audio_subject_bindings": {
                    "vocals": {"subject_id": "hero", "speaker_id": "S1"},
                },
            }},
            references=[{
                "label": "<Audio 1>", "kind": "audio", "name": "vocals",
            }],
            stored_bindings=[],
        )

        self.assertEqual([{
            "audio_label": "<Audio 1>", "stem": "vocals",
            "subject_label": "<Subject 1>", "speaker_id": "S1",
        }], bindings)

    def test_rejects_stale_speaker_bindings_that_conflict_with_current_config(self):
        with self.assertRaisesRegex(ValueError, "conflicts with current config"):
            _speaker_bindings_for_compile(
                segment={"references": {
                    "actor_ids": ["hero"],
                    "audio_subject_bindings": {
                        "vocals": {"subject_id": "hero", "speaker_id": "S1"},
                    },
                }},
                references=[{
                    "label": "<Audio 1>", "kind": "audio", "name": "vocals",
                }],
                stored_bindings=[{
                    "audio_label": "<Audio 1>", "stem": "vocals",
                    "subject_label": "<Subject 1>", "speaker_id": "S2",
                }],
            )

    def test_rejects_stored_speaker_binding_removed_from_current_config(self):
        with self.assertRaisesRegex(ValueError, "conflicts with current config"):
            _speaker_bindings_for_compile(
                segment={"references": {"actor_ids": ["hero"]}},
                references=[],
                stored_bindings=[{
                    "subject_label": "<Subject 1>", "speaker_id": "S1",
                }],
            )

    def test_rejects_duplicate_speaker_ids_in_audio_bindings(self):
        with self.assertRaisesRegex(ValueError, "speaker ID S1"):
            _scene_references(
                {
                    "references": {
                        "actor_ids": ["hero", "companion"],
                        "reference_audio_paths": ["voice_a.wav", "voice_b.wav"],
                        "_stem_audio_tags": {
                            "voice_a.wav": "voice_a stem",
                            "voice_b.wav": "voice_b stem",
                        },
                        "audio_subject_bindings": {
                            "voice_a": {"subject_id": "hero", "speaker_id": "S1"},
                            "voice_b": {"subject_id": "companion", "speaker_id": "S1"},
                        },
                    },
                },
                None,
                None,
            )

    def test_rejects_duplicate_speaker_ids_from_current_relay_bindings(self):
        with self.assertRaisesRegex(ValueError, "speaker ID S1"):
            _speaker_bindings_for_compile(
                segment={
                    "references": {"actor_ids": ["hero", "companion"]},
                    "prompt_relay": [
                        {"subject_label": "<Subject 1>", "speaker_id": "S1"},
                        {"subject_label": "<Subject 2>", "speaker_id": "S1"},
                    ],
                },
                references=[],
                stored_bindings=[],
            )

    def test_audio_subject_bindings_are_serialized_without_inventing_full_mix_subject(self):
        from feverslop.prompting.dspy_h3_prompt_builder import _scene_references

        references, _images = _scene_references(
            {
                "references": {
                    "actor_ids": ["singer", "drummer"],
                    "reference_audio_paths": ["vocals.wav", "drums.wav", "song.wav"],
                    "_stem_audio_tags": {
                        "vocals.wav": "audio_transfer - vocal singing lip-synced to the audio signal",
                        "drums.wav": "drums stem",
                        "song.wav": "full_mix - original song for beat and rhythm continuity",
                    },
                    "audio_subject_bindings": {
                        "vocals": {"subject_id": "singer", "speaker_id": "S1"},
                        "drums": {"subject_id": "drummer"},
                    },
                },
            },
            None,
            None,
        )
        audio = {item["name"]: item for item in references if item["kind"] == "audio"}
        self.assertIn("<Subject 1> (S1)", audio["vocals"]["description"])
        self.assertIn("<Subject 2>", audio["drums"]["description"])
        full_mix = [item for item in audio.values() if "full_mix" in item["description"]]
        self.assertEqual(1, len(full_mix))
        self.assertNotIn("Subject", full_mix[0]["description"])

    def test_vocal_binding_requires_speaker_id_and_known_subject(self):
        from feverslop.prompting.dspy_h3_prompt_builder import _scene_references

        with self.assertRaisesRegex(ValueError, "speaker_id"):
            _scene_references(
                {"references": {"actor_ids": ["singer"], "audio_subject_bindings": {"vocals": {"subject_id": "singer"}}}},
                {"vocals": Path("vocals.wav")},
                None,
            )

    def test_audio_binding_rejects_unknown_subject_stem_and_speaker(self):
        from feverslop.prompting.dspy_h3_prompt_builder import _scene_references

        base = {"actor_ids": ["singer"], "reference_audio_paths": ["vocals.wav"]}
        with self.assertRaisesRegex(ValueError, "known subject"):
            _scene_references({"references": {**base, "audio_subject_bindings": {"vocals": {"subject_id": "ghost", "speaker_id": "S1"}}}}, None, None)
        with self.assertRaisesRegex(ValueError, "unselected stem"):
            _scene_references({"references": {**base, "audio_subject_bindings": {"drums": {"subject_id": "singer"}}}}, None, None)
        with self.assertRaisesRegex(ValueError, "invalid speaker_id"):
            _scene_references({"references": {**base, "audio_subject_bindings": {"vocals": {"subject_id": "singer", "speaker_id": "speaker-1"}}}}, None, None)

    def test_structured_audio_binding_is_rendered_in_prompt_sections(self):
        from feverslop.prompting.dspy_h3_models import AudioSubjectBinding

        prompt = ReferenceVideoPrompt(
            subject_definitions=[], summary="summary", retention_analysis=[], detailed_description="shot",
            overall_soundscape="music", audio_subject_bindings=[
                AudioSubjectBinding(audio_label="<Audio 1>", stem="vocals", subject_label="<Subject 1>", speaker_id="S1"),
            ],
        ).render()
        self.assertIn("<Audio 1> (vocals) -> <Subject 1> (S1)", prompt)
        self.assertIn("Audio subject bindings:", prompt)

    def test_checkpoint_revision_hashes_bundled_guides_and_judge_contract(self):
        generator = FakeGenerator()
        generator.base_guide_path = "minimax-h3-base.md"
        generator.reference_guide_path = "minimax-h3-references.md"
        generator.judge_attempts = 5

        revision = DspyH3PromptBuilder(generator).checkpoint_revision()

        self.assertEqual(3, revision["contract"])
        self.assertEqual(28, revision["compiler_version"])
        self.assertEqual(5, revision["judge_attempts"])
        self.assertRegex(revision["base_guide_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(revision["reference_guide_sha256"], r"^[0-9a-f]{64}$")

    def test_writes_each_checkpoint_before_generating_the_next_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            calls = []

            def generate(_request):
                calls.append(len(calls) + 1)
                if len(calls) == 2:
                    checkpoint = project / "output/render/scenes/scene_0001/h3_prompt.json"
                    self.assertTrue(checkpoint.is_file())
                return FakeGeneratedPrompt()

            DspyH3PromptBuilder(generate).build_all_h3_prompts(
                stage1_segments=[
                    {"scene": 1, "segment_id": "seg-1"},
                    {"scene": 2, "segment_id": "seg-2"},
                ],
                concept_prompts={"seg-1": "one", "seg-2": "two"},
                scene_details={},
                global_context={},
                output_json_path=project / "h3.json",
                artifact_store=JsonArtifactStore(),
                checkpoint_store=H3PromptCheckpointStore(project),
                generator_revision={"guide": "v1"},
            )

            self.assertEqual([1, 2], calls)

    def test_interruption_keeps_only_completed_scene_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            calls = 0

            def generate(_request):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("provider interrupted")
                return FakeGeneratedPrompt()

            with self.assertRaisesRegex(RuntimeError, "provider interrupted"):
                DspyH3PromptBuilder(generate, allow_fallback=False).build_all_h3_prompts(
                    stage1_segments=[
                        {"scene": number, "segment_id": f"seg-{number}"}
                        for number in range(1, 6)
                    ],
                    concept_prompts={},
                    scene_details={},
                    global_context={},
                    output_json_path=project / "h3.json",
                    artifact_store=JsonArtifactStore(),
                    checkpoint_store=H3PromptCheckpointStore(project),
                    generator_revision={"guide": "v1"},
                )

            checkpoints = sorted(project.glob("output/render/scenes/*/h3_prompt.json"))
            self.assertEqual(3, len(checkpoints))
            self.assertFalse((project / "h3.json").exists())

    def test_matching_checkpoint_resumes_without_generator_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            checkpoint_store = H3PromptCheckpointStore(project)
            kwargs = {
                "stage1_segments": [{"scene": 1, "segment_id": "seg-1"}],
                "concept_prompts": {"seg-1": "one"},
                "scene_details": {},
                "global_context": {},
                "output_json_path": project / "h3.json",
                "artifact_store": JsonArtifactStore(),
                "checkpoint_store": checkpoint_store,
                "generator_revision": {"guide": "v1"},
            }
            DspyH3PromptBuilder(FakeGenerator()).build_all_h3_prompts(**kwargs)
            generator = FakeGenerator()

            DspyH3PromptBuilder(generator).build_all_h3_prompts(**kwargs)

            self.assertEqual([], generator.requests)
            aggregate = json.loads((project / "h3.json").read_text(encoding="utf-8"))
            self.assertEqual("seg-1", aggregate[0]["segment_id"])

    def test_selected_generation_replaces_only_matching_legacy_aggregate_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            aggregate_path = project / "h3.json"
            aggregate_path.write_text(json.dumps([
                {"segment_id": "seg-1", "prompt": "keep unchanged"},
                {"segment_id": "seg-2", "prompt": "replace me"},
            ]), encoding="utf-8")

            DspyH3PromptBuilder(FakeGenerator()).build_all_h3_prompts(
                stage1_segments=[{"scene": 2, "segment_id": "seg-2"}],
                concept_prompts={"seg-2": "two"},
                scene_details={},
                global_context={},
                output_json_path=aggregate_path,
                artifact_store=JsonArtifactStore(),
                checkpoint_store=H3PromptCheckpointStore(project),
                generator_revision={"guide": "v1"},
                preserve_existing_aggregate=True,
            )

            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual("keep unchanged", aggregate[0]["prompt"])
            self.assertEqual(FakeGeneratedPrompt.rendered_prompt, aggregate[1]["prompt"])

    def test_selected_regeneration_replaces_selected_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            store = H3PromptCheckpointStore(project)
            base_kwargs = {
                "concept_prompts": {"seg-1": "one", "seg-2": "two"},
                "scene_details": {},
                "global_context": {},
                "output_json_path": project / "h3.json",
                "artifact_store": JsonArtifactStore(),
                "checkpoint_store": store,
                "generator_revision": {"guide": "v1"},
            }
            DspyH3PromptBuilder(FakeGenerator()).build_all_h3_prompts(
                stage1_segments=[
                    {"scene": 1, "segment_id": "seg-1"},
                    {"scene": 2, "segment_id": "seg-2"},
                ],
                **base_kwargs,
            )
            first_path = project / "output/render/scenes/scene_0001/h3_prompt.json"
            second_path = project / "output/render/scenes/scene_0002/h3_prompt.json"
            first_before = first_path.read_bytes()
            second_before = second_path.read_bytes()
            generator = FakeGenerator(type("Generated", (), {"rendered_prompt": "regenerated scene two"})())

            DspyH3PromptBuilder(generator).build_all_h3_prompts(
                stage1_segments=[{"scene": 2, "segment_id": "seg-2"}],
                preserve_existing_aggregate=True,
                reuse_checkpoints=False,
                **base_kwargs,
            )

            self.assertEqual(first_before, first_path.read_bytes())
            self.assertNotEqual(second_before, second_path.read_bytes())
            self.assertEqual(1, len(generator.requests))

    def test_build_all_forwards_reporter_warning_callback_to_generator(self):
        generator = CallbackGenerator()
        builder = DspyH3PromptBuilder(generator)
        warnings = []

        class Store:
            def write_json(self, _path, payload):
                return payload

        builder.build_all_h3_prompts(
            stage1_segments=[{"segment_id": "seg-1"}],
            concept_prompts={}, scene_details={}, global_context={},
            mode="ref", output_json_path="prompts.json", artifact_store=Store(),
            warning_callback=lambda text, title=None: warnings.append((title, text)),
        )

        self.assertIsNotNone(generator.warning_callback)

    def test_preserves_valid_prompt_when_judge_repair_generation_fails(self):
        from types import SimpleNamespace

        plan = ResolvedPromptPlan(
            creative_intent="A short performance.",
            style_opening="Live-action cinematic imagery uses cool practical lighting.",
            shots=[PlannedShot(
                shot_number=1,
                description="A performer turns toward the window.",
                start_seconds=0,
                end_seconds=5,
            )],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )
        warnings = []

        class RepairFailingGenerator:
            judge_attempts = 2

            def __init__(self):
                self.calls = 0
                self.warning_callback = None

            def __call__(self, request):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("planner unavailable")
                return SimpleNamespace(plan=plan)

            def set_warning_callback(self, callback):
                self.warning_callback = callback

            def _warning(self, message, *, title="H3 warning"):
                if self.warning_callback is not None:
                    self.warning_callback(message, title=title)

            def judge_compiled_prompt(self, **_kwargs):
                return PromptJudgeResult(
                    verdict="bad",
                    issues=["shot 1 lacks camera motion"],
                )

        generator = RepairFailingGenerator()

        class Store:
            def write_json(self, _path, payload):
                return payload

        result = DspyH3PromptBuilder(generator, allow_fallback=False).build_all_h3_prompts(
            stage1_segments=[{"segment_id": "seg-1", "duration": 5}],
            concept_prompts={"seg-1": "A performer turns toward the window."},
            scene_details={},
            global_context={},
            mode="r2v",
            output_json_path="prompts.json",
            artifact_store=Store(),
            warning_callback=lambda text, title=None: warnings.append((title, text)),
        )

        self.assertEqual(2, generator.calls)
        self.assertEqual([
            ("H3 compiled prompt judge retry",
             "H3 compiled prompt judge retry 2/2: shot 1 lacks camera motion"),
            ("H3 judge repair",
             "H3 judge repair was rejected; preserving the valid compiled prompt "
             "and BAD verdict: planner unavailable"),
        ], warnings)
        self.assertEqual(1, len(result))
        entry = result[0]
        self.assertEqual("seg-1", entry["segment_id"])
        self.assertNotEqual("", entry["prompt"])
        self.assertEqual("bad", entry["prompt_judge"]["verdict"])

    def test_reference_signature_requires_explicit_spatial_subject_placement(self):
        signature = build_dspy_signatures()[3]
        instructions = signature.__doc__ or ""

        self.assertIn("exact frame position", instructions)
        self.assertIn("Never place a subject inside an audience", instructions)
        self.assertIn("required prop", instructions)
        self.assertIn("Role-defining props", instructions)
        self.assertIn("not let another referenced subject inherit it", instructions)
        self.assertIn("preserve exactly one\n        persistent visible instance", instructions)
        self.assertIn("must not appear in two positions", instructions)
        self.assertIn("unless the resolved plan explicitly requires", instructions)
        self.assertIn("each such shot", instructions)
        self.assertIn("does not count as showing the actor", instructions)
        self.assertIn("visual shot", instructions)

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
                },
            },
            None,
            None,
        )

        self.assertEqual("A weathered hiker.", references[0]["description"])
        self.assertEqual("A dark ancient forest.", references[1]["description"])
        self.assertEqual("Leo", references[0]["name"])
        self.assertEqual("Ancient Forest", references[1]["name"])

    def test_scene_references_ignore_stale_actor_paths_for_explicit_empty_cast(self):
        references, _images = _scene_references(
            {
                "references": {
                    "actor_ids": [],
                    "actor_msr_paths": ["stale-singer.png"],
                    "location_msr_path": "crowd.png",
                },
            },
            None,
            None,
        )

        self.assertEqual(["crowd.png"], [reference["source"] for reference in references])
        self.assertEqual(["environment"], [reference["role"] for reference in references])

    def test_location_only_reference_normalization_preserves_empty_cast(self):
        references = normalize_scene_references(
            {"subject_mode": "location_only", "actor_ids": []},
            {"actors": [{"id": "leo"}], "structured_locations": [{"id": "forest"}]},
        )
        self.assertEqual("location_only", references["subject_mode"])
        self.assertEqual([], references["actor_ids"])

    def test_single_visible_actor_is_bound_to_vocal_stem_for_vocal_scene(self):
        references = normalize_scene_references(
            {"actor_ids": ["jack"]},
            {"actors": [{"id": "jack"}]},
            segment_type="vocals",
        )

        self.assertEqual(
            {"vocals": {"subject_id": "jack", "speaker_id": "S1"}},
            references["audio_subject_bindings"],
        )

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
                    },
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
        self.assertEqual("reference", audio[0]["copy_mode"])

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
                    },
                },
                None,
                root,
            )

        self.assertEqual("", references[0]["description"])
        self.assertEqual([picture], images)

    def test_passes_general_steering_and_prompt_guidance_to_generator(self):
        generator = FakeGenerator()
        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={
                "segment_id": "seg-1",
                "duration_seconds": 2,
                "fps": 24,
                "h3_creative_prompt": "A low tracking shot preserves the planned movement.",
            },
            concept="A singer performs.",
            scene_details={"character_motion": "Singer gestures toward the crowd."},
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
        self.assertNotIn("must be visible in at least one described shot", notes)
        self.assertIn("Character Motion: Singer gestures toward the crowd.", generator.requests[0]["user_prompt"])
        self.assertIn(
            "Existing backend-neutral scene motion prompt:\nA low tracking shot preserves the planned movement.",
            generator.requests[0]["user_prompt"],
        )

    def test_passes_source_language_metadata_without_inferring_from_names(self):
        generator = FakeGenerator()

        DspyH3PromptBuilder(generator).build_h3_prompt(
            segment={"segment_id": "seg-1"},
            concept="The singer says C'era Nocturna.",
            scene_details={},
            global_context={"language": "de"},
            mode="base",
        )

        notes = generator.requests[0]["notes"]
        self.assertIn('"source_language": "de"', notes)
        self.assertIn("do not infer language from proper names", notes.lower())

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
                ],
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

    def test_normalizes_structured_vocal_fields_without_dropping_them(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 2,
            "fps": 24,
            "ltx": {"prompt_relay": [{
                "frame_start": 0, "frame_end": 48, "state": "dialogue",
                "prompt": "structured dialogue event",
                "dialogue": "Systems ready",
                "subject_label": "<Subject 2>",
                "speaker_id": "S2",
            }]},
        })

        self.assertEqual("Systems ready", shots[0]["dialogue"])
        self.assertEqual("<Subject 2>", shots[0]["subject_label"])
        self.assertEqual("S2", shots[0]["speaker_id"])

    def test_stamps_singing_relay_window_to_vocal_subject(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 6.0,
            "fps": 24,
            "ltx": {"prompt_relay": [
                {"frame_start": 0, "frame_end": 72, "state": "singing", "prompt": "The singer turns."},
                {"frame_start": 72, "frame_end": 144, "state": "instrumental", "prompt": "The camera pulls back."},
            ]},
        })
        _stamp_relay_speaker_binding(
            shots,
            {"vocals": {"subject_label": "<Subject 1>", "speaker_id": "S1", "subject_id": "hero"}},
        )

        self.assertEqual("<Subject 1>", shots[0]["subject_label"])
        self.assertEqual("S1", shots[0]["speaker_id"])
        self.assertNotIn("subject_label", shots[1])
        self.assertNotIn("speaker_id", shots[1])

    def test_stamping_only_affects_singing_states(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 6.0,
            "fps": 24,
            "ltx": {"prompt_relay": [
                {"frame_start": 0, "frame_end": 72, "state": "dialogue", "prompt": "structured line"},
                {"frame_start": 72, "frame_end": 144, "state": "instrumental", "prompt": "no vocals"},
            ]},
        })
        _stamp_relay_speaker_binding(
            shots,
            {"vocals": {"subject_label": "<Subject 1>", "speaker_id": "S1", "subject_id": "hero"}},
        )

        for shot in shots:
            self.assertNotIn("subject_label", shot)
            self.assertNotIn("speaker_id", shot)

    def test_stamping_is_noop_without_a_vocal_binding(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 3.0,
            "fps": 24,
            "ltx": {"prompt_relay": [
                {"frame_start": 0, "frame_end": 72, "state": "singing", "prompt": "The singer turns."},
            ]},
        })
        _stamp_relay_speaker_binding(shots, {})

        self.assertNotIn("subject_label", shots[0])
        self.assertNotIn("speaker_id", shots[0])

    def test_stamping_preserves_an_existing_subject_label(self):
        shots = _normalize_relay_segments({
            "duration_seconds": 3.0,
            "fps": 24,
            "ltx": {"prompt_relay": [
                {
                    "frame_start": 0, "frame_end": 72, "state": "singing",
                    "prompt": "The singer turns.",
                    "subject_label": "<Subject 2>", "speaker_id": "S2",
                },
            ]},
        })
        _stamp_relay_speaker_binding(
            shots,
            {"vocals": {"subject_label": "<Subject 1>", "speaker_id": "S1", "subject_id": "hero"}},
        )

        self.assertEqual("<Subject 2>", shots[0]["subject_label"])
        self.assertEqual("S2", shots[0]["speaker_id"])

    def test_passes_relay_segments_to_generator_without_appending_non_guide_sections(self):
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
        self.assertNotIn("Temporal shot directions:", result["prompt"])
        self.assertNotIn("[Shot 1, 0.00-6.40sec]", result["prompt"])
        self.assertIn("actor.png", " ".join(reference["source"] for reference in result["references"]))

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

    def test_generator_gives_the_structured_judge_enough_output_tokens(self):
        class Client:
            base_url = "http://your-llm-server.local/v1"
            api_key = "none-needed"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            temperature = 0.75
            max_tokens = 65536
            prompt_judge_max_tokens = 8192
            dspy_cache = False

        guides = files("feverslop.prompting.guides")
        with patch("dspy.LM") as lm_factory:
            VideoPromptGenerator(
                base_guide_path=guides / "minimax-h3-base.md",
                reference_guide_path=guides / "minimax-h3-references.md",
                llm=LLM(),
            )

        self.assertEqual(65536, lm_factory.call_args_list[0].kwargs["max_tokens"])
        self.assertEqual(8192, lm_factory.call_args_list[1].kwargs["max_tokens"])

    def test_reference_limits_use_plural_picture_field(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        generator.limits = ReferenceLimits()
        generator.image_analyzer = type("Analyzer", (), {"should_analyze": lambda *_: False})()

        resolved = generator._resolve_references([
            ReferenceAsset(kind=ReferenceKind.PICTURE, source="actor.png", description="actor"),
            ReferenceAsset(kind=ReferenceKind.PICTURE, source="location.png", description="location"),
        ])

        self.assertEqual([item.label for item in resolved], ["<Picture 1>", "<Picture 2>"])

    def test_h3_creative_plan_schema_contains_no_compiler_owned_reference_fields(self):
        from feverslop.prompting import dspy_h3_models

        plan_type = getattr(dspy_h3_models, "H3CreativePlan", None)

        self.assertIsNotNone(plan_type)
        self.assertNotIn("subjects", plan_type.model_fields)
        self.assertNotIn("reference_usage", plan_type.model_fields)
        self.assertNotIn("continuation_intents", plan_type.model_fields)
        self.assertNotIn("alignment_instruction", plan_type.model_fields)
        for field in ("shot_number", "start_seconds", "end_seconds", "hard_cut_after"):
            self.assertNotIn(field, H3CreativeShot.model_fields)

    def test_h3_creative_fields_reject_compiler_owned_syntax(self):
        for description in (
            "The performer enters beside <Picture 99>.",
            "The camera cuts at 00:03.500.",
        ):
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    H3CreativeShot(description=description)
        self.assertEqual(
            "The station clock reads 12:30.",
            H3CreativeShot(description="The station clock reads 12:30.").description,
        )

    def test_planner_rejects_llm_authored_shot_count_without_relay_structure(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        generator.planner = lambda **_: type("Prediction", (), {"plan": PromptPlan(
            creative_intent="Invented cuts.",
            shots=[
                PlannedShot(shot_number=1, description="First invented shot."),
                PlannedShot(shot_number=2, description="Second invented shot."),
            ],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )})()

        with self.assertRaisesRegex(ValueError, "authoritative scene structure"):
            generator._plan(VideoPromptRequest(
                mode=PromptMode.R2V,
                user_prompt="One planned scene",
                duration_seconds=5,
            ), [])

    def test_planner_uses_authoritative_relay_timing_instead_of_authored_timing(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        generator.planner = lambda **_: type("Prediction", (), {"plan": PromptPlan(
            creative_intent="Two ordered actions.",
            shots=[
                PlannedShot(
                    shot_number=9, start_seconds=20, end_seconds=21,
                    description="The performer raises one hand.",
                ),
                PlannedShot(
                    shot_number=4, start_seconds=30, end_seconds=31,
                    description="The performer lowers the hand.",
                ),
            ],
            overall_soundscape="Quiet room tone.",
            music_intent=MusicIntent.NONE,
        )})()
        request = VideoPromptRequest(
            mode=PromptMode.R2V,
            user_prompt="A scene",
            duration_seconds=5,
            relay_segments=[
                {"start_seconds": 0.0, "end_seconds": 2.0, "prompt": "raise"},
                {"start_seconds": 2.0, "end_seconds": 5.0, "prompt": "lower"},
            ],
        )

        result = generator._plan(request, [])

        self.assertEqual([1, 2], [shot.shot_number for shot in result.shots])
        self.assertEqual(
            [(0.0, 2.0), (2.0, 5.0)],
            [(shot.start_seconds, shot.end_seconds) for shot in result.shots],
        )

    def test_h3_planner_signature_requests_only_creative_scene_enrichment(self):
        from feverslop.prompting.dspy_h3_signatures import build_h3_signature_bundle

        instructions = build_h3_signature_bundle().build_prompt_plan.__doc__ or ""

        self.assertIn("creative MiniMax H3 shot prose only", instructions)
        self.assertIn("complete grammatical sentence", instructions)
        self.assertIn("350-500", instructions)
        self.assertNotIn("Map each subject", instructions)

    def test_planner_cannot_override_deterministic_reference_assignments(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        authored = PromptPlan(
            creative_intent="A singer crosses a fractured neon room.",
            subjects=[PlannedSubject(
                name="Invented duplicate",
                description="A wide shot instead of a subject definition",
                source_references=["<Picture 9>"],
            )],
            reference_usage=[ReferenceUsage(
                reference_label="<Audio 9>",
                purpose="invented",
                details="invented",
            )],
            shots=[PlannedShot(
                shot_number=1,
                description="The singer walks through the room.",
                start_seconds=0,
                end_seconds=5,
                involved_subjects=["Invented duplicate"],
                reference_labels=["<Picture 9>", "<Audio 9>"],
            )],
            overall_soundscape="Glass fragments click across the floor.",
            music_intent=MusicIntent.NONE,
        )
        calls = []
        generator.planner = lambda **kwargs: (
            calls.append(kwargs)
            or type("Prediction", (), {"plan": authored.model_copy(deep=True)})()
        )
        request = VideoPromptRequest(
            mode=PromptMode.R2V,
            user_prompt="A scene",
            duration_seconds=5.0,
        )
        references = [
            ResolvedReference(
                label="<Picture 1>", kind="picture", source="actor.png",
                role="subject", name="Jack", description="A dark-haired singer in a silver coat",
            ),
            ResolvedReference(
                label="<Picture 2>", kind="picture", source="room.png",
                role="environment", name="Glitch Room", description="A fractured neon room",
            ),
            ResolvedReference(
                label="<Audio 1>", kind="audio", source="vocals.wav",
                role="audio_reuse", name="vocals", description="The scene vocal stem",
            ),
        ]

        result = generator._plan(request, references)

        self.assertEqual(1, len(calls))
        self.assertEqual(["Jack", "Glitch Room"], [subject.name for subject in result.subjects])
        self.assertEqual(
            [["<Picture 1>"], ["<Picture 2>"]],
            [subject.source_references for subject in result.subjects],
        )
        self.assertEqual(["<Audio 1>"], [usage.reference_label for usage in result.reference_usage])
        self.assertEqual(
            ["<Picture 1>", "<Picture 2>", "<Audio 1>"],
            result.shots[0].reference_labels,
        )

    def test_unknown_planner_references_are_discarded_without_retry(self):
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

        result = generator._plan(request, references)
        self.assertEqual("Invalid", result.creative_intent)
        self.assertEqual(1, len(calls))
        self.assertEqual([["<Picture 1>"]], [subject.source_references for subject in result.subjects])
        self.assertEqual(["<Picture 1>"], result.shots[0].reference_labels)

    def test_does_not_retry_creative_planner_for_compiler_owned_reference_fields(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        plans = [
            PromptPlan(
                creative_intent="Invalid attempt",
                subjects=[PlannedSubject(
                    name="Singer",
                    description="A singer",
                    source_references=["<Picture 9>"],
                )],
                shots=[PlannedShot(shot_number=1, description="The singer performs.")],
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
                shots=[PlannedShot(shot_number=1, description="The singer performs.")],
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

        self.assertEqual("Invalid attempt", result.creative_intent)
        self.assertEqual(1, len(calls))
        self.assertEqual([["<Picture 1>"]], [subject.source_references for subject in result.subjects])

    def test_planner_retries_when_a_loaded_picture_is_not_mapped_to_a_subject(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        invalid = PromptPlan(
            creative_intent="Missing location",
            subjects=[PlannedSubject(
                name="Drummer",
                description="A drummer",
                source_references=["<Picture 1>"],
            )],
            shots=[PlannedShot(shot_number=1, description="The drummer performs.")],
            overall_soundscape="A song",
            music_intent=MusicIntent.NONE,
        )
        valid = PromptPlan(
            creative_intent="Mapped location",
            subjects=[
                PlannedSubject(name="Drummer", description="A drummer", source_references=["<Picture 1>"]),
                PlannedSubject(name="Stage", description="A black stage", source_references=["<Picture 2>"]),
            ],
            shots=[PlannedShot(shot_number=1, description="The drummer performs on stage.")],
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

    def test_compiler_maps_visual_subjects_without_planner_warning(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        invalid = PromptPlan(
            creative_intent="Performance",
            subjects=[],
            shots=[PlannedShot(shot_number=1, description="The singer performs over the reef.")],
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

        result = generator._plan(request, references)

        self.assertEqual(1, len(calls))
        self.assertEqual(
            [["<Picture 1>"], ["<Picture 2>"]],
            [subject.source_references for subject in result.subjects],
        )
        self.assertEqual(["Lead Singer", "The Azure Reef"], [subject.name for subject in result.subjects])

    def test_reference_renderer_retries_unknown_subject_with_mismatch_details(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        calls = []

        def renderer(**kwargs):
            calls.append(kwargs)
            subject = "<Subject 3>" if len(calls) == 1 else "<Subject 1>"
            return type("Output", (), {
                "summary": f"{subject} performs.",
                "retention_analysis": [RetentionAnalysis(
                    target_label="<Subject 1>", mode="fully_preserved", details="stable",
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
                    target_label="<Subject 1>", mode="fully_preserved", details="stable",
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

    def test_field_judge_repair_changes_only_addressed_creative_field(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        original = ResolvedPromptPlan(
            creative_intent="Performance",
            shots=[PlannedShot(
                shot_number=1,
                description="walks slowly",
                camera_behavior="shakes",
            )],
            overall_soundscape="Music.",
            music_intent=MusicIntent.NONE,
        )
        generator.planner = lambda **_: type("Prediction", (), {"plan": H3CreativePlan(
            creative_intent="Replacement",
            shots=[H3CreativeShot(
                description="must not replace this action",
                camera_behavior="slow pan",
            )],
            overall_soundscape="Music.",
            music_intent=MusicIntent.NONE,
        )})()
        issue = CreativeFieldIssue(
            shot_id="shot-0001",
            field="camera_behavior",
            issue_code="camera.invalid",
            repair_instruction="Use a slow pan.",
        )

        result = generator._repair_creative_plan(
            VideoPromptRequest(mode=PromptMode.R2V, user_prompt="A scene", duration_seconds=5),
            original,
            [],
            [issue],
        )

        self.assertEqual("slow pan", result.shots[0].camera_behavior)
        self.assertEqual("walks slowly", result.shots[0].description)
        self.assertEqual("Performance", result.creative_intent)

    def test_field_repair_ignores_invalid_unaddressed_candidate_fields(self):
        generator = object.__new__(CoreVideoPromptGenerator)
        original = ResolvedPromptPlan(
            creative_intent="Performance",
            shots=[PlannedShot(
                shot_number=1,
                description="walks slowly",
                camera_behavior="shakes",
            )],
            overall_soundscape="Music.",
            music_intent=MusicIntent.NONE,
        )
        generator.planner = lambda **_: type("Prediction", (), {"plan": PromptPlan(
            creative_intent="Replacement",
            shots=[PlannedShot(
                shot_number=1,
                description="<Picture 9> invalid backend syntax",
                camera_behavior="slow pan",
            )],
            overall_soundscape="Music.",
            music_intent=MusicIntent.NONE,
        )})()
        issue = CreativeFieldIssue(
            shot_id="shot-0001",
            field="camera_behavior",
            issue_code="camera.invalid",
            repair_instruction="Use a slow pan.",
        )

        result = generator._repair_creative_plan(
            VideoPromptRequest(mode=PromptMode.R2V, user_prompt="A scene", duration_seconds=5),
            original,
            [],
            [issue],
        )

        self.assertEqual("slow pan", result.shots[0].camera_behavior)
        self.assertEqual("walks slowly", result.shots[0].description)

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

    def test_judge_output_accepts_section_level_feedback_for_normalization(self):
        judge = build_h3_signature_bundle().judge_final_prompt

        self.assertEqual(dict[str, Any], judge.output_fields["judge"].annotation)

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
                },
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
                },
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

    def test_rejects_contradictory_locked_facts_before_generator_call(self):
        generator = FakeGenerator()
        builder = DspyH3PromptBuilder(generator)

        with self.assertRaisesRegex(ValueError, "source-a.*source-b"):
            builder.build_h3_prompt(
                segment={
                    "segment_id": "seg-1",
                    "locked_facts": [
                        {"category": "wardrobe", "key": "hero", "value": "coat", "source_id": "source-a"},
                        {"category": "wardrobe", "key": "hero", "value": "jacket", "source_id": "source-b"},
                    ],
                },
                concept="A hero waits.",
                scene_details={},
                global_context={},
            )

        self.assertEqual([], generator.requests)

    def test_keeps_reference_contract_out_of_guide_prompt(self):
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

        self.assertNotIn("Reference identity and continuity contract:", result["prompt"])
        self.assertNotIn("main festival stage", result["prompt"])
        self.assertNotIn("Singer remains bound to microphone", result["prompt"])
        self.assertNotIn("Drummer remains bound to drum kit", result["prompt"])
        self.assertNotIn("deterministic_contract", result)
        self.assertNotIn("reference_profile", result)

    def test_builder_does_not_reintroduce_inactive_actor_contracts(self):
        builder = DspyH3PromptBuilder(FakeGenerator())
        result = builder.build_h3_prompt(
            segment={
                "segment_id": "crowd-only",
                "reference_profile": "live_concert",
                "references": {
                    "actor_ids": [],
                    "actor_msr_paths": ["stale-singer.png"],
                    "location_msr_path": "crowd.png",
                    "actor_reference_descriptions": [
                        {"id": "singer", "name": "Singer", "role": "Lead singer"},
                    ],
                    "prop_bindings": {"singer": ["microphone"]},
                },
            },
            concept="A crowd fills the frame.",
            scene_details={},
            global_context={},
            mode="ref",
        )

        self.assertNotIn("Singer", result["prompt"])
        self.assertNotIn("microphone", result["prompt"])

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

        self.assertNotIn("Reference identity and continuity contract:", result["prompt"])
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
