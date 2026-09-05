import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.duration_capability import DurationCapability
from feverslop.domain.canonical_render_plan import stable_scene_id
from feverslop.errors import FeverSlopDataError

# Access the private helper for direct unit testing
from feverslop.pipeline import render_plan_builder as rpb_module
from feverslop.pipeline.render_plan_builder import (
    DetailListPicker,
    _clamp_relay_segment,
    build_original_style_i2v_prompt,
    build_render_plan,
)


class DetailListPickerTests(unittest.TestCase):
    def test_random_pick_is_deterministic_for_scene_and_seed(self):
        picker = DetailListPicker(seed=123)
        items = ["one", "two", "three"]

        self.assertEqual(
            picker.pick("camera_motion", items, scene_number=4, strategy="random"),
            picker.pick("camera_motion", items, scene_number=4, strategy="random"),
        )

    def test_random_no_repeat_cycles_before_reusing_items(self):
        picker = DetailListPicker(seed=99)
        items = ["one", "two", "three"]

        first_cycle = [
            picker.pick("camera_motion", items, scene_number=i, strategy="random_no_repeat")
            for i in range(1, 4)
        ]
        fourth = picker.pick("camera_motion", items, scene_number=4, strategy="random_no_repeat")

        self.assertEqual(set(items), set(first_cycle))
        self.assertIn(fourth, items)


class OriginalStylePromptTests(unittest.TestCase):
    def test_relay_binding_preserves_generated_vocal_subject(self):
        binding = rpb_module._vocal_relay_binding({
            "vocal_performers": [{"subject_id": "mordren_vale", "speaker_id": "S1"}],
        })

        self.assertEqual({"subject_id": "mordren_vale", "speaker_id": "S1"}, binding)

    def test_vocal_prompt_contains_singing_but_no_silent_motion(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 1,
                "type": "vocals",
                "ltx_base_prompt": "A singer in a red jacket stands under neon rain.",
                "base_concept": "neon rain performance",
            },
            seed=7,
        ).lower()

        self.assertIn("sings", prompt)
        self.assertIn("lip sync", prompt)
        self.assertNotIn("no lip", prompt)
        self.assertLess(prompt.index("singer"), 80)

    def test_silent_mode_vocal_prompt_excludes_singing_terms(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 1,
                "type": "vocals",
                "silent_mode": True,
                "ltx_base_prompt": "A warrior tells the story through movement under neon rain.",
                "base_concept": "silent story scene",
            },
            seed=7,
        ).lower()

        self.assertIn("silent", prompt)
        self.assertIn("no lip", prompt)
        self.assertIn("no vocal performance", prompt)
        for banned in ("sings", "singing", "lip sync", "lip-sync"):
            self.assertNotIn(banned, prompt)

    def test_silent_mode_ignores_legacy_explicit_i2v_prompt_with_singing(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 1,
                "type": "vocals",
                "silent_mode": True,
                "zimage_prompt": "A warrior tells the story through movement under neon rain.",
                "i2v_prompt_from_t2i": "The warrior sings with expressive lip sync.",
                "base_concept": "silent story scene",
            },
            seed=7,
        ).lower()

        self.assertIn("no vocal performance", prompt)
        self.assertNotIn("the warrior sings with expressive lip sync", prompt)
        for banned in ("sings", "singing", "lip sync", "lip-sync"):
            self.assertNotIn(banned, prompt)

    def test_instrumental_prompt_excludes_singing_terms(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 2,
                "type": "instrumental",
                "ltx_base_prompt": "A guitarist waits beside a wall of amber lights.",
                "base_concept": "quiet instrumental break",
            },
            seed=7,
        ).lower()

        for banned in ("sing", "sings", "singing", "lip sync", "lip-sync"):
            self.assertNotIn(banned, prompt)

    def test_original_style_prompt_uses_startframe_prompt_as_visual_foundation(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 1,
                "type": "instrumental",
                "zimage_prompt": "A cinematic image of an old shaman kneeling among moss and roots.",
                "ltx_base_prompt": "The old shaman stands beside a gnarled tree and grips his staff.",
                "base_concept": "forest ritual",
            },
            seed=7,
        ).lower()

        self.assertIn("lock the first frame", prompt)
        self.assertIn("kneeling among moss and roots", prompt)
        self.assertNotIn("stands beside a gnarled tree", prompt)


class BuildRenderPlanTests(unittest.TestCase):
    def test_continuation_intents_are_materialized_with_profile_aligned_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            h3_path = temp / "h3.json"
            output_path = temp / "render_plan.json"
            scene_prompts_path.write_text(json.dumps([{
                "scene": 1,
                "segment_id": "scene-001",
                "type": "instrumental",
                "start": 10.0,
                "end": 18.0,
                "duration": 8.0,
                "zimage_prompt": "z",
                "ltx_base_prompt": "base",
            }]), encoding="utf-8")
            relay_path.write_text(json.dumps([{"scene": 1, "prompt_relay": []}]), encoding="utf-8")
            h3_path.write_text(json.dumps([{
                "segment_id": "scene-001",
                "continuation_intents": [{
                    "action_id": "orbit",
                    "requires_continuation": True,
                    "desired_duration_seconds": 8.0,
                }],
            }]), encoding="utf-8")

            build_render_plan(
                scene_prompts_path, relay_path, output_path,
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                h3_prompts_json=h3_path,
                duration_capability=DurationCapability.create(
                    fps=24, min_seconds=2.0, max_seconds=3.0,
                    preferred_seconds=3.0, frame_alignment=8, frame_offset=0,
                ),
            )

            scenes = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([1_001_001, 1_001_002, 1_001_003], [scene["scene"] for scene in scenes])
            self.assertEqual([f"{stable_scene_id("scene-001:continuation")}-{n:04d}" for n in (1, 2, 3)], [scene["segment_id"] for scene in scenes])
            group = scenes[0]["metadata"]["continuation_groups"][0]
            segments = group["segments"]
            self.assertEqual("scene-001:orbit", group["group_id"])
            self.assertEqual([f"{stable_scene_id("scene-001:continuation")}-{n:04d}" for n in (1, 2, 3)], [s["segment_id"] for s in segments])
            self.assertEqual(10.0, group["semantic_start_seconds"])
            self.assertEqual(18.0, group["semantic_end_seconds"])
            self.assertEqual(10.0, segments[0]["start_seconds"])
            self.assertEqual(18.0, segments[-1]["end_seconds"])
            self.assertTrue(all(s["duration_seconds"] <= 3.0 for s in segments))
            self.assertEqual([False, True, True], [s["starts_with_anchor"] for s in segments])

    def test_seed_minus_one_generates_and_persists_a_different_seed_per_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"
            scene_prompts_path.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "segment_id": "s1",
                        "type": "instrumental",
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 2.0,
                        "zimage_prompt": "z1",
                        "ltx_base_prompt": "scene one",
                    },
                    {
                        "scene": 2,
                        "segment_id": "s2",
                        "type": "instrumental",
                        "start": 2.0,
                        "end": 4.0,
                        "duration": 2.0,
                        "zimage_prompt": "z2",
                        "ltx_base_prompt": "scene two",
                    },
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps([
                    {"scene": 1, "prompt_relay": []},
                    {"scene": 2, "prompt_relay": []},
                ]),
                encoding="utf-8",
            )

            with patch("feverslop.pipeline.render_plan_builder.random.SystemRandom") as system_random:
                system_random.return_value.randint.side_effect = [111, 222]
                build_render_plan(
                    scene_prompts_json=scene_prompts_path,
                    ltx_prompt_relay_json=relay_path,
                    output_json_file=output_path,
                    video_settings=VideoSettings(fps=24, width=1280, height=704),
                    artifact_store=JsonArtifactStore(),
                    seed=-1,
                )

            self.assertEqual([111, 222], [scene["seed"] for scene in json.loads(output_path.read_text(encoding="utf-8"))])

    def test_render_plan_includes_original_style_prompt_and_mode_hints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"

            scene_prompts_path.write_text(
                json.dumps(
                    [
                        {
                            "scene": 1,
                            "segment_id": "s1",
                            "type": "vocals",
                            "start": 0.0,
                            "end": 2.0,
                            "duration": 2.0,
                            "lyrics": "hello",
                            "base_concept": "stage",
                            "zimage_prompt": "z1",
                            "ltx_base_prompt": "A singer faces the camera on a small stage.",
                        },
                        {
                            "scene": 16,
                            "segment_id": "s16",
                            "type": "vocals",
                            "start": 2.0,
                            "end": 4.0,
                            "duration": 2.0,
                            "lyrics": "mixed line",
                            "base_concept": "stage",
                            "zimage_prompt": "z16",
                            "ltx_base_prompt": "The same singer remains framed on the stage.",
                        },
                        {
                            "scene": 3,
                            "segment_id": "s3",
                            "type": "instrumental",
                            "start": 4.0,
                            "end": 6.0,
                            "duration": 2.0,
                            "lyrics": "",
                            "base_concept": "instrumental",
                            "zimage_prompt": "z3",
                            "ltx_base_prompt": "The performer holds still in a spotlight.",
                        },
                    ],
                ),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps(
                    [
                        {"scene": 1, "prompt_relay": []},
                        {
                            "scene": 16,
                            "prompt_relay": [
                                {"frame_start": 0, "frame_end": 20, "state": "singing", "lyrics": "mixed"},
                                {"frame_start": 20, "frame_end": 30, "state": "instrumental"},
                                {"frame_start": 30, "frame_end": 48, "state": "singing", "lyrics": "line"},
                            ],
                        },
                        {"scene": 3, "prompt_relay": []},
                    ],
                ),
                encoding="utf-8",
            )

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704, megapixels=0.98),
                artifact_store=JsonArtifactStore(),
                seed=4242,
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            by_scene = {item["scene"]: item for item in plan}

            self.assertTrue(all(item["ltx"].get("original_style_i2v_prompt") for item in plan))
            self.assertEqual({4242}, {item["seed"] for item in plan})
            self.assertEqual({0.98}, {item["megapixels"] for item in plan})
            self.assertEqual("single_prompt", by_scene[1]["ltx"]["render_mode_hint"])
            self.assertEqual("relay", by_scene[16]["ltx"]["render_mode_hint"])
            self.assertEqual("single_prompt", by_scene[3]["ltx"]["render_mode_hint"])
            self.assertNotIn("lip sync", by_scene[3]["ltx"]["original_style_i2v_prompt"].lower())

    def test_silent_mode_forces_vocal_relay_segments_to_silent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"

            scene_prompts_path.write_text(
                json.dumps(
                    [
                        {
                            "scene": 1,
                            "segment_id": "s1",
                            "type": "vocals",
                            "silent_mode": True,
                            "start": 0.0,
                            "end": 2.0,
                            "duration": 2.0,
                            "lyrics": "do not sing this",
                            "base_concept": "silent story action",
                            "zimage_prompt": "z",
                            "ltx_base_prompt": "The warrior acts out the story without singing.",
                        },
                    ],
                ),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps(
                    [
                        {
                            "scene": 1,
                            "prompt_relay": [
                                {"frame_start": 0, "frame_end": 48, "state": "singing", "lyrics": "do not sing this"},
                            ],
                        },
                    ],
                ),
                encoding="utf-8",
            )

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
            )

            scene = json.loads(output_path.read_text(encoding="utf-8"))[0]
            relay = scene["ltx"]["prompt_relay"][0]
            combined = json.dumps(scene["ltx"], ensure_ascii=False).lower()

        self.assertEqual("instrumental", relay["state"])
        self.assertIn("no vocal performance", relay["prompt"].lower())
        self.assertIn("silent", scene["ltx"]["original_style_i2v_prompt"].lower())
        for banned in ("sings", "singing", "lip sync", "lip-sync"):
            self.assertNotIn(banned, combined)


    def test_stem_audio_merged_into_reference_audio_paths(self):
        """Stem audio paths are merged into references.reference_audio_paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"
            vocal_path = temp / "vocals.wav"
            vocal_path.write_bytes(b"fake audio")
            fullmix_path = temp / "full_mix.wav"
            fullmix_path.write_bytes(b"fake audio")
            existing_path = temp / "input" / "reference.wav"
            existing_path.parent.mkdir()
            existing_path.write_bytes(b"fake audio")

            scene_prompts_path.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "segment_id": "s1",
                        "type": "vocals",
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 2.0,
                        "lyrics": "sing it",
                        "base_concept": "warrior",
                        "zimage_prompt": "z",
                        "ltx_base_prompt": "The warrior sings.",
                        "i2v_prompt_from_t2i": "I2V",
                        "references": {"reference_audio_paths": [str(existing_path)]},
                    },
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(json.dumps([{"scene": 1, "prompt_relay": []}]), encoding="utf-8")

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                stem_list=["vocals", "full_mix"],
                input_audio=fullmix_path,
                stem_files={"vocals": vocal_path, "full_mix": fullmix_path},
                project_dir=temp,
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            scene = plan[0]

            self.assertIn("stem_audio", scene)
            self.assertIn("references", scene)
            refs = scene["references"]
            self.assertIn("reference_audio_paths", refs)
            audio_paths = refs["reference_audio_paths"]
            self.assertEqual(
                ["input/reference.wav", "vocals.wav", "full_mix.wav"],
                audio_paths,
            )
            self.assertIn("_stem_audio_tags", refs)
            stem_tags = refs["_stem_audio_tags"]
            self.assertEqual({"vocals.wav", "full_mix.wav"}, set(stem_tags))
            self.assertEqual(
                {"vocals": "vocals.wav", "full_mix": "full_mix.wav"},
                scene["stem_audio"]["paths"],
            )
            self.assertIn("vocals", scene["stem_audio"]["stems"])
            self.assertIn("full_mix", scene["stem_audio"]["stems"])
            vocals_tag_found = any("audio_transfer" in v for v in stem_tags.values())
            self.assertTrue(vocals_tag_found, f"Expected audio_transfer in stem_tags, got {stem_tags}")

    def test_stem_audio_prioritizes_vocals_and_full_mix(self):
        """Stems list is reordered so vocals + full_mix come first."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"
            stems_dir = temp / "stems"
            stems_dir.mkdir()
            paths = {}
            for name in ["drums", "bass", "vocals", "full_mix"]:
                p = stems_dir / f"{name}.wav"
                p.write_bytes(b"fake")
                paths[name] = p

            scene_prompts_path.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "segment_id": "s1",
                        "type": "vocals",
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 2.0,
                        "lyrics": "test",
                        "base_concept": "beat",
                        "zimage_prompt": "z",
                        "ltx_base_prompt": "Test",
                        "i2v_prompt_from_t2i": "Test",
                    },
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(json.dumps([{"scene": 1, "prompt_relay": []}]), encoding="utf-8")

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                stem_list=["drums", "bass", "vocals", "full_mix"],
                input_audio=paths["full_mix"],
                stem_files=paths,
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            scene = plan[0]
            stems = scene["stem_audio"]["stems"]
            self.assertEqual(stems[0], "vocals")
            self.assertEqual(stems[1], "full_mix")

    def test_render_plan_routes_drummer_scene_to_drums_and_full_mix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"
            paths = {}
            for name in ("vocals", "drums", "bass", "other", "full_mix"):
                path = temp / f"{name}.wav"
                path.write_bytes(b"audio")
                paths[name] = path
            scene_prompts_path.write_text(json.dumps([{
                "scene": 1,
                "segment_id": "s1",
                "type": "instrumental",
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "lyrics": "",
                "base_concept": "The drummer performs.",
                "zimage_prompt": "A drummer performs.",
                "references": {"actor_reference_descriptions": [
                    {"name": "Drummer", "role": "Percussionist"},
                ]},
            }]), encoding="utf-8")
            relay_path.write_text(json.dumps([{
                "scene": 1,
                "prompt_relay": [{"frame_start": 0, "frame_end": 48, "state": "instrumental"}],
            }]), encoding="utf-8")

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                stem_list=["vocals", "full_mix"],
                input_audio=paths["full_mix"],
                stem_files=paths,
                project_dir=temp,
            )

            scene = json.loads(output_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(["drums", "full_mix"], scene["stem_audio"]["stems"])
            self.assertNotIn("vocals.wav", scene["references"]["reference_audio_paths"])

    def test_render_plan_persists_h3_performance_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts = temp / "scenes.json"
            relay = temp / "relay.json"
            h3 = temp / "h3.json"
            output = temp / "plan.json"
            scene_prompts.write_text(json.dumps([{
                "scene": 1, "segment_id": "s1", "type": "instrumental",
                "start": 0.0, "end": 2.0, "duration": 2.0,
                "zimage_prompt": "A drummer", "base_concept": "A drummer",
            }]), encoding="utf-8")
            relay.write_text(json.dumps([{"scene": 1, "prompt_relay": []}]), encoding="utf-8")
            timing = {"bpm": 120.0, "beats": [{"time_seconds": 0.5, "downbeat": True, "impact": 0.8}]}
            h3.write_text(json.dumps([{
                "segment_id": "s1",
                "prompt": "H3",
                "reference_profile": "live_concert",
                "performance_timing": timing,
            }]), encoding="utf-8")

            build_render_plan(
                scene_prompts_json=scene_prompts,
                ltx_prompt_relay_json=relay,
                output_json_file=output,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                h3_prompts_json=h3,
            )

            scene = json.loads(output.read_text(encoding="utf-8"))[0]
            self.assertEqual(timing, scene["performance_timing"])
            self.assertNotIn("reference_profile", scene["h3"])
    def test_render_plan_prefers_explicit_i2v_prompt_from_t2i(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"

            scene_prompts_path.write_text(
                json.dumps(
                    [
                        {
                            "scene": 1,
                            "segment_id": "s1",
                            "type": "instrumental",
                            "start": 0.0,
                            "end": 2.0,
                            "duration": 2.0,
                            "lyrics": "",
                            "base_concept": "mountain peak",
                            "zimage_prompt": "T2I PROMPT",
                            "t2i_prompt": "T2I PROMPT",
                            "ltx_base_prompt": "T2I PROMPT",
                            "i2v_prompt_from_t2i": "I2V FROM T2I",
                        },
                    ],
                ),
                encoding="utf-8",
            )
            relay_path.write_text(json.dumps([{ "scene": 1, "prompt_relay": [] }]), encoding="utf-8")

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("I2V FROM T2I", plan[0]["ltx"]["original_style_i2v_prompt"])
        self.assertEqual("T2I PROMPT", plan[0]["ltx"]["base_prompt"])

    def test_frame_counts_are_snapped_to_absolute_scene_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"

            scenes = []
            relay_scenes = []
            for scene_number, start, end in (
                (1, 0.0, 2.49),
                (2, 2.49, 4.98),
                (3, 4.98, 7.47),
            ):
                scenes.append(
                    {
                        "scene": scene_number,
                        "segment_id": f"s{scene_number}",
                        "type": "instrumental",
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 3),
                        "lyrics": "",
                        "base_concept": "stage",
                        "zimage_prompt": "z",
                        "ltx_base_prompt": "The performer remains still on stage.",
                    },
                )
                relay_scenes.append({"scene": scene_number, "prompt_relay": []})

            scene_prompts_path.write_text(json.dumps(scenes), encoding="utf-8")
            relay_path.write_text(json.dumps(relay_scenes), encoding="utf-8")

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual([60, 60, 59], [scene["frame_count"] for scene in plan])

    def test_duration_seconds_is_derived_from_scene_span_not_prompt_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"
            scene_prompts_path.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "segment_id": "s1",
                        "type": "instrumental",
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 3.0,
                        "zimage_prompt": "z1",
                        "ltx_base_prompt": "scene one",
                    },
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps([{"scene": 1, "prompt_relay": []}]),
                encoding="utf-8",
            )

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
            )

            entry = json.loads(output_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(entry["abs_start_seconds"], 0.0)
            self.assertEqual(entry["abs_end_seconds"], 2.0)
            self.assertEqual(entry["duration_seconds"], 2.0)

    def test_rejects_non_positive_scene_span(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"
            scene_prompts_path.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "segment_id": "s1",
                        "type": "instrumental",
                        "start": 5.0,
                        "end": 5.0,
                        "duration": 1.0,
                        "zimage_prompt": "z1",
                        "ltx_base_prompt": "scene one",
                    },
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps([{"scene": 1, "prompt_relay": []}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-positive scene span"):
                build_render_plan(
                    scene_prompts_json=scene_prompts_path,
                    ltx_prompt_relay_json=relay_path,
                    output_json_file=output_path,
                    video_settings=VideoSettings(fps=24, width=1280, height=704),
                    artifact_store=JsonArtifactStore(),
                )


class FrameBoundaryTests(unittest.TestCase):
    def test_clamp_relay_segment_exclusive_frame_end(self):
        result = _clamp_relay_segment(0, 48, 48)
        self.assertEqual((0, 48), result)

    def test_clamp_relay_segment_clamps_end_to_frame_count(self):
        result = _clamp_relay_segment(0, 100, 48)
        self.assertEqual((0, 48), result)

    def test_fallback_relay_uses_frame_count_not_minus_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"

            scene_prompts_path.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "segment_id": "s1",
                        "type": "instrumental",
                        "start": 0.0,
                        "end": 2.0,
                        "duration": 2.0,
                        "lyrics": "",
                        "base_concept": "test",
                        "zimage_prompt": "z",
                        "ltx_base_prompt": "A test prompt.",
                    },
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps([{"scene": 1, "prompt_relay": []}]),
                encoding="utf-8",
            )

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            frame_count = plan[0]["frame_count"]
            fb_end = plan[0]["ltx"]["prompt_relay"][0]["frame_end"]
            self.assertEqual(frame_count, fb_end)


class TestFeverSlopDataError(unittest.TestCase):
    def test_is_fever_slop_subclass(self):
        from feverslop.errors import FeverSlopError
        self.assertTrue(issubclass(FeverSlopDataError, FeverSlopError))


def _minimal_scene(scene_num=1, **overrides):
    """Build a minimal valid scene dict."""
    base = {
        "scene": scene_num,
        "duration": 2.0,
        "start": 0.0,
        "end": 2.0,
        "zimage_prompt": "a test scene",
        "segment_id": f"seg_{scene_num}",
        "type": "vocals",
        "lyrics": "",
        "base_concept": "",
        "camera_motion": "",
        "character_motion": "",
    }
    base.update(overrides)
    return base


def _minimal_relay(scene_num=1, **overrides):
    """Build a minimal relay scene dict."""
    base = {
        "scene": scene_num,
        "prompt_relay": [{
            "frame_start": 0,
            "frame_end": 60,
            "state": "singing",
            "lyrics": "",
            "text": "",
        }],
    }
    if overrides:
        base.update(overrides)
    return base


class TestDefensiveGuardsSceneKeys(unittest.TestCase):
    """Verify that missing scene keys raise FeverSlopDataError with context."""

    def _run_with_scenes(self, scenes, relays=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            store = JsonArtifactStore()
            settings = VideoSettings(width=1280, height=720, fps=30)

            scenes_file = tmp / "scenes.json"
            scenes_file.write_text(json.dumps(scenes))

            if relays is None:
                relays = [_minimal_relay(s.get("scene", 1)) for s in scenes]
            relay_file = tmp / "relay.json"
            relay_file.write_text(json.dumps(relays))

            output = tmp / "render_plan.json"
            return build_render_plan(
                scenes_file, relay_file, output, settings, artifact_store=store,
            )

    def test_missing_scene_key_raises_data_error(self):
        scene = _minimal_scene()
        del scene["scene"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene], [])
        self.assertIn("'scene'", str(ctx.exception))

    def test_missing_duration_raises_data_error(self):
        scene = _minimal_scene()
        del scene["duration"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene])
        self.assertIn("'duration'", str(ctx.exception))

    def test_missing_start_raises_data_error(self):
        scene = _minimal_scene()
        del scene["start"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene])
        self.assertIn("'start'", str(ctx.exception))

    def test_missing_end_raises_data_error(self):
        scene = _minimal_scene()
        del scene["end"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene])
        self.assertIn("'end'", str(ctx.exception))

    def test_missing_zimage_prompt_raises_data_error(self):
        scene = _minimal_scene()
        del scene["zimage_prompt"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene])
        self.assertIn("'zimage_prompt'", str(ctx.exception))

    def test_missing_segment_id_raises_data_error(self):
        scene = _minimal_scene()
        del scene["segment_id"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene])
        self.assertIn("'segment_id'", str(ctx.exception))

    def test_missing_type_raises_data_error(self):
        scene = _minimal_scene()
        del scene["type"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_scenes([scene])
        self.assertIn("'type'", str(ctx.exception))


class TestDefensiveGuardsRelayKeys(unittest.TestCase):
    """Verify that missing relay keys raise FeverSlopDataError with context."""

    def _run_with_relay(self, scenes, relays):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            store = JsonArtifactStore()
            settings = VideoSettings(width=1280, height=720, fps=30)

            (tmp / "scenes.json").write_text(json.dumps(scenes))
            (tmp / "relay.json").write_text(json.dumps(relays))
            output = tmp / "render_plan.json"
            return build_render_plan(
                tmp / "scenes.json", tmp / "relay.json", output, settings,
                artifact_store=store,
            )

    def test_missing_relay_frame_start_raises_data_error(self):
        relay = _minimal_relay()
        del relay["prompt_relay"][0]["frame_start"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_relay([_minimal_scene()], [relay])
        self.assertIn("'frame_start'", str(ctx.exception))

    def test_missing_relay_frame_end_raises_data_error(self):
        relay = _minimal_relay()
        del relay["prompt_relay"][0]["frame_end"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_relay([_minimal_scene()], [relay])
        self.assertIn("'frame_end'", str(ctx.exception))

    def test_missing_relay_state_raises_data_error(self):
        relay = _minimal_relay()
        del relay["prompt_relay"][0]["state"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_relay([_minimal_scene()], [relay])
        self.assertIn("'state'", str(ctx.exception))

    def test_missing_relay_scene_key_in_relay_by_scene(self):
        relay = _minimal_relay()
        del relay["scene"]
        with self.assertRaises(FeverSlopDataError) as ctx:
            self._run_with_relay([_minimal_scene()], [relay])
        self.assertIn("'scene'", str(ctx.exception))


class TestDefensiveGuardsValidInput(unittest.TestCase):
    """Verify valid input produces render plan without errors."""

    def test_valid_input_produces_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            store = JsonArtifactStore()
            settings = VideoSettings(width=1280, height=720, fps=30)

            scenes = [_minimal_scene(1)]
            relays = [_minimal_relay(1)]
            (tmp / "scenes.json").write_text(json.dumps(scenes))
            (tmp / "relay.json").write_text(json.dumps(relays))
            output = tmp / "render_plan.json"

            result = build_render_plan(
                tmp / "scenes.json", tmp / "relay.json", output, settings,
                artifact_store=store,
            )
            self.assertTrue(result.exists())
            data = json.loads(result.read_text())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["scene"], 1)

class VocalPromptWordBoundaryTests(unittest.TestCase):
    """Test that _contains_vocal_performance_prompt uses word boundaries."""

    def test_contains_vocal_performance_prompt_word_boundary(self):
        func = rpb_module._contains_vocal_performance_prompt
        # "things" should NOT match "sings"
        self.assertFalse(func("the singer shows things"))
        # "resigns" should NOT match "sings"
        self.assertFalse(func("the minister resigns today"))
        # "sings" should match
        self.assertTrue(func("she sings beautifully"))
        self.assertTrue(func("he sings"))
        # "singing" should match
        self.assertTrue(func("they are singing along"))

    def test_contains_vocal_performance_prompt_all_tokens_match(self):
        func = rpb_module._contains_vocal_performance_prompt
        self.assertTrue(func("she sings"))
        self.assertTrue(func("singing loud"))
        self.assertTrue(func("perfect lip sync"))
        self.assertTrue(func("a lip-sync performance"))
        self.assertTrue(func("lip-syncing is fun"))
        self.assertTrue(func("she belts out the chorus"))


if __name__ == "__main__":
    unittest.main()
