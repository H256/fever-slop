import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.pipeline.render_plan_builder import (
    DetailListPicker,
    build_original_style_i2v_prompt,
    build_render_plan,
)
from feverslop.config.video_settings import VideoSettings


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
                    ]
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
                    ]
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

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            by_scene = {item["scene"]: item for item in plan}

            self.assertTrue(all(item["ltx"].get("original_style_i2v_prompt") for item in plan))
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
                        }
                    ]
                ),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps(
                    [
                        {
                            "scene": 1,
                            "prompt_relay": [
                                {"frame_start": 0, "frame_end": 48, "state": "singing", "lyrics": "do not sing this"}
                            ],
                        }
                    ]
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
        """Stem audio paths are merged into references.reference_audio_paths.
        This ensures prompt generators can produce <Audio N> tags for stems.
        """
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
                    }
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
            # Both stems should be in the audio paths (vocals first, full_mix second)
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
            # Verify semantic descriptions exist
            self.assertIn("vocals", scene["stem_audio"]["stems"])
            self.assertIn("full_mix", scene["stem_audio"]["stems"])
            # Check that vocals tag exists
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
                    }
                ]),
                encoding="utf-8",
            )
            relay_path.write_text(json.dumps([{"scene": 1, "prompt_relay": []}]), encoding="utf-8")

            # Config with stems in arbitrary order
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
            # Stem list should be reordered: vocals, full_mix first
            stems = scene["stem_audio"]["stems"]
            self.assertEqual(stems[0], "vocals")
            self.assertEqual(stems[1], "full_mix")
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
                        }
                    ]
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
                    }
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


if __name__ == "__main__":
    unittest.main()
