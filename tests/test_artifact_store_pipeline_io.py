import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.srt import SrtScene
from feverslop.pipeline.prompt_relay_builder import (
    build_scene_prompt_relay,
    lyrics_for_time_range,
)
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.pipeline.scene_duration_enforcer import write_scene_srt
from feverslop.pipeline.stage1_segment_builder import build_stage1_segment_json


class FakeArtifactStore:
    def __init__(self):
        self.json_reads = {}
        self.json_writes = {}
        self.text_writes = {}

    def read_json(self, path):
        return self.json_reads[str(path)]

    def write_json(self, path, data):
        self.json_writes[str(path)] = data
        return Path(path)

    def read_text(self, path):
        raise AssertionError("not used")

    def write_text(self, path, text):
        self.text_writes[str(path)] = text
        return Path(path)


class ArtifactStorePipelineIoTests(unittest.TestCase):
    def test_full_performance_path_preserves_intro_pause_cut_and_offscreen_voice(self):
        from feverslop.domain.locked_scene_facts import LockedSceneFacts
        from feverslop.prompting.dspy_h3_prompt_builder import _normalize_relay_segments
        from feverslop.prompting.deterministic_h3_compiler import (
            DeterministicH3Compiler, creative_shots_from_plan, plan_with_authoritative_relay,
        )
        from feverslop.prompting.dspy_h3_models import MusicIntent, PlannedShot, ResolvedPromptPlan

        with tempfile.TemporaryDirectory() as directory:
            srt = Path(directory) / "scenes.srt"
            write_scene_srt(srt, [SrtScene(scene=i, start=a, end=b, text="Orbit")
                                 for i, a, b in ((1, 0, 2), (2, 2, 5), (3, 5, 8))],
                            artifact_store=JsonArtifactStore())
            store = FakeArtifactStore()
            store.json_reads["timeline"] = [
                dict(type="instrumental", start=0, end=2),
                dict(type="vocals", start=2, end=8, word_timestamps=[
                    dict(word="I", start=2.98, end=3.02, source="whisper", word_id="tiny", speaker_id="S3"),
                    dict(word="First", start=2.5, end=3, source="whisper", word_id="w1", speaker_id="S1"),
                    dict(word="Hold", start=4, end=6, source="whisper", word_id="w2", speaker_id="S1"),
                    dict(word="Echo", start=6.5, end=7, source="whisper", word_id="w3", speaker_id="S2", offscreen=True),
                ]),
            ]
            build_stage1_segment_json(srt, "timeline", "stage1", artifact_store=store)
            store.json_reads["scenes"] = [dict(s, zimage_prompt="A slow camera orbit")
                                          for s in store.json_writes["stage1"]]
            for fps in (24, 25, 30):
                settings = VideoSettings(fps=fps, width=1280, height=704)
                build_scene_prompt_relay(srt, "timeline", "relay", settings, artifact_store=store)
                store.json_reads["relay"] = store.json_writes["relay"]
                build_render_plan("scenes", "relay", "render", settings, artifact_store=store)
                prompts = []
                for scene in store.json_writes["render"]:
                    phases = _normalize_relay_segments(scene)
                    duration = scene["duration_seconds"]
                    plan = ResolvedPromptPlan(creative_intent="Orbit", subjects=[], reference_usage=[],
                        shots=[PlannedShot(shot_number=1, start_seconds=0, end_seconds=duration,
                                          description="The camera circles slowly.")],
                        overall_soundscape="Air", music_intent=MusicIntent.NONE)
                    plan = plan_with_authoritative_relay(plan, phases)
                    prompt = DeterministicH3Compiler().compile(mode="r2v", plan=plan,
                        facts=LockedSceneFacts.create(scene_id=str(scene["scene"]), facts=[]),
                        shots=creative_shots_from_plan(plan), shot_windows={"shot-01": (0, duration)},
                        relay_segments=phases, duration_seconds=duration)
                    prompts.append(prompt.split("detailed_description:", 1)[1].split("non_diegetic_music:", 1)[0])
                self.assertNotIn("<d>", prompts[0])
                self.assertNotIn("Hold", prompts[1])
                self.assertIn("Hold", prompts[2])
                self.assertIn("(S2)", prompts[2])
                self.assertIn("sings offscreen", prompts[2])
                self.assertEqual(4, " ".join(prompts).count("<d>"), prompts)
                self.assertTrue(all("[Shot 2]" not in p for p in prompts))
                self.assertTrue(all("camera circles slowly" in p for p in prompts))
                self.assertIn("No sung vocal performance occurs", prompts[1])

    def test_prompt_relay_uses_only_available_timestamped_words(self):
        result = lyrics_for_time_range(
            "Ich trug mein Name wie ein Messer",
            0.0,
            4.0,
            0.0,
            4.0,
            (
                {"word": "mein", "start": 1.5, "end": 2.0},
                {"word": "Name", "start": 2.0, "end": 2.5},
            ),
        )

        self.assertEqual("mein Name", result)

    def _write_scene_srt(self, directory: Path) -> Path:
        path = directory / "scenes.srt"
        path.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nScene 1\n",
            encoding="utf-8",
        )
        return path

    def test_local_artifact_store_reads_and_writes_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "scene.srt"
            store = JsonArtifactStore()

            store.write_text(path, "hello")

            self.assertEqual("hello", store.read_text(path))

    def test_stage1_relay_and_render_plan_write_json_through_artifact_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_srt = self._write_scene_srt(temp)
            timeline_path = temp / "timeline.json"
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            stage1_path = temp / "stage1.json"
            render_plan_path = temp / "render_plan.json"
            store = FakeArtifactStore()
            store.json_reads[str(timeline_path)] = [
                {"kind": "vocals", "start": 0.0, "end": 1.0, "lyrics": "line"},
            ]
            store.json_reads[str(scene_prompts_path)] = [
                {
                    "scene": 1,
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 2.0,
                    "duration": 2.0,
                    "type": "vocals",
                    "zimage_prompt": "image prompt",
                    "t2i_prompt": "image prompt",
                },
            ]

            build_stage1_segment_json(scene_srt, timeline_path, stage1_path, artifact_store=store)
            build_scene_prompt_relay(
                scene_srt,
                timeline_path,
                relay_path,
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=store,
            )
            store.json_reads[str(relay_path)] = store.json_writes[str(relay_path)]
            build_render_plan(
                scene_prompts_path,
                relay_path,
                render_plan_path,
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=store,
            )

            self.assertIn(str(stage1_path), store.json_writes)
            self.assertIn(str(relay_path), store.json_writes)
            self.assertIn(str(render_plan_path), store.json_writes)
            self.assertEqual(48, store.json_writes[str(render_plan_path)][0]["frame_count"])

    def test_splits_lyrics_at_scene_boundary_in_stage1_and_relay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_srt = temp / "scenes.srt"
            scene_srt.write_text(
                "1\n00:00:00,000 --> 00:00:08,000\nScene 1\n\n"
                "2\n00:00:08,000 --> 00:00:10,000\nScene 2\n",
                encoding="utf-8",
            )
            timeline_path = temp / "timeline.json"
            stage1_path = temp / "stage1.json"
            relay_path = temp / "relay.json"
            store = FakeArtifactStore()
            store.json_reads[str(timeline_path)] = [
                {
                    "kind": "vocals",
                    "start": 0.0,
                    "end": 10.0,
                    "lyrics": "one two three four five",
                    "word_timestamps": [
                        {"word": "one", "start": 0.0, "end": 0.5},
                        {"word": "two", "start": 0.5, "end": 1.0},
                        {"word": "three", "start": 1.0, "end": 1.5},
                        {"word": "four", "start": 7.5, "end": 9.5},
                        {"word": "five", "start": 9.5, "end": 10.0},
                    ],
                },
            ]

            build_stage1_segment_json(scene_srt, timeline_path, stage1_path, artifact_store=store)
            build_scene_prompt_relay(
                scene_srt,
                timeline_path,
                relay_path,
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=store,
            )

            stage1 = store.json_writes[str(stage1_path)]
            relay = store.json_writes[str(relay_path)]
            self.assertEqual("one two three", stage1[0]["lyrics"])
            self.assertEqual("four five", stage1[1]["lyrics"])
            self.assertIn("one two three", relay[0]["prompt_relay"][0]["prompt"])
            self.assertNotIn("four", relay[0]["prompt_relay"][0]["prompt"])
            self.assertIn("four five", relay[1]["prompt_relay"][0]["prompt"])
            self.assertNotIn("one two three", relay[1]["prompt_relay"][0]["prompt"])

    def test_measured_performance_metadata_reaches_relay_and_render_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            srt = self._write_scene_srt(temp)
            store = FakeArtifactStore()
            store.json_reads["timeline"] = [{"type": "vocals", "start": 0, "end": 2,
                "lyrics": "hold", "speaker_id": "voice1", "subject_id": "singer1", "offscreen": True,
                "word_timestamps": [{"word": "hold", "word_id": "w1", "source": "whisper",
                                     "start": 0.7, "end": 1.4}]}]
            build_stage1_segment_json(srt, "timeline", "stage1", artifact_store=store)
            self.assertEqual(0.7, store.json_writes["stage1"][0]["performance_intervals"][1]["start"])
            for fps in (24, 25, 30):
                settings = VideoSettings(fps=fps, width=1280, height=704)
                build_scene_prompt_relay(srt, "timeline", "relay", settings,
                                        artifact_store=store, min_segment_duration=1)
                relay = store.json_writes["relay"][0]["prompt_relay"]
                self.assertEqual(["instrumental", "singing", "instrumental"], [r["state"] for r in relay])
                self.assertLessEqual(abs(relay[1]["frame_start"] / fps - 0.7), 1 / fps)
                store.json_reads["relay"] = store.json_writes["relay"]
                store.json_reads["scenes"] = [{"scene": 1, "segment_id": "segment_001", "start": 0, "end": 2, "duration": 2, "type": "mixed", "zimage_prompt": "A singer waits", "vocal_performers": [{"subject_id": "singer2", "speaker_id": "voice2"}]}]
                build_render_plan("scenes", "relay", "plan", settings, artifact_store=store)
                rendered = store.json_writes["plan"][0]["ltx"]["prompt_relay"][1]
                self.assertEqual("w1", rendered["word_timestamps"][0]["word_id"])
                self.assertTrue(rendered["performance_phase"])
                self.assertEqual("voice1", rendered["speaker_id"])
                self.assertTrue(rendered["offscreen"])

    def test_projection_reports_initial_and_completed_progress_without_lyrics(self):
        from types import SimpleNamespace
        messages = []
        reporter = SimpleNamespace(message=messages.append)
        with tempfile.TemporaryDirectory() as directory:
            store = FakeArtifactStore()
            store.json_reads["timeline"] = []
            srt = self._write_scene_srt(Path(directory))
            build_stage1_segment_json(srt, "timeline", "stage1", artifact_store=store, reporter=reporter)
            build_scene_prompt_relay(srt, "timeline", "relay", VideoSettings(fps=24, width=1280, height=704),
                                     artifact_store=store, reporter=reporter)
        self.assertEqual(4, len(messages))
        self.assertIn("0/1", messages[0])
        self.assertIn("1/1", messages[-1])

    def test_scene_srt_writer_uses_artifact_store_for_text(self):
        store = FakeArtifactStore()

        write_scene_srt(
            "scenes.srt",
            [SrtScene(scene=1, start=0.0, end=1.0, text="Scene 1")],
            artifact_store=store,
        )

        self.assertIn("scenes.srt", store.text_writes)
        self.assertIn("00:00:00,000 --> 00:00:01,000", store.text_writes["scenes.srt"])


if __name__ == "__main__":
    unittest.main()
