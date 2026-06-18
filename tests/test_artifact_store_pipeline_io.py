import json
import tempfile
import unittest
from pathlib import Path

from autoprompter.adapters.local_artifacts import JsonArtifactStore
from autoprompter.pipeline.prompt_relay_builder import build_scene_prompt_relay
from autoprompter.pipeline.render_plan_builder import build_render_plan
from autoprompter.pipeline.scene_duration_enforcer import SrtScene, write_scene_srt
from autoprompter.pipeline.stage1_segment_builder import build_stage1_segment_json
from autoprompter.config.video_settings import VideoSettings


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
                {"kind": "vocals", "start": 0.0, "end": 1.0, "lyrics": "line"}
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
                }
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
