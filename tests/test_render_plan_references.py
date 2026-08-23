import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.config.video_settings import VideoSettings
from feverslop.pipeline.render_plan_builder import build_render_plan


class RenderPlanReferencesTests(unittest.TestCase):
    def _base_files(self, temp: Path, scene: dict):
        scene_prompts = temp / "scene_prompts.json"
        relay = temp / "relay.json"
        scene_prompts.write_text(json.dumps([scene]), encoding="utf-8")
        relay.write_text(
            json.dumps([
                {
                    "scene": 1,
                    "prompt_relay": [
                        {"frame_start": 0, "frame_end": 24, "state": "singing", "prompt": "singing"},
                    ],
                },
            ]),
            encoding="utf-8",
        )
        return scene_prompts, relay

    def test_render_plan_carries_scene_reference_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts, relay = self._base_files(
                temp,
                {
                    "scene": 1,
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "type": "vocals",
                    "zimage_prompt": "image",
                    "references": {"actor_ids": ["singer"], "location_id": "stage"},
                },
            )

            output = build_render_plan(
                scene_prompts,
                relay,
                temp / "render_plan.json",
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
            )

            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual({"actor_ids": ["singer"], "location_id": "stage"}, plan[0]["references"])

    def test_render_plan_rejects_more_than_four_scene_actors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts, relay = self._base_files(
                temp,
                {
                    "scene": 1,
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "type": "vocals",
                    "zimage_prompt": "image",
                    "references": {"actor_ids": ["a", "b", "c", "d", "e"], "location_id": "stage"},
                },
            )

            with self.assertRaisesRegex(ValueError, "at most 4 actors"):
                build_render_plan(
                    scene_prompts,
                    relay,
                    temp / "render_plan.json",
                    VideoSettings(fps=24, width=1280, height=704),
                    artifact_store=JsonArtifactStore(),
                )

    def test_h3_render_plan_accepts_up_to_eight_scene_actors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_ids = [f"actor_{index}" for index in range(1, 7)]
            scene_prompts, relay = self._base_files(
                temp,
                {
                    "scene": 1,
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "type": "vocals",
                    "zimage_prompt": "image",
                    "references": {"actor_ids": actor_ids, "location_id": "stage"},
                },
            )

            output = build_render_plan(
                scene_prompts,
                relay,
                temp / "render_plan.json",
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                max_scene_actors=8,
            )

            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(actor_ids, plan[0]["references"]["actor_ids"])

    def test_audio_references_are_project_relative_without_generated_stems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            audio = project / "input" / "reference.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"audio")
            absolute = str(audio)
            scene_prompts, relay = self._base_files(project, {
                "scene": 1,
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "type": "vocals",
                "zimage_prompt": "image",
                "references": {
                    "reference_audio_paths": [absolute],
                    "_stem_audio_tags": {absolute: "audio_transfer"},
                },
            })

            output = build_render_plan(
                scene_prompts,
                relay,
                project / "render_plan.json",
                VideoSettings(fps=24, width=1280, height=704),
                artifact_store=JsonArtifactStore(),
                project_dir=project,
            )

            references = json.loads(output.read_text(encoding="utf-8"))[0]["references"]
            self.assertEqual(["input/reference.wav"], references["reference_audio_paths"])
            self.assertEqual({"input/reference.wav": "audio_transfer"}, references["_stem_audio_tags"])
