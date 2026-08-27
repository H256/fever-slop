import tempfile
import unittest
import json
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import RenderVideoScenesRequest
from feverslop.application.render_video import RenderVideoScenesUseCase
from feverslop.composition.stage_runners import _assemble_declared_cutless_groups
from feverslop.domain.continuation_segments import split_semantic_action
from feverslop.pipeline.continuation_render_plan import materialize_continuation_entries


class ContinuationEndToEndTests(unittest.TestCase):
    def test_render_forced_resume_and_cutless_assembly_are_production_shaped(self):
        class FakePostprocessor:
            last_frame_index = 71

            @staticmethod
            def _frame_count(_clip):
                return 72

            @staticmethod
            def extract_last_frame(_clip, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"boundary")
                return path

            @staticmethod
            def extract_first_frame(_clip, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"boundary")
                return path

            @staticmethod
            def trim_clip(spec):
                return spec.output_file

            @staticmethod
            def write_concat_list(_clips, output_file):
                return output_file

            @staticmethod
            def concat_clips(_concat_list, output_file, **_kwargs):
                return output_file

        class FakeBackend:
            pipeline_name = "minimax-h3-r2v"

            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.project_dir = output_dir.parent
                self.postprocessor = FakePostprocessor()
                self.requests = []

            def render_video(self, request):
                self.requests.append(request)
                output = self.output_dir / f"scene_{request.scene_number:04}" / "final.mp4"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"video")
                return output

        segments = split_semantic_action(
            action_id="orbit",
            start_seconds=0.0,
            duration_seconds=6.0,
            max_duration_seconds=3.0,
            fps=24,
        )
        entries = materialize_continuation_entries(
            {
                "scene": 7,
                "segment_id": "semantic-007",
                "fps": 24,
                "metadata": {"segment_id": "semantic-007"},
            },
            group={
                "group_id": "semantic-007:orbit",
                "segments": [segment.__dict__ for segment in segments],
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "render_plan.json"
            plan_path.write_text(json.dumps(entries), encoding="utf-8")
            backend = FakeBackend(root / "render")
            request = RenderVideoScenesRequest(
                render_plan_path=plan_path,
                workflow_path=root / "workflow.json",
                audio_file=root / "song.mp3",
                storyboard_dir=root / "storyboard",
                output_dir=backend.output_dir,
                scene_numbers={7},
                skip_existing=False,
            )
            use_case = RenderVideoScenesUseCase(backend, JsonArtifactStore())
            first_outputs = use_case.execute(request)
            backend.requests.clear()
            resumed_outputs = use_case.execute(request)

            self.assertEqual(len(entries), len(first_outputs))
            self.assertEqual(len(entries), len(resumed_outputs))
            self.assertTrue(backend.requests[1].scene["keyframes"]["boundary_frame_manifest"])

            assembled = _assemble_declared_cutless_groups(
                entries,
                resumed_outputs,
                output_dir=root / "assembled",
                postprocessor=FakePostprocessor(),
            )

        self.assertEqual([root / "assembled" / "cutless_0001.mp4"], assembled)

    def test_materialized_chain_is_addressable_by_production_cutless_stage(self):
        class FakePostprocessor:
            @staticmethod
            def _frame_count(_clip):
                return 72

            @staticmethod
            def extract_first_frame(_clip, path):
                path.write_bytes(b"boundary")
                return path

            @staticmethod
            def extract_last_frame(_clip, path):
                path.write_bytes(b"boundary")
                return path

            @staticmethod
            def trim_clip(spec):
                return spec.output_file

            @staticmethod
            def write_concat_list(_clips, output_file):
                return output_file

            @staticmethod
            def concat_clips(_concat_list, output_file, **_kwargs):
                return output_file

        segments = split_semantic_action(
            action_id="orbit",
            start_seconds=4.0,
            duration_seconds=6.0,
            max_duration_seconds=3.0,
            fps=24,
        )
        entries = materialize_continuation_entries(
            {
                "scene": 7,
                "segment_id": "semantic-007",
                "fps": 24,
                "metadata": {"segment_id": "semantic-007"},
            },
            group={
                "group_id": "semantic-007:orbit",
                "segments": [segment.__dict__ for segment in segments],
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips = []
            for entry in entries:
                clip = root / f"{entry['technical_segment_id']}.mp4"
                clip.write_bytes(b"clip")
                clips.append(clip)
            assembled = _assemble_declared_cutless_groups(
                entries,
                clips,
                output_dir=root,
                postprocessor=FakePostprocessor(),
            )

            self.assertEqual([root / "cutless_0001.mp4"], assembled)
            self.assertTrue((root / "cutless_0001.json").is_file())


if __name__ == "__main__":
    unittest.main()
