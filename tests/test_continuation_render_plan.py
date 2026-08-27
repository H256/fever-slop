import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from feverslop.application.render_video import _attach_r2v_continuation_anchor
from feverslop.application.render_video import RenderVideoScenesRequest, RenderVideoScenesUseCase
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.domain.continuation_segments import split_semantic_action
from feverslop.pipeline.continuation_render_plan import materialize_continuation_entries


class ContinuationRenderPlanTests(unittest.TestCase):
    def test_production_r2v_use_case_renders_and_anchors_complete_technical_chain(self):
        class FakePostprocessor:
            last_frame_index = 3

            @staticmethod
            def extract_last_frame(_source, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"boundary")
                return destination

        class FakeBackend:
            pipeline_name = "minimax-h3-r2v"

            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.postprocessor = FakePostprocessor()
                self.project_dir = output_dir.parent
                self.requests = []

            def render_video(self, request):
                self.requests.append(request)
                output = self.output_dir / f"scene_{request.scene_number:04}" / "final.mp4"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"video")
                return output

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "render_plan.json"
            plan_path.write_text(
                __import__("json").dumps([
                    {
                        "scene": 7_001_001,
                        "semantic_scene": 7,
                        "technical_segment_id": "orbit-0001",
                        "segment_id": "orbit-0001",
                        "duration_seconds": 3.0,
                        "frame_count": 72,
                    },
                    {
                        "scene": 7_001_002,
                        "semantic_scene": 7,
                        "technical_segment_id": "orbit-0002",
                        "segment_id": "orbit-0002",
                        "continuation_predecessor_id": "orbit-0001",
                        "duration_seconds": 3.0,
                        "frame_count": 72,
                    },
                ]),
                encoding="utf-8",
            )
            backend = FakeBackend(root / "render")
            RenderVideoScenesUseCase(backend, JsonArtifactStore()).execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_path,
                    workflow_path=root / "workflow.json",
                    audio_file=root / "song.mp3",
                    storyboard_dir=root / "storyboard",
                    output_dir=backend.output_dir,
                    scene_numbers={7},
                    skip_existing=False,
                ),
            )

        self.assertEqual(2, len(backend.requests))
        self.assertEqual(
            "last_frame_from_previous",
            backend.requests[1].scene["keyframes"]["startframe_mode"],
        )
        self.assertEqual(
            "render/scene_7001001/final.mp4",
            backend.requests[1].scene["keyframes"]["boundary_frame_manifest"]["source_clip_path"],
        )

    def test_r2v_anchor_materialization_persists_verified_boundary(self):
        class FakePostprocessor:
            last_frame_index = 47

            @staticmethod
            def extract_last_frame(_source, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"last-frame")
                return destination

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predecessor = root / "scene_7001001" / "final.mp4"
            predecessor.parent.mkdir()
            predecessor.write_bytes(b"video")
            result = _attach_r2v_continuation_anchor(
                {
                    "scene": 7_001_002,
                    "continuation_predecessor_id": "orbit-0001",
                },
                predecessor_id="orbit-0001",
                predecessor_clip=predecessor,
                output_dir=root / "render",
                backend=SimpleNamespace(
                    postprocessor=FakePostprocessor(),
                    project_dir=root,
                ),
            )

            manifest = result["keyframes"]["boundary_frame_manifest"]
            self.assertEqual("scene_7001001/final.mp4", manifest["source_clip_path"])
            self.assertEqual("render/keyframes/scene_7001001_to_7001002_start.png", manifest["frame_path"])
            self.assertEqual(47, manifest["frame_index"])
            self.assertEqual("last_frame_from_previous", result["keyframes"]["startframe_mode"])

    def test_materializes_each_segment_as_an_addressable_render_entry(self):
        segments = split_semantic_action(
            action_id="orbit",
            start_seconds=4.0,
            duration_seconds=8.0,
            max_duration_seconds=3.0,
            fps=24,
        )
        entries = materialize_continuation_entries(
            {
                "scene": 7,
                "segment_id": "semantic-007",
                "abs_start_seconds": 4.0,
                "abs_end_seconds": 12.0,
                "duration_seconds": 8.0,
                "frame_count": 192,
                "fps": 24,
                "metadata": {"segment_id": "semantic-007", "type": "instrumental"},
            },
            group={
                "group_id": "semantic-007:orbit",
                "semantic_action": "orbit",
                "semantic_start_seconds": 4.0,
                "semantic_end_seconds": 12.0,
                "segments": [segment.__dict__ for segment in segments],
            },
        )

        self.assertEqual(["orbit-0001", "orbit-0002", "orbit-0003"], [entry["technical_segment_id"] for entry in entries])
        self.assertEqual([7_001_001, 7_001_002, 7_001_003], [entry["scene"] for entry in entries])
        self.assertEqual([7, 7, 7], [entry["semantic_scene"] for entry in entries])
        self.assertEqual([None, "orbit-0001", "orbit-0002"], [entry["continuation_predecessor_id"] for entry in entries])
        self.assertEqual(4.0, entries[0]["abs_start_seconds"])
        self.assertEqual(12.0, entries[-1]["abs_end_seconds"])
        self.assertEqual(192, sum(entry["frame_count"] for entry in entries))


if __name__ == "__main__":
    unittest.main()
