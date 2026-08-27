import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.cutless_assembly import CutlessAssemblyService
from feverslop.composition.cutless_assembly import assemble_declared_cutless_group
from feverslop.composition.stage_runners import _assemble_declared_cutless_groups


class CutlessAssemblyPipelineTests(unittest.TestCase):
    def test_concat_stage_replaces_only_addressable_declared_groups(self):
        class FakePostprocessor:
            @staticmethod
            def _frame_count(_clip):
                return 48

            def extract_first_frame(self, _clip, path):
                path.write_bytes(b"same")
                return path

            def extract_last_frame(self, _clip, path):
                path.write_bytes(b"same")
                return path

            def trim_clip(self, spec):
                return spec.output_file

            def write_concat_list(self, _clips, output_file):
                return output_file

            def concat_clips(self, _concat_list, output_file, **_kwargs):
                return output_file

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips = [root / "a.mp4", root / "b.mp4", root / "independent.mp4"]
            for clip in clips:
                clip.write_bytes(b"clip")
            render_plan = [
                {"scene": 1, "metadata": {"segment_id": "a", "continuation_groups": [{
                    "group_id": "group", "segments": [
                        {"segment_id": "a", "duration_seconds": 2.0},
                        {"segment_id": "b", "duration_seconds": 2.0},
                    ],
                }]}},
                {"scene": 2, "metadata": {"segment_id": "b"}},
                {"scene": 3, "metadata": {"segment_id": "independent"}},
            ]

            output = _assemble_declared_cutless_groups(
                render_plan, clips, output_dir=root,
                postprocessor=FakePostprocessor(),
            )

            self.assertEqual([root / "cutless_0001.mp4", clips[2]], output)
            self.assertTrue((root / "cutless_0001.json").is_file())

    def test_declared_group_derives_diagnostics_and_assembles_without_crossfade(self):
        class FakePostprocessor:
            def __init__(self):
                self.concat_kwargs = None

            def trim_clip(self, spec):
                return spec.output_file

            def write_concat_list(self, clips, output_file):
                return output_file

            def concat_clips(self, _concat_list, output_file, **kwargs):
                self.concat_kwargs = kwargs
                return output_file

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips = {segment_id: root / f"{segment_id}.mp4" for segment_id in ("a", "b")}
            for clip in clips.values():
                clip.write_bytes(b"clip")
            frames = {"a.first.png": b"first-a", "a.last.png": b"same", "b.first.png": b"same", "b.last.png": b"last-b"}
            fake = FakePostprocessor()
            output = assemble_declared_cutless_group(
                group={"group_id": "g", "segments": [
                    {"segment_id": "a", "duration_seconds": 2.0},
                    {"segment_id": "b", "duration_seconds": 2.0},
                ]},
                clips_by_segment=clips,
                frame_count=lambda _clip: 48,
                extract_first_frame=lambda _clip, path: (path.write_bytes(frames[path.name]), path)[1],
                extract_last_frame=lambda _clip, path: (path.write_bytes(frames[path.name]), path)[1],
                assembly_service=CutlessAssemblyService(fake),
                output_file=root / "assembled.mp4",
                diagnostics_file=root / "diagnostics.json",
                fps=24,
            )

            self.assertEqual(root / "assembled.mp4", output)
            self.assertTrue(fake.concat_kwargs["reencode"])
            payload = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual("accept", payload["assembly"]["outcome"])
            self.assertEqual(["b"], payload["assembly"]["trim_first_frame_segments"])
            self.assertEqual(0, payload["assembly"]["diagnostics"][0]["timing_delta_frames"])


if __name__ == "__main__":
    unittest.main()
