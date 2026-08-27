import tempfile
import unittest
from pathlib import Path

from feverslop.composition.stage_runners import _assemble_declared_cutless_groups
from feverslop.domain.continuation_segments import split_semantic_action
from feverslop.pipeline.continuation_render_plan import materialize_continuation_entries


class ContinuationEndToEndTests(unittest.TestCase):
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
