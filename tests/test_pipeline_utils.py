import json
import tempfile
import unittest
from pathlib import Path

from feverslop.domain.timeline import TimelineSegment
from feverslop.pipeline.utils import save_timeline_json


class PipelineUtilsTests(unittest.TestCase):
    def test_saves_raw_whisper_sidecar_next_to_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            timeline_path = Path(temp) / "timeline_song.json"
            raw = [{"start": 1.0, "end": 2.0, "text": "hello", "words": []}]

            save_timeline_json(
                [TimelineSegment(start=1.0, end=2.0, kind="vocals", text="hello")],
                timeline_path,
                whisper_raw=raw,
            )

            sidecar = timeline_path.with_name("timeline_song_whisper_raw.json")
            self.assertEqual(raw, json.loads(sidecar.read_text(encoding="utf-8")))
