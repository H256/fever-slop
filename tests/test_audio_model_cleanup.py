import gc
import unittest
from unittest.mock import patch

import torch


class AudioModelCleanupTests(unittest.TestCase):
    def test_demucs_close_releases_model_before_collecting_cuda_cache(self):
        from feverslop.adapters.audio import demucs_separator

        separator = demucs_separator.DemucsSeparator.__new__(demucs_separator.DemucsSeparator)
        separator.model = object()
        events = []

        self.assertTrue(hasattr(separator, "close"), "DemucsSeparator must expose close()")

        with (
            patch.object(gc, "collect", side_effect=lambda: events.append("gc")),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache", side_effect=lambda: events.append("empty_cache")),
        ):
            separator.close()

        self.assertIsNone(separator.model)
        self.assertEqual(["gc", "empty_cache"], events)

    def test_whisper_close_releases_model_before_collecting_cuda_cache(self):
        from feverslop.adapters.audio import vocal_timeline_analyzer

        analyzer = vocal_timeline_analyzer.VocalTimelineAnalyzer.__new__(
            vocal_timeline_analyzer.VocalTimelineAnalyzer
        )
        analyzer.model = object()
        events = []

        self.assertTrue(hasattr(analyzer, "close"), "VocalTimelineAnalyzer must expose close()")

        with (
            patch.object(gc, "collect", side_effect=lambda: events.append("gc")),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.cuda,
                "empty_cache",
                side_effect=lambda: events.append("empty_cache"),
            ),
        ):
            analyzer.close()

        self.assertIsNone(analyzer.model)
        self.assertEqual(["gc", "empty_cache"], events)


if __name__ == "__main__":
    unittest.main()
