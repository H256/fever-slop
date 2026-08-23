import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            vocal_timeline_analyzer.VocalTimelineAnalyzer,
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


class DemucsSeparatorLazyModelTests(unittest.TestCase):
    def test_demucs_constructor_does_not_load_model_eagerly(self):
        from feverslop.adapters.audio import demucs_separator

        with patch.object(demucs_separator.pretrained, "get_model") as mock_get_model:
            separator = demucs_separator.DemucsSeparator(
                model_name="fake-model",
                device="cpu",
            )

        mock_get_model.assert_not_called()
        self.assertIsNone(separator.model)
        self.assertEqual(separator.model_name, "fake-model")

    def test_demucs_separate_loads_model_lazy_and_reuses_it(self):
        from feverslop.adapters.audio import demucs_separator

        fake_model = MagicMock()
        fake_model.samplerate = 44100
        fake_model.sources = ["vocals", "drums", "bass", "other"]
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            source = tmp_root / "song.wav"
            source.write_bytes(b"RIFF-fake")
            outdir = tmp_root / "stems"

            with (
                patch.object(
                    demucs_separator.pretrained,
                    "get_model",
                    return_value=fake_model,
                ) as mock_get_model,
                patch.object(
                    demucs_separator.torchaudio,
                    "load",
                    return_value=(torch.zeros(2, 16000), 44100),
                ),
                patch.object(
                    demucs_separator,
                    "apply_model",
                    return_value=torch.zeros(1, 4, 16000),
                ),
                patch.object(
                    demucs_separator.torchaudio,
                    "save",
                    side_effect=lambda path, *args, **kwargs: Path(
                        path,
                    ).write_bytes(b"fake-wav-bytes"),
                ),
            ):
                separator = demucs_separator.DemucsSeparator(
                    model_name="fake-model",
                    device="cpu",
                )
                first = separator.separate(source, outdir)
                second = separator.separate(source, outdir)

            mock_get_model.assert_called_once_with("fake-model")
            fake_model.to.assert_called_with("cpu")
            fake_model.eval.assert_called()
            self.assertEqual(first, second)
            for stem in fake_model.sources:
                expected = outdir / f"{stem}_song.wav"
                self.assertEqual(first[stem], expected)
                self.assertTrue(expected.exists())

    def test_demucs_close_without_load_is_noop(self):
        from feverslop.adapters.audio import demucs_separator

        with (
            patch.object(demucs_separator.pretrained, "get_model") as mock_get_model,
            patch.object(gc, "collect") as mock_gc,
        ):
            separator = demucs_separator.DemucsSeparator(
                model_name="fake-model",
                device="cpu",
            )
            separator.close()

        mock_get_model.assert_not_called()
        mock_gc.assert_not_called()
        self.assertIsNone(separator.model)


if __name__ == "__main__":
    unittest.main()
