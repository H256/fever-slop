import unittest
from pathlib import Path
from types import SimpleNamespace

from feverslop.application.h3_prompt_pipeline import _configured_audio_paths


class ConfiguredAudioPathTests(unittest.TestCase):
    def test_selects_configured_stems_in_configured_order(self):
        config = SimpleNamespace(
            minimax_h3_audio_refs=SimpleNamespace(stems=["vocals", "full_mix"]),
        )
        stem_files = {
            "drums": Path("drums.wav"),
            "full_mix": Path("full_mix.wav"),
            "vocals": Path("vocals.wav"),
            "bass": Path("bass.wav"),
        }

        selected = _configured_audio_paths(config, stem_files)

        self.assertEqual(
            {"vocals": Path("vocals.wav"), "full_mix": Path("full_mix.wav")},
            selected,
        )
        self.assertEqual(["vocals", "full_mix"], list(selected))

    def test_returns_none_when_no_configured_stem_exists(self):
        config = SimpleNamespace(
            minimax_h3_audio_refs=SimpleNamespace(stems=["vocals", "full_mix"]),
        )

        self.assertIsNone(_configured_audio_paths(config, {"drums": Path("drums.wav")}))


if __name__ == "__main__":
    unittest.main()