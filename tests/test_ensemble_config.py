from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.project_config import ProjectConfig


def _write_config(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, handle)
    handle.close()
    return Path(handle.name)


def _base_config() -> dict:
    return {
        "project_name": "band",
        "input_audio": "song.wav",
        "subject_mode": "multi",
        "actors": [
            {"id": "singer_01", "name": "Singer"},
            {"id": "guitarist_01", "name": "Guitarist"},
        ],
    }


class EnsembleConfigLoadingTest(unittest.TestCase):
    def test_ensembles_loaded_from_config(self) -> None:
        payload = _base_config()
        payload["ensembles"] = [
            {
                "id": "main_band",
                "members": [
                    {"actor_id": "singer_01", "role": "singer"},
                    {"actor_id": "guitarist_01", "role": "guitarist"},
                ],
                "required_scene_types": ["performance"],
            }
        ]
        config = ProjectConfig.load(_write_config(payload))
        self.assertEqual(len(config.ensembles), 1)
        self.assertEqual(config.ensembles[0].id, "main_band")
        self.assertEqual(
            config.ensembles[0].member_actor_ids,
            ("singer_01", "guitarist_01"),
        )
        self.assertEqual(config.ensembles[0].required_scene_types, ("performance",))

    def test_no_ensembles_defaults_empty(self) -> None:
        config = ProjectConfig.load(_write_config(_base_config()))
        self.assertEqual(config.ensembles, ())

    def test_malformed_ensemble_raises(self) -> None:
        payload = _base_config()
        payload["ensembles"] = [{"members": [{"actor_id": "a"}]}]
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(payload))


if __name__ == "__main__":
    unittest.main()
