import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig


class GlobalAssetConfigTests(unittest.TestCase):
    def test_app_config_has_overridable_library_root(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "app.json"
            path.write_text(json.dumps({"global_library_path": "library"}), encoding="utf-8")
            config = AppConfig.load(path)
            self.assertEqual((path.parent / "library").resolve(), config.global_library_path)

    def test_project_config_reads_global_asset_declarations_without_breaking_legacy_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "song.wav").write_bytes(b"audio")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.wav",
                "actors": [{"id": "local", "name": "Local"}],
                "global_cast": [{"asset_id": "ava", "look_id": "default", "role": "lead"}],
                "global_locations": [{"asset_id": "nightclub", "look_id": "default"}],
                "global_styles": [{"asset_id": "neon-noir"}],
                "global_props": [{"asset_id": "guitar", "look_id": "default"}],
            }), encoding="utf-8")
            config = ProjectConfig.load(config_path)
            self.assertEqual("local", config.actors[0].id)
            self.assertEqual("ava", config.global_cast[0].asset_id)
            self.assertEqual("guitar", config.global_props[0].asset_id)


if __name__ == "__main__":
    unittest.main()
