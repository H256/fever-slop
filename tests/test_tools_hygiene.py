import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import r2v_prompt_check, repair_scene_srt


class ToolsHygieneTests(unittest.TestCase):
    def test_r2v_config_reads_utf8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "app_config.json"
            config.write_text(json.dumps({"llm": {"api_key": "ä-secret"}}), encoding="utf-8")

            self.assertEqual("ä-secret", r2v_prompt_check.load_api_key(config))

    def test_r2v_expected_fields_reject_empty_values(self):
        missing = r2v_prompt_check.missing_expected_fields(
            {"subject_definitions": "", "summary": "ok"},
            ["subject_definitions", "summary"],
        )

        self.assertEqual(["subject_definitions"], missing)

    def test_repair_scene_srt_rejects_non_writable_output_before_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output.srt"
            with patch.object(repair_scene_srt.os, "access", return_value=False):
                with self.assertRaisesRegex(PermissionError, "not writable"):
                    repair_scene_srt.ensure_output_writable(output)


if __name__ == "__main__":
    unittest.main()
