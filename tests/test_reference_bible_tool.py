import json
import tempfile
import unittest
from pathlib import Path

from feverslop.tools.reference_bible import build_arg_parser, load_reference_subjects


class ReferenceBibleToolTests(unittest.TestCase):
    def test_parser_accepts_project_and_workflow_paths(self):
        args = build_arg_parser().parse_args(
            [
                "--project-config",
                "projects/demo/config.json",
                "--hero-workflow",
                "workflows/image_t2i_startframe_v1.json",
                "--edit-workflow",
                "workflows/image_edit_flux2_klein_1ref_v1.json",
            ]
        )

        self.assertEqual("projects/demo/config.json", args.project_config)
        self.assertEqual("workflows/image_edit_flux2_klein_1ref_v1.json", args.edit_workflow)

    def test_load_reference_subjects_falls_back_to_legacy_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": "song.mp3", "subject": "a singer in a red coat"}),
                encoding="utf-8",
            )

            subjects, locations = load_reference_subjects(config_path)

            self.assertEqual("subject", subjects[0].id)
            self.assertEqual("a singer in a red coat", subjects[0].image_prompt)
            self.assertEqual([], locations)
