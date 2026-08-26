import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from feverslop.cli.app import build_arg_parser
from feverslop.cli.video_workflow_cli import run_profiles_command


class VideoWorkflowCliTests(unittest.TestCase):
    def _config(self):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "app_config.json"
        path.write_text(
            json.dumps(
                {
                    "video_workflow_profiles": [
                        {
                            "name": "ltx25-i2v-draft",
                            "pipeline": "ltx_i2v",
                            "workflow": "workflows/video/ltx_25/i2v/i2v_draft.json",
                            "purpose": "preview",
                            "stages": 2,
                            "output_scale": 1.0,
                            "supports_per_pass_loras": True,
                            "default": True,
                        },
                        {
                            "name": "ltx25-i2v-final",
                            "pipeline": "ltx_i2v",
                            "workflow": "workflows/video/ltx_25/i2v/i2v_final.json",
                            "purpose": "final",
                            "stages": 2,
                            "output_scale": 1.0,
                            "supports_per_pass_loras": True,
                            "default": True,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.addCleanup(temp_dir.cleanup)
        return path

    def _run(self, args):
        stream = io.StringIO()
        code = run_profiles_command(args, console=Console(file=stream, force_terminal=False, width=300))
        return code, stream.getvalue()

    def test_list_groups_profiles_and_marks_default(self):
        path = self._config()

        args = build_arg_parser().parse_args(["profiles", "list", "--app-config", str(path)])
        code, output = self._run(args)

        self.assertEqual(0, code)
        self.assertIn("ltx_i2v", output)
        self.assertIn("preview", output)
        self.assertIn("ltx25-i2v-draft", output)
        self.assertIn("DEFAULT", output)

    def test_preflight_reports_requested_and_default_resolved_profile(self):
        path = self._config()

        args = build_arg_parser().parse_args([
            "profiles", "preflight", "--app-config", str(path),
            "--pipeline", "ltx_i2v", "--purpose", "final",
        ])
        code, output = self._run(args)

        self.assertEqual(0, code)
        self.assertIn("Requested profile", output)
        self.assertIn("<default>", output)
        self.assertIn("Resolved profile", output)
        self.assertIn("ltx25-i2v-final", output)

    def test_preflight_reports_explicit_profile(self):
        path = self._config()

        args = build_arg_parser().parse_args([
            "profiles", "preflight", "--app-config", str(path),
            "--pipeline", "ltx_i2v", "--purpose", "preview",
            "--profile", "ltx25-i2v-final",
        ])
        code, output = self._run(args)

        self.assertEqual(1, code)
        self.assertIn("does not match pipeline/purpose", output.replace("\n", ""))
        self.assertNotIn("Resolved profile", output)

    def test_preflight_rejects_missing_default_before_any_backend(self):
        path = self._config()

        args = build_arg_parser().parse_args([
            "profiles", "preflight", "--app-config", str(path),
            "--pipeline", "ltx_msr", "--purpose", "final",
        ])
        code, output = self._run(args)

        self.assertEqual(1, code)
        self.assertIn("No video workflow profile is configured", output)


if __name__ == "__main__":
    unittest.main()
