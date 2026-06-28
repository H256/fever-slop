import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console

from feverslop.tools.reference_bible import build_arg_parser, load_reference_subjects, resolve_view_names
from feverslop.tools.reference_bible import run


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
        self.assertEqual("msr", args.view_set)

    def test_parser_accepts_full_bible_view_set(self):
        args = build_arg_parser().parse_args(
            [
                "--project-config",
                "projects/demo/config.json",
                "--hero-workflow",
                "workflows/image_t2i_startframe_v1.json",
                "--edit-workflow",
                "workflows/image_edit_flux2_klein_1ref_v1.json",
                "--view-set",
                "full",
            ]
        )

        self.assertEqual("full", args.view_set)

    def test_msr_view_set_uses_actor_sheet_views_and_single_location_background(self):
        actor_views, location_views = resolve_view_names("msr")

        self.assertEqual(("hero_closeup", "front", "left", "back"), actor_views)
        self.assertEqual(("hero",), location_views)

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

    def test_load_reference_subjects_reads_resolved_context_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": "input/song.mp3", "subject": ""}),
                encoding="utf-8",
            )
            prompts_dir = temp / "output" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "resolved_context_song.json").write_text(
                json.dumps(
                    {
                        "actors": [{"id": "mara", "name": "Mara", "image_prompt": "portrait"}],
                        "structured_locations": [{"id": "stage", "name": "Stage", "image_prompt": "stage"}],
                    }
                ),
                encoding="utf-8",
            )

            subjects, locations = load_reference_subjects(config_path)

            self.assertEqual("mara", subjects[0].id)
            self.assertEqual("stage", locations[0].id)

    def test_run_fails_loudly_when_no_reference_subjects_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": "input/song.mp3", "subject": "", "locations": []}),
                encoding="utf-8",
            )
            args = build_arg_parser().parse_args(
                [
                    "--project-config",
                    str(config_path),
                    "--app-config",
                    "app_config.json",
                    "--hero-workflow",
                    "workflows/image_t2i_startframe_v1.json",
                    "--edit-workflow",
                    "workflows/image_edit_flux2_klein_1ref_v1.json",
                ]
            )

            with self.assertRaisesRegex(ValueError, "No reference actors or locations found"):
                run(args)

    def test_run_prints_reference_summary_before_rendering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": "song.mp3", "subject": "a singer"}),
                encoding="utf-8",
            )
            args = build_arg_parser().parse_args(
                [
                    "--project-config",
                    str(config_path),
                    "--app-config",
                    "app_config.json",
                    "--hero-workflow",
                    "workflows/image_t2i_startframe_v1.json",
                    "--edit-workflow",
                    "workflows/image_edit_flux2_klein_1ref_v1.json",
                ]
            )
            fake_generator = Mock()
            fake_generator.view_names = ("hero", "front")
            fake_generator.generate_subject_bible.return_value = temp / "manifest.json"
            generator_factory = Mock(return_value=fake_generator)
            record_console = Console(file=io.StringIO(), record=True, force_terminal=False)

            with patch("feverslop.tools.reference_bible.AppConfig.load") as app_config, \
                    patch("feverslop.tools.reference_bible.ComfyUIClient"), \
                    patch("feverslop.tools.reference_bible.ComfyUIModelResolver"), \
                    patch("feverslop.tools.reference_bible.ComfyUIImageBackend"), \
                    patch("feverslop.tools.reference_bible.ReferenceBibleGenerator", generator_factory), \
                    patch("feverslop.tools.reference_bible.console", record_console):
                app_config.return_value.comfyui.base_url = "http://localhost:8188"
                app_config.return_value.comfyui.prompt_timeout_seconds = 1
                app_config.return_value.comfyui.model_overrides = []

                run(args)

            printed = record_console.export_text()
            self.assertIn("Reference Bible render plan", printed)
            self.assertIn("Actors: 1", printed)
            self.assertIn("Total renders: 4", printed)
            self.assertEqual(("hero_closeup", "front", "left", "back"), generator_factory.call_args.kwargs["actor_view_names"])
            self.assertEqual(("hero",), generator_factory.call_args.kwargs["location_view_names"])
