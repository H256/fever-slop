import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console
from rich.progress import SpinnerColumn

from feverslop.tools.reference_bible import (
    build_arg_parser,
    load_reference_subjects,
    resolve_view_names,
    run,
)


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
            ],
        )

        self.assertEqual("projects/demo/config.json", args.project_config)
        self.assertEqual("workflows/image_edit_flux2_klein_1ref_v1.json", args.edit_workflow)
        self.assertEqual("msr", args.view_set)
        self.assertEqual("image_views", args.reference_generation)

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
            ],
        )

        self.assertEqual("full", args.view_set)

    def test_sequence_mode_constructs_backend_and_uses_one_work_unit_per_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_audio": "song.mp3",
                        "actors": [{"id": "singer", "name": "Singer", "image_prompt": "singer"}],
                        "locations": [{"id": "stage", "name": "Stage", "image_prompt": "stage"}],
                    },
                ),
                encoding="utf-8",
            )
            sequence_workflow = temp / "sequence.json"
            args = build_arg_parser().parse_args(
                [
                    "--project-config",
                    str(config_path),
                    "--app-config",
                    "app_config.json",
                    "--hero-workflow",
                    "hero.json",
                    "--edit-workflow",
                    "edit.json",
                    "--view-set",
                    "full",
                    "--reference-generation",
                    "sequence_sheet",
                    "--sequence-workflow",
                    str(sequence_workflow),
                ],
            )
            self.assertEqual("sequence_sheet", args.reference_generation)
            self.assertEqual(str(sequence_workflow), args.sequence_workflow)

            sequence_backend = Mock()
            sequence_backend_factory = Mock(return_value=sequence_backend)
            planner_factory = Mock()
            fake_generator = Mock()
            fake_generator.generate_subject_bible.return_value = temp / "actor.json"
            fake_generator.generate_location_bible.return_value = temp / "location.json"
            generator_factory = Mock(return_value=fake_generator)
            generator_factory.view_names = ("front", "right", "rear", "left", "wide", "close")
            record_console = Console(file=io.StringIO(), record=True, force_terminal=False)

            with patch("feverslop.tools.reference_bible.AppConfig.load") as app_config, \
                    patch("feverslop.tools.reference_bible.ComfyUIClient"), \
                    patch("feverslop.tools.reference_bible.ComfyUIModelResolver"), \
                    patch("feverslop.tools.reference_bible.ComfyUIImageBackend"), \
                    patch("feverslop.tools.reference_bible.ComfyUISequenceToSheetBackend", sequence_backend_factory), \
                    patch("feverslop.tools.reference_bible.ReferenceSheetPlanner", planner_factory), \
                    patch("feverslop.tools.reference_bible.LocalOpenAIClient"), \
                    patch("feverslop.tools.reference_bible.ReferenceBibleGenerator", generator_factory), \
                    patch("feverslop.tools.reference_bible.console", record_console):
                app_config.return_value.comfyui.base_url = "http://localhost:8188"
                app_config.return_value.comfyui.prompt_timeout_seconds = 1
                app_config.return_value.comfyui.model_overrides = []

                manifests = run(args)

            printed = record_console.export_text()
            self.assertEqual([temp / "actor.json", temp / "location.json"], manifests)
            sequence_backend_factory.assert_called_once()
            self.assertEqual(str(sequence_workflow), sequence_backend_factory.call_args.kwargs["workflow_path"])
            self.assertIs(sequence_backend, generator_factory.call_args.kwargs["sequence_backend"])
            self.assertIs(planner_factory.return_value, generator_factory.call_args.kwargs["sequence_planner"])
            self.assertIn("Actor views: 6", printed)
            self.assertIn("Location views: 6", printed)
            self.assertIn("Total renders: 2", printed)

    def test_msr_view_set_uses_actor_sheet_views_and_single_location_background(self):
        actor_views, location_views = resolve_view_names("msr")

        self.assertEqual(("msr_sheet",), actor_views)
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
                    },
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
                ],
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
                ],
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
            self.assertIn("Actor views: 1", printed)
            self.assertIn("Total renders: 1", printed)
            self.assertEqual(("msr_sheet",), generator_factory.call_args.kwargs["actor_view_names"])
            self.assertEqual(("hero",), generator_factory.call_args.kwargs["location_view_names"])

    def test_run_progress_avoids_unicode_spinner_column(self):
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
                ],
            )
            fake_generator = Mock()
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

            column_types = [type(column) for column in run._last_progress_columns]
            self.assertNotIn(SpinnerColumn, column_types)
