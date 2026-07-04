import unittest
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from tools.repair_scene_srt import main as repair_scene_srt_main
import movie_pipeline
import run_pipeline


class RunnerScriptTests(unittest.TestCase):
    def test_os_specific_runner_scripts_are_removed(self):
        self.assertFalse(Path("test.ps1").exists())
        self.assertFalse(Path("test.bat").exists())

    def test_run_pipeline_parser_defaults_match_python_runner_contract(self):
        args = run_pipeline.build_arg_parser().parse_args([])

        self.assertIsNone(args.project_root)
        self.assertIsNone(args.project_config)
        self.assertEqual("app_config.json", args.app_config)
        self.assertEqual(10, args.concept_batch_size)
        self.assertEqual(os.fspath(Path("workflows") / "image_t2i_startframe_v1.json"), args.storyboard_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "image_t2i_startframe_krea_v1.json"), args.reference_hero_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "image_edit_flux2_klein_1ref_v1.json"), args.reference_edit_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "video_ltxv_msr_1actor_1background_v1.json"), args.msr_workflow)
        self.assertEqual("", args.relay_workflow)
        self.assertEqual(os.fspath(Path("workflows") / "video_ltxv_i2v_v1.json"), args.single_prompt_workflow)
        self.assertEqual("ltx_i2v", args.video_pipeline)
        self.assertEqual("single_prompt", args.render_mode)
        self.assertEqual("#PROMPT", args.single_prompt_title)
        self.assertEqual("text", args.single_prompt_input)
        self.assertEqual("original", args.rolling_frame_profile)
        self.assertFalse(args.randomize_seed)
        self.assertEqual(16, args.smoke_scene)
        self.assertFalse(args.skip_main_pipeline)

    def test_run_pipeline_parser_accepts_powershell_parity_flags(self):
        args = run_pipeline.build_arg_parser().parse_args(
            [
                "projects/song",
                "--render-mode",
                "auto",
                "--relay-workflow",
                "relay.json",
                "--single-prompt-workflow",
                "single.json",
                "--video-pipeline",
                "ltx_msr",
                "--reference-hero-workflow",
                "hero.json",
                "--reference-edit-workflow",
                "edit.json",
                "--msr-workflow",
                "msr.json",
                "--skip-msr-reference-render",
                "--storyboard-lora-strength",
                "0.4",
                "--video-character-lora-strength",
                "0.8",
                "--video-lora-1-strength-model",
                "0.7",
                "--video-lora-1-strength-clip",
                "0.6",
                "--lora-split-enabled",
                "--randomize-seed",
                "--smoke-only",
                "--no-skip-existing",
                "--skip-tests",
                "--skip-main-pipeline",
                "--skip-relay-compact",
                "--skip-anchor-fix",
                "--skip-storyboard",
                "--skip-storyboard-page",
                "--skip-ltx",
                "--skip-final-concat",
                "--diagnostic-original-audio-mux",
                "--no-original-audio-mux",
            ]
        )

        self.assertEqual("projects/song", args.project_root)
        self.assertEqual("auto", args.render_mode)
        self.assertEqual("ltx_msr", args.video_pipeline)
        self.assertEqual("hero.json", args.reference_hero_workflow)
        self.assertEqual("edit.json", args.reference_edit_workflow)
        self.assertEqual("msr.json", args.msr_workflow)
        self.assertTrue(args.skip_msr_reference_render)
        self.assertEqual(0.4, args.storyboard_lora_strength)
        self.assertEqual(0.8, args.video_character_lora_strength)
        self.assertEqual(0.7, args.video_lora_1_strength_model)
        self.assertEqual(0.6, args.video_lora_1_strength_clip)
        self.assertTrue(args.lora_split_enabled)
        self.assertTrue(args.randomize_seed)
        self.assertTrue(args.smoke_only)
        self.assertTrue(args.no_skip_existing)
        self.assertTrue(args.skip_tests)
        self.assertTrue(args.skip_main_pipeline)
        self.assertTrue(args.skip_relay_compact)
        self.assertTrue(args.skip_anchor_fix)
        self.assertTrue(args.skip_storyboard)
        self.assertTrue(args.skip_storyboard_page)
        self.assertTrue(args.skip_ltx)
        self.assertTrue(args.skip_final_concat)
        self.assertTrue(args.diagnostic_original_audio_mux)
        self.assertTrue(args.no_original_audio_mux)

    def test_repair_scene_srt_cli_writes_repaired_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_srt = temp / "input.srt"
            output_srt = temp / "output.srt"
            input_srt.write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nScene 1\n\n"
                "2\n00:00:00,500 --> 00:00:02,000\nScene 2\n",
                encoding="utf-8",
            )

            argv = [
                "repair_scene_srt.py",
                "--input-srt",
                str(input_srt),
                "--output-srt",
                str(output_srt),
                "--min-duration",
                "1.0",
                "--max-duration",
                "3.0",
            ]
            with patch.object(sys, "argv", argv):
                repair_scene_srt_main()

            self.assertTrue(output_srt.exists())
            self.assertIn("00:00:00,000 --> 00:00:02,000", output_srt.read_text(encoding="utf-8"))

    def test_movie_pipeline_parser_accepts_skip_stage_flags(self):
        args = movie_pipeline.build_arg_parser().parse_args(
            [
                "projects/tm3",
                "--reference-backend",
                "local",
                "--render-backend",
                "local",
                "--skip-movie-bible",
                "--skip-movie-plan",
                "--skip-movie-references",
                "--skip-movie-msr-enrich",
                "--skip-movie-render",
                "--force-movie-references",
            ]
        )

        self.assertEqual("projects/tm3", args.project_dir)
        self.assertEqual("local", args.reference_backend)
        self.assertEqual("local", args.render_backend)
        self.assertTrue(args.skip_movie_bible)
        self.assertTrue(args.skip_movie_plan)
        self.assertTrue(args.skip_movie_references)
        self.assertTrue(args.skip_movie_msr_enrich)
        self.assertTrue(args.skip_movie_render)
        self.assertTrue(args.force_movie_references)

    def test_movie_pipeline_cli_can_run_references_only_with_local_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir))

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-render",
                    ]
                )
            )

            manifest = json.loads((project / "movie" / "references" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(project.resolve(), result.project_dir)
            self.assertEqual(project / "movie" / "references" / "manifest.json", result.reference_manifest_path)
            self.assertIsNone(result.final_video_path)
            self.assertEqual("local", manifest["generator_backend"])
            self.assertEqual("movie/references/actors/mara/msr_sheet.png", manifest["actors"][0]["msr_sheet_path"])
            self.assertTrue((project / "movie" / "render_plan_msr.json").exists())

    def test_movie_pipeline_cli_can_skip_references_and_render_with_local_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = _write_movie_project(Path(temp_dir), ready=True)

            result = movie_pipeline.run(
                movie_pipeline.build_arg_parser().parse_args(
                    [
                        str(project),
                        "--reference-backend",
                        "local",
                        "--render-backend",
                        "local",
                        "--skip-movie-references",
                    ]
                )
            )

            self.assertEqual(project / "output" / "movie" / "test-movie.mp4", result.final_video_path)
            self.assertTrue(result.final_video_path.exists())


def _write_movie_project(root: Path, *, ready: bool = False) -> Path:
    project = root / "test-movie"
    references = project / "movie" / "references"
    references.mkdir(parents=True)
    (project / "movie" / "render_plan.json").write_text(
        json.dumps(
            {
                "title": "Test Movie",
                "resolution": {"width": 1280, "height": 704},
                "shots": [
                    {
                        "shot_id": "shot_0001",
                        "description": "Mara enters the archive",
                        "duration_seconds": 1,
                        "reference_ids": {"actors": ["mara"], "location": "archive"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "actors": [
            {
                "id": "mara",
                "name": "Mara",
                "visual_description": "gothic archivist",
                "image_prompt": "Full-body cinematic character reference sheet for Mara. gothic archivist. Four vertical panels in one image.",
                "prompt": "Full-body cinematic character reference sheet for Mara. gothic archivist. Four vertical panels in one image.",
                "msr_sheet_path": "movie/references/actors/mara/msr_sheet.png" if ready else "",
            }
        ],
        "locations": [
            {
                "id": "archive",
                "name": "Archive",
                "prompt": "Archive",
                "msr_sheet_path": "movie/references/locations/archive/views/hero.png" if ready else "",
            }
        ],
        "generator_backend": "local" if ready else "",
    }
    (references / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project


if __name__ == "__main__":
    unittest.main()
