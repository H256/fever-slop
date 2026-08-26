from __future__ import annotations

import unittest
from unittest.mock import patch

import full_auto
import movie_pipeline
from feverslop.cli.movie_cli import config_from_args
from feverslop.composition.movie_pipeline_jobs import movie_runtime_config


class LegacyWorkflowOverrideTests(unittest.TestCase):
    def test_explicit_legacy_paths_survive_all_supported_movie_profile_defaults(self):
        args = movie_pipeline.build_arg_parser().parse_args([
            "projects/legacy",
            "--hero-workflow", r"legacy\image\hero.json",
            "--msr-workflow", r"legacy\ltx\msr.json",
            "--r2v-workflow", r"legacy\h3\r2v.json",
            "--sequence-to-sheet-workflow", r"legacy\sequence\sheet.json",
        ])

        config = config_from_args(args)

        self.assertEqual("legacy/image/hero.json", config["hero_workflow"])
        self.assertEqual("legacy/ltx/msr.json", config["msr_workflow"])
        self.assertEqual("legacy/h3/r2v.json", config["r2v_workflow"])
        self.assertEqual("legacy/sequence/sheet.json", config["sequence_to_sheet_workflow"])

    def test_explicit_audio_workflow_survives_full_auto_defaults(self):
        args = full_auto.build_arg_parser().parse_args([
            "--idea", "legacy song",
            "--style", "minimal",
            "--workflow", r"legacy\audio\song.json",
        ])

        with patch("feverslop.cli.full_auto.build_full_auto_use_case") as build:
            full_auto.run_full_auto_command(args)

        self.assertEqual(
            "legacy/audio/song.json",
            build.call_args.kwargs["workflow_path"].as_posix(),
        )

    def test_resolving_legacy_paths_does_not_mutate_the_input_mapping(self):
        legacy = {
            "hero_workflow": r"legacy\image\hero.json",
            "msr_workflow": r"legacy\ltx\msr.json",
            "r2v_workflow": r"legacy\h3\r2v.json",
            "sequence_to_sheet_workflow": r"legacy\sequence\sheet.json",
        }
        before = dict(legacy)

        movie_runtime_config(legacy)

        self.assertEqual(before, legacy)


if __name__ == "__main__":
    unittest.main()
