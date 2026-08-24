from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rich.console import Console

import main
from feverslop.application.canonical_plan_regeneration import (
    CanonicalPlanRegenerationService,
)
from feverslop.application.effective_render_plan import project_effective_scene
from feverslop.cli.canonical_plan_cli import run_canonical_plan_command
from feverslop.composition.arg_parser import build_arg_parser as build_pipeline_parser
from feverslop.domain.canonical_render_plan import build_canonical_scene


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "canonical_overrides"
    / "regression_corpus.json"
)


def _read_path(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        current = current[part]
    return current


class OverrideRegressionCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_semantic_overrides_survive_regeneration_and_reach_every_pipeline(self):
        self.assertTrue(FIXTURE_PATH.is_file(), "semantic regression corpus is missing")

        self.assertEqual(5, len(self.corpus["cases"]))
        self.assertEqual(
            {"classic_i2v", "ltx_msr", "ltx_ingredients", "minimax_h3"},
            {case["pipeline"] for case in self.corpus["cases"]},
        )

        service = CanonicalPlanRegenerationService()
        for number, case in enumerate(self.corpus["cases"], start=1):
            with self.subTest(case=case["id"]):
                segment_id = f"regression-{case['id']}"
                existing_canonical = build_canonical_scene(
                    segment_id=segment_id,
                    generated_roles={case["role"]: case["generated"]},
                )
                override = {
                    "value": case["override"],
                    "provenance": {
                        "source": "human",
                        "note": case["operator_intent"],
                    },
                }
                existing_canonical["roles"][case["role"]]["override"] = override
                existing = {
                    "scene": number,
                    "metadata": {"segment_id": segment_id},
                    "canonical": existing_canonical,
                }
                regenerated = {
                    "scene": number,
                    "metadata": {"segment_id": segment_id},
                    "canonical": build_canonical_scene(
                        segment_id=segment_id,
                        generated_roles={case["role"]: case["regenerated"]},
                    ),
                }

                merged = service.merge([existing], [regenerated]).scenes[0]
                projected = project_effective_scene(merged)
                role = merged["canonical"]["roles"][case["role"]]

                self.assertEqual(case["regenerated"], role["generated"]["value"])
                self.assertEqual(override, role["override"])
                self.assertEqual(case["override"], _read_path(projected, case["runtime_path"]))
                if case["pipeline"] == "classic_i2v":
                    self.assertEqual(
                        case["override"],
                        _read_path(projected, "ltx.original_style_i2v_prompt"),
                    )

    def test_diagnostics_separate_explicit_prompt_inspection_from_safe_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            plans = project / "output" / "render" / "plans"
            plans.mkdir(parents=True)
            scenes = []
            for number, case in enumerate(self.corpus["cases"], start=1):
                canonical = build_canonical_scene(
                    segment_id=f"regression-{case['id']}",
                    generated_roles={case["role"]: case["generated"]},
                )
                canonical["roles"][case["role"]]["override"] = {
                    "value": case["override"],
                    "provenance": {"source": "human", "note": case["operator_intent"]},
                }
                scenes.append({
                    "scene": number,
                    "canonical": canonical,
                    "request_payload": "SENSITIVE REQUEST PAYLOAD MUST NOT BE LOGGED",
                })
            (plans / "base.json").write_text(json.dumps(scenes), encoding="utf-8")

            show_output = io.StringIO()
            show_args = main.build_arg_parser().parse_args(
                ["plan", "show", str(project), "--scene", "1"],
            )
            self.assertEqual(
                0,
                run_canonical_plan_command(
                    show_args,
                    console=Console(file=show_output, force_terminal=False, width=500),
                ),
            )
            rendered_show = show_output.getvalue()
            first = self.corpus["cases"][0]
            self.assertIn("Generated", rendered_show)
            self.assertIn("Override", rendered_show)
            self.assertIn("Effective", rendered_show)
            self.assertIn(first["generated"], rendered_show)
            self.assertIn(first["override"], rendered_show)
            self.assertIn("human", rendered_show)
            self.assertNotIn("SENSITIVE REQUEST PAYLOAD", rendered_show)

            for argv in (
                ["status", str(project)],
                ["plan", "overrides", str(project)],
            ):
                with self.subTest(argv=argv):
                    safe_output = io.StringIO()
                    args = main.build_arg_parser().parse_args(argv)
                    run_canonical_plan_command(
                        args,
                        console=Console(file=safe_output, force_terminal=False, width=240),
                    )
                    rendered = safe_output.getvalue()
                    self.assertNotIn("SENSITIVE REQUEST PAYLOAD", rendered)
                    for case in self.corpus["cases"]:
                        self.assertNotIn(str(case["generated"]), rendered)
                        self.assertNotIn(str(case["override"]), rendered)

    def test_documented_operator_commands_match_current_cli_parsers(self):
        project = "projects/my-song"
        commands = (
            ["plan", "path", project],
            ["plan", "validate", project],
            ["plan", "show", project, "--scene", "3"],
            ["plan", "overrides", project],
            ["plan", "overrides", project, "--orphans"],
            ["status", project],
            ["plan-migrate", project],
            ["plan-migrate", project, "--apply"],
            ["run", project, "--dry-run"],
            ["run", project, "--resume"],
            ["run", project, "--dry-run", "--scenes", "3"],
            ["run", project, "--resume", "--scenes", "3"],
            ["run", project, "--resume", "--stage", "main_pipeline"],
            ["run", project, "--resume", "--stage", "h3_prompts"],
            ["run", project, "--resume", "--stage", "render_plan"],
            ["run", project, "--resume", "--stage", "msr_reference_sheets"],
            ["run", project, "--resume", "--stage", "ingredients_sheets"],
            ["run", project, "--resume", "--stage", "ltx_prepare_workflows"],
            ["run", project, "--resume", "--stage", "ltx_render_scenes"],
        )
        parser = main.build_arg_parser()
        for argv in commands:
            with self.subTest(argv=argv):
                self.assertIsNotNone(parser.parse_args(argv))

        full_regeneration = build_pipeline_parser().parse_args([
            project,
            "--stage",
            "main_pipeline",
            "--stage",
            "h3_prompts",
            "--stage",
            "render_plan",
            "--skip-tests",
        ])
        self.assertEqual(
            ["main_pipeline", "h3_prompts", "render_plan"],
            full_regeneration.stages,
        )

    def test_operator_documents_share_one_edit_target_and_all_six_journeys(self):
        root = Path(__file__).parent.parent
        documents = {
            path: (root / path).read_text(encoding="utf-8")
            for path in (
                "README.md",
                "documentation/render-plan-artifacts.md",
                "documentation/running.md",
                "documentation/project_workflow.md",
                "documentation/project-json-editing.md",
            )
        }
        for path, content in documents.items():
            with self.subTest(path=path):
                self.assertIn("base.json", content)

        running = documents["documentation/running.md"]
        for heading in (
            "### New project",
            "### Legacy project migration",
            "### Correct and rerender one scene",
            "### Interrupted H3 generation",
            "### Stale prepared workflow",
            "### Intentional full plan regeneration",
        ):
            self.assertIn(heading, running)

        combined = "\n".join(documents.values())
        for obsolete_guidance in (
            "`base.json` or `anchored.json`",
            "For a targeted V4 rerender, edit `ltx.prompt_relay` here",
            "Edit this file only when you are passing this exact file",
            "Use `output/render/plans/ingredients.json` for",
        ):
            self.assertNotIn(obsolete_guidance, combined)


if __name__ == "__main__":
    unittest.main()
