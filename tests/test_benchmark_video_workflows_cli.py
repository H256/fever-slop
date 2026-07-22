from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from feverslop.domain.prepared_workflow import SCHEMA, SceneWorkflowManifest
from feverslop.tools import benchmark_video_workflows as cli


class BenchmarkVideoWorkflowsCliTests(unittest.TestCase):
    def test_parser_requires_cases_output_and_comfyui_url(self):
        parser = cli.build_arg_parser()

        for arguments in ([], ["--case", "one=workflow.json"], ["--output", "report.json"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_case_parser_rejects_malformed_blank_and_non_workflow_paths(self):
        for value in ("candidate", "=workflow.json", "   =workflow.json", "candidate=scene.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cli.parse_case(value)

        case = cli.parse_case("candidate=prepared/workflow.json")

        self.assertEqual("candidate", case.name)
        self.assertEqual(Path("prepared/workflow.json"), case.prepared_workflow)

    def test_preflight_rejects_duplicate_names_and_normalized_workflow_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_prepared_workflow(root, "one", scene=1)
            second = _write_prepared_workflow(root, "two", scene=2)

            with self.assertRaisesRegex(ValueError, "duplicate benchmark case name"):
                cli.preflight_cases(
                    (cli.parse_case(f"Candidate={first}"), cli.parse_case(f"candidate={second}"))
                )
            with self.assertRaisesRegex(ValueError, "duplicate prepared workflow path"):
                cli.preflight_cases(
                    (cli.parse_case(f"one={first}"), cli.parse_case(f"two={first.parent / '.' / 'workflow.json'}"))
                )

    def test_preflight_rejects_missing_files_manifests_and_mixed_projects_or_pipelines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing" / "workflow.json"
            with self.assertRaisesRegex(FileNotFoundError, "prepared workflow"):
                cli.preflight_cases((cli.parse_case(f"missing={missing}"),))

            orphan = root / "orphan" / "workflow.json"
            orphan.parent.mkdir()
            orphan.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "manifest"):
                cli.preflight_cases((cli.parse_case(f"orphan={orphan}"),))

            first = _write_prepared_workflow(root / "project-a", "one", scene=1)
            other_project = _write_prepared_workflow(root / "project-b", "two", scene=2)
            with self.assertRaisesRegex(ValueError, "same project root"):
                cli.preflight_cases(
                    (cli.parse_case(f"one={first}"), cli.parse_case(f"two={other_project}"))
                )

            other_pipeline = _write_prepared_workflow(
                root / "project-a", "three", scene=3, pipeline="another"
            )
            with self.assertRaisesRegex(ValueError, "same pipeline"):
                cli.preflight_cases(
                    (cli.parse_case(f"one={first}"), cli.parse_case(f"three={other_pipeline}"))
                )

    def test_run_uses_injected_composition_and_prints_report_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = _write_prepared_workflow(root, "one", scene=1)
            report = root / "benchmarks" / "run.json"
            arguments = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(report),
                    "--comfyui-url", "http://comfy.test:8188",
                ]
            )
            captured: dict[str, object] = {}

            class UseCase:
                def execute(self, cases):
                    captured["executed_cases"] = cases
                    return report

            def compose(**kwargs):
                captured.update(kwargs)
                return UseCase()

            output = io.StringIO()
            written = cli.run(arguments, compose=compose, output=output)

            self.assertEqual(report, written)
            self.assertEqual(report, captured["report_path"])
            self.assertEqual(root.resolve(), captured["project_dir"])
            self.assertEqual("test-pipeline", captured["expected_pipeline"])
            self.assertEqual("http://comfy.test:8188", captured["comfyui_url"])
            self.assertEqual(report.resolve(), Path(output.getvalue().strip()))

    def test_run_rejects_invalid_manifest_budget_before_composition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = _write_prepared_workflow(
                root,
                "over-budget",
                scene=1,
                render_frame_count=25,
                max_render_frames=17,
            )
            args = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(root / "run.json"),
                    "--comfyui-url", "http://localhost:8188",
                ]
            )

            with self.assertRaisesRegex(ValueError, "limited to 17 frames"):
                cli.run(args, compose=lambda **_kwargs: self.fail("must not compose"))

    def test_main_returns_nonzero_for_structurally_malformed_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            malformed_payloads = ([], {"schema": SCHEMA, "assets": []})

            for index, payload in enumerate(malformed_payloads):
                with self.subTest(payload=payload):
                    workflow = root / str(index) / "workflow.json"
                    workflow.parent.mkdir()
                    workflow.write_text("{}", encoding="utf-8")
                    workflow.with_name("manifest.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

                    exit_code = cli.main(
                        [
                            "--case", f"candidate={workflow}",
                            "--output", str(root / f"run-{index}.json"),
                            "--comfyui-url", "http://localhost:8188",
                        ],
                        compose=lambda **_kwargs: self.fail("must not compose"),
                    )

                    self.assertEqual(2, exit_code)

    def test_main_returns_nonzero_for_overflowing_manifest_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = root / "prepared" / "workflow.json"
            workflow.parent.mkdir()
            workflow.write_text("{}", encoding="utf-8")
            workflow.with_name("manifest.json").write_text(
                f'{{"schema": "{SCHEMA}", "assets": [], "scene": 1e999}}',
                encoding="utf-8",
            )

            exit_code = cli.main(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(root / "run.json"),
                    "--comfyui-url", "http://localhost:8188",
                ],
                compose=lambda **_kwargs: self.fail("must not compose"),
            )

            self.assertEqual(2, exit_code)

    def test_evidence_directory_uses_the_full_report_filename(self):
        json_evidence = cli.evidence_directory(Path("run.json"))
        text_evidence = cli.evidence_directory(Path("run.txt"))

        self.assertEqual(Path("run.json.evidence"), json_evidence)
        self.assertEqual(Path("run.txt.evidence"), text_evidence)
        self.assertNotEqual(json_evidence, text_evidence)

    def test_run_rejects_existing_report_or_evidence_before_composition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = _write_prepared_workflow(root, "one", scene=1)
            report = root / "run.json"
            args = cli.build_arg_parser().parse_args(
                ["--case", f"candidate={workflow}", "--output", str(report), "--comfyui-url", "http://localhost:8188"]
            )

            for collision in (report, root / "run.json.evidence"):
                with self.subTest(collision=collision):
                    if collision.suffix:
                        collision.parent.mkdir(parents=True, exist_ok=True)
                        collision.write_text("existing", encoding="utf-8")
                    else:
                        collision.mkdir(parents=True, exist_ok=True)
                    with self.assertRaises(FileExistsError):
                        cli.run(args, compose=lambda **_kwargs: self.fail("must not compose"))
                    if collision.is_dir():
                        collision.rmdir()
                    else:
                        collision.unlink()


def _write_prepared_workflow(
    project: Path,
    name: str,
    *,
    scene: int,
    pipeline: str = "test-pipeline",
    render_frame_count: int | None = None,
    max_render_frames: int | None = None,
) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    template = project / "templates" / f"{name}.json"
    template.parent.mkdir(exist_ok=True)
    template.write_text("{}", encoding="utf-8")
    render_plan = project / "render_plan.json"
    render_plan.write_text("[]", encoding="utf-8")
    workflow = project / "prepared" / name / "workflow.json"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("{}", encoding="utf-8")
    manifest = SceneWorkflowManifest.create(
        project_dir=project,
        scene=scene,
        pipeline=pipeline,
        workflow_path=workflow,
        template_path=template,
        render_plan_path=render_plan,
        assets=[],
        seed=1,
        fps=24,
        frame_count=25,
        render_frame_count=render_frame_count,
        width=1280,
        height=704,
        max_render_frames=max_render_frames,
    )
    manifest.write(workflow.with_name("manifest.json"))
    return workflow


if __name__ == "__main__":
    unittest.main()
