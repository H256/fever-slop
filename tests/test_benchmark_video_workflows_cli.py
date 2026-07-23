from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from feverslop.domain import prepared_workflow as prepared_workflow_domain
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

    def test_preflight_rejects_cross_platform_unsafe_manifest_workflow_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            for index, stored_path in enumerate(
                ("C:/project/prepared/workflow.json", "..\\outside\\workflow.json")
            ):
                with self.subTest(stored_path=stored_path):
                    workflow = _write_prepared_workflow(root, f"unsafe-{index}", scene=index + 1)
                    manifest_path = workflow.with_name("manifest.json")
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload["workflow"]["path"] = stored_path
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "invalid workflow path"):
                        cli.preflight_cases((cli.parse_case(f"candidate={workflow}"),))

    def test_preflight_rejects_non_external_render_plan_escape_before_outside_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            workflow = _write_prepared_workflow(project, "candidate", scene=1)
            outside = root / "outside" / "plan.json"
            outside.parent.mkdir()
            outside.write_text("[]", encoding="utf-8")
            manifest_path = workflow.with_name("manifest.json")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["render_plan"] = {
                "path": "../outside/plan.json",
                "sha256": prepared_workflow_domain.sha256_file(outside),
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            original_sha256 = prepared_workflow_domain.sha256_file

            def guarded_sha256(path):
                if Path(path).resolve() == outside.resolve():
                    self.fail("preflight must not read outside render plan")
                return original_sha256(path)

            args = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(project / "run.json"),
                    "--comfyui-url", "http://localhost:8188",
                ]
            )
            with (
                patch(
                    "feverslop.domain.prepared_workflow.sha256_file",
                    side_effect=guarded_sha256,
                ),
                self.assertRaisesRegex(ValueError, "invalid render_plan path"),
            ):
                cli.run(args, compose=lambda **_kwargs: self.fail("must not compose"))

    def test_preflight_rejects_non_external_asset_escape_before_outside_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            workflow = _write_prepared_workflow(project, "candidate", scene=1)
            outside = root / "outside" / "asset.png"
            outside.parent.mkdir()
            outside.write_bytes(b"image")
            manifest_path = workflow.with_name("manifest.json")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["assets"] = [
                {
                    "role": "reference",
                    "path": "..\\outside\\asset.png",
                    "sha256": prepared_workflow_domain.sha256_file(outside),
                    "comfyui_name": "asset.png",
                }
            ]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            original_sha256 = prepared_workflow_domain.sha256_file

            def guarded_sha256(path):
                if Path(path).resolve() == outside.resolve():
                    self.fail("preflight must not read outside asset")
                return original_sha256(path)

            args = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(project / "run.json"),
                    "--comfyui-url", "http://localhost:8188",
                ]
            )
            with (
                patch(
                    "feverslop.domain.prepared_workflow.sha256_file",
                    side_effect=guarded_sha256,
                ),
                self.assertRaisesRegex(ValueError, r"invalid asset\[reference\] path"),
            ):
                cli.run(args, compose=lambda **_kwargs: self.fail("must not compose"))

    def test_preflight_preserves_explicitly_external_template_semantics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            workflow = _write_prepared_workflow(project, "candidate", scene=1)
            external_template = root / "shared" / "template.json"
            external_template.parent.mkdir()
            external_template.write_text("{}", encoding="utf-8")
            manifest_path = workflow.with_name("manifest.json")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["template"] = {
                "path": str(external_template.resolve()),
                "sha256": prepared_workflow_domain.sha256_file(external_template),
                "external": True,
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            cases, project_dir, pipeline = cli.preflight_cases(
                (cli.parse_case(f"candidate={workflow}"),)
            )

            self.assertEqual((workflow.resolve(),), tuple(case.prepared_workflow for case in cases))
            self.assertEqual(project.resolve(), project_dir)
            self.assertEqual("test-pipeline", pipeline)

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

    def test_composition_serializes_real_report_paths_relative_to_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            workflow = _write_prepared_workflow(project, "candidate", scene=1)
            rendered = project / "output" / "render" / "scene.mp4"
            rendered.parent.mkdir(parents=True)
            rendered.write_bytes(b"video")
            report = project / "output" / "benchmarks" / "run.json"

            class Renderer:
                def render(self, _workflow_path):
                    return rendered

            class Clock:
                def __init__(self):
                    self.values = iter((1.0, 2.0))

                def now(self):
                    return next(self.values)

            with (
                patch(
                    "feverslop.tools.benchmark_video_workflows.PreparedWorkflowRenderer",
                    return_value=Renderer(),
                ),
                patch(
                    "feverslop.tools.benchmark_video_workflows.MonotonicClock",
                    return_value=Clock(),
                ),
            ):
                use_case = cli.compose_benchmark(
                    project_dir=project,
                    expected_pipeline="test-pipeline",
                    report_path=report,
                    comfyui_url="http://localhost:8188",
                )
                use_case.execute((cli.parse_case(f"candidate={workflow}"),))

            result = json.loads(report.read_text(encoding="utf-8"))["results"][0]
            self.assertEqual("prepared/candidate/workflow.json", result["prepared_workflow"])
            self.assertEqual(
                "output/benchmarks/run.json.evidence/candidate.mp4",
                result["output_path"],
            )

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

    def test_main_writes_failure_diagnostics_only_to_stderr(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = cli.main(
                [
                    "--case", "candidate=missing/workflow.json",
                    "--output", "run.json",
                    "--comfyui-url", "http://localhost:8188",
                ],
                compose=lambda **_kwargs: self.fail("must not compose"),
            )

        self.assertEqual(2, exit_code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("prepared workflow does not exist", stderr.getvalue())

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

    def test_run_reservation_blocks_concurrent_run_and_cleans_up_on_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = _write_prepared_workflow(root, "one", scene=1)
            report = root / "run.json"
            args = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(report),
                    "--comfyui-url", "http://localhost:8188",
                ]
            )
            test_case = self

            class UseCase:
                def execute(self, _cases):
                    test_case.assertTrue(cli.reservation_path(report).exists())
                    with test_case.assertRaises(FileExistsError):
                        cli.run(
                            args,
                            compose=lambda **_kwargs: test_case.fail("must not compose"),
                        )
                    report.write_text("{}", encoding="utf-8")
                    return report

            cli.run(args, compose=lambda **_kwargs: UseCase(), output=io.StringIO())

            self.assertFalse(cli.reservation_path(report).exists())

    def test_run_failure_cleans_reservation_and_partial_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = _write_prepared_workflow(root, "one", scene=1)
            report = root / "run.json"
            evidence = cli.evidence_directory(report)
            args = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(report),
                    "--comfyui-url", "http://localhost:8188",
                ]
            )

            class UseCase:
                def execute(self, _cases):
                    evidence.mkdir()
                    (evidence / "candidate.mp4").write_bytes(b"partial")
                    raise RuntimeError("render interrupted")

            with self.assertRaisesRegex(RuntimeError, "render interrupted"):
                cli.run(args, compose=lambda **_kwargs: UseCase())

            self.assertFalse(cli.reservation_path(report).exists())
            self.assertFalse(evidence.exists())

    def test_run_interrupt_after_first_capture_cleans_partial_run_and_reraises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = _write_prepared_workflow(root, "one", scene=1)
            report = root / "run.json"
            evidence = cli.evidence_directory(report)
            args = cli.build_arg_parser().parse_args(
                [
                    "--case", f"candidate={workflow}",
                    "--output", str(report),
                    "--comfyui-url", "http://localhost:8188",
                ]
            )

            class UseCase:
                def execute(self, _cases):
                    evidence.mkdir()
                    (evidence / "candidate.mp4").write_bytes(b"captured")
                    raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                cli.run(args, compose=lambda **_kwargs: UseCase())

            self.assertFalse(report.exists())
            self.assertFalse(evidence.exists())
            self.assertFalse(cli.reservation_path(report).exists())


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
