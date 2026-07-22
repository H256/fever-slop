from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Protocol, TextIO

from feverslop.adapters.benchmark_artifacts import LocalBenchmarkArtifactStore
from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.json_benchmark_store import JsonBenchmarkResultStore
from feverslop.adapters.prepared_workflow import PreparedWorkflowRenderer
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.application.benchmark_video_workflows import BenchmarkVideoWorkflowsUseCase
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.domain.scene_duration_limits import validate_render_frame_budget
from feverslop.domain.workflow_benchmark import WorkflowBenchmarkCase
from feverslop.errors import FeverSlopValidationError


_SAFE_CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class _BenchmarkUseCase(Protocol):
    def execute(self, cases: tuple[WorkflowBenchmarkCase, ...]) -> Path: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark already prepared FeverSlop video workflows.",
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case,
        metavar="NAME=PATH",
        help="Named prepared workflow.json; repeat for each benchmark case.",
    )
    parser.add_argument("--output", required=True, help="New JSON benchmark report path.")
    parser.add_argument("--comfyui-url", required=True, help="ComfyUI HTTP base URL.")
    return parser


def parse_case(value: str) -> WorkflowBenchmarkCase:
    if not isinstance(value, str) or "=" not in value:
        raise ValueError("benchmark case must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name:
        raise ValueError("benchmark case name cannot be blank")
    if not _SAFE_CASE_NAME.fullmatch(name):
        raise ValueError(
            "benchmark case name must contain only letters, numbers, dot, underscore, or hyphen"
        )
    if not raw_path:
        raise ValueError("benchmark workflow path cannot be blank")
    workflow_path = Path(raw_path)
    if workflow_path.name.casefold() != "workflow.json":
        raise ValueError("benchmark case path must name workflow.json")
    return WorkflowBenchmarkCase(name, workflow_path)


def preflight_cases(
    cases: tuple[WorkflowBenchmarkCase, ...],
) -> tuple[tuple[WorkflowBenchmarkCase, ...], Path, str]:
    if not cases:
        raise ValueError("at least one benchmark case is required")

    normalized_cases: list[WorkflowBenchmarkCase] = []
    names: set[str] = set()
    workflow_paths: set[str] = set()
    project_dir: Path | None = None
    expected_pipeline: str | None = None

    for case in cases:
        normalized_name = case.name.casefold()
        if normalized_name in names:
            raise ValueError(f"duplicate benchmark case name: {case.name}")
        names.add(normalized_name)

        workflow_path = case.prepared_workflow.expanduser().resolve()
        normalized_path = os.path.normcase(os.fspath(workflow_path))
        if normalized_path in workflow_paths:
            raise ValueError(f"duplicate prepared workflow path: {workflow_path}")
        workflow_paths.add(normalized_path)
        if workflow_path.name.casefold() != "workflow.json":
            raise ValueError(f"prepared workflow path must name workflow.json: {workflow_path}")
        if not workflow_path.is_file():
            raise FileNotFoundError(f"prepared workflow does not exist: {workflow_path}")

        manifest_path = workflow_path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared workflow manifest does not exist: {manifest_path}")
        manifest = _read_manifest(manifest_path, case.name)
        case_project_dir = _derive_project_dir(workflow_path, manifest)

        if project_dir is None:
            project_dir = case_project_dir
        elif os.path.normcase(os.fspath(project_dir)) != os.path.normcase(
            os.fspath(case_project_dir)
        ):
            raise ValueError("all prepared workflows must use the same project root")

        if expected_pipeline is None:
            expected_pipeline = manifest.pipeline
        elif expected_pipeline != manifest.pipeline:
            raise ValueError("all prepared workflows must use the same pipeline")

        mismatches = manifest.verify(case_project_dir)
        if mismatches:
            raise ValueError(
                f"prepared workflow verification failed for {case.name}: "
                + "; ".join(mismatches)
            )
        try:
            validate_render_frame_budget(
                scene_number=manifest.scene,
                render_frame_count=manifest.render_frame_count,
                fps=manifest.fps,
                workflow_path=(
                    manifest.render_budget_workflow_path or manifest.template.path
                ),
                max_render_frames=manifest.max_render_frames,
                max_render_duration_seconds=manifest.max_render_duration_seconds,
                round_render_frames_to_8n1=manifest.round_render_frames_to_8n1,
            )
        except FeverSlopValidationError as exc:
            raise ValueError(
                f"invalid prepared workflow render budget for {case.name}: {exc}"
            ) from None
        normalized_cases.append(WorkflowBenchmarkCase(case.name, workflow_path))

    assert project_dir is not None
    assert expected_pipeline is not None
    return tuple(normalized_cases), project_dir, expected_pipeline


def _read_manifest(path: Path, case_name: str) -> SceneWorkflowManifest:
    try:
        return SceneWorkflowManifest.read(path)
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"prepared workflow manifest is malformed for {case_name}: {exc}"
        ) from None


def _derive_project_dir(workflow_path: Path, manifest: SceneWorkflowManifest) -> Path:
    if manifest.workflow.external:
        raise ValueError("prepared workflow manifest cannot use an external workflow path")
    relative = PurePosixPath(manifest.workflow.path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("prepared workflow manifest contains an invalid workflow path")

    project_dir = workflow_path
    for _part in relative.parts:
        project_dir = project_dir.parent
    project_dir = project_dir.resolve()
    if (project_dir / Path(*relative.parts)).resolve() != workflow_path.resolve():
        raise ValueError(
            f"prepared path {workflow_path} does not match manifest workflow {manifest.workflow.path}"
        )
    return project_dir


def evidence_directory(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.name}.evidence")


def compose_benchmark(
    *,
    project_dir: Path,
    expected_pipeline: str,
    report_path: Path,
    comfyui_url: str,
) -> BenchmarkVideoWorkflowsUseCase:
    client = ComfyUIClient(base_url=comfyui_url)
    renderer = PreparedWorkflowRenderer(
        project_dir=project_dir,
        render_queue=ComfyUIRenderQueue(client),
        postprocessor=VideoPostProcessor(),
        expected_pipeline=expected_pipeline,
        asset_uploader=ComfyUIVideoAssetUploader(client),
        model_resolver=ComfyUIModelResolver(client),
    )
    return BenchmarkVideoWorkflowsUseCase(
        renderer=renderer,
        clock=MonotonicClock(),
        artifact_store=LocalBenchmarkArtifactStore(evidence_directory(report_path)),
        result_store=JsonBenchmarkResultStore(report_path, base_path=report_path.parent),
    )


def run(
    args: argparse.Namespace,
    *,
    compose: Callable[..., _BenchmarkUseCase] = compose_benchmark,
    output: TextIO | None = None,
) -> Path:
    cases, project_dir, expected_pipeline = preflight_cases(tuple(args.case))
    report_path = Path(args.output).expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"benchmark report already exists: {report_path}")
    evidence_path = evidence_directory(report_path)
    if evidence_path.exists():
        raise FileExistsError(f"benchmark evidence already exists: {evidence_path}")
    comfyui_url = str(args.comfyui_url).strip()
    if not comfyui_url:
        raise ValueError("ComfyUI URL cannot be blank")

    use_case = compose(
        project_dir=project_dir,
        expected_pipeline=expected_pipeline,
        report_path=report_path,
        comfyui_url=comfyui_url,
    )
    written = use_case.execute(cases)
    print(written.resolve(), file=output)
    return written


def main(
    argv: list[str] | None = None,
    *,
    compose: Callable[..., _BenchmarkUseCase] = compose_benchmark,
) -> int:
    try:
        run(build_arg_parser().parse_args(argv), compose=compose)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
