from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.adapters.pipeline_runner_options import build_runner_argv


class RunPipelineAdapter:
    def __init__(self, *, run_pipeline, build_arg_parser):
        self._run_pipeline = run_pipeline
        self._build_arg_parser = build_arg_parser

    def run(self, *, project_config_path: Path, options: dict[str, Any]) -> Path | None:
        argv = build_runner_argv(project_config_path, options)
        result = self._run_pipeline(self._build_arg_parser().parse_args(argv))
        return result.final_video_path
