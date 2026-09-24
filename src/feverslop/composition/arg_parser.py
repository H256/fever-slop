from __future__ import annotations

import argparse

from feverslop.adapters.pipeline_runner_options import add_runner_options
from feverslop.domain.stages import PipelineStage


PUBLIC_PIPELINE_STAGES = tuple(
    stage for stage in PipelineStage
    if stage is not PipelineStage.SYNC_PROJECT_SETTINGS
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FeverSlop pipeline from Python.")
    parser.add_argument("project_root", nargs="?", default=None)
    parser.add_argument("--project-config", default=None)
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        choices=[stage.value for stage in PUBLIC_PIPELINE_STAGES],
        default=None,
        help="Run only a specific atomic pipeline stage. May be passed more than once.",
    )
    add_runner_options(parser)
    parser.add_argument(
        "--format",
        dest="timeline_format",
        choices=["mlt", "openshot", "both"],
        default="both",
        help="Timeline export format for export_timeline (default: both; MLT and OpenShot).",
    )
    return parser
