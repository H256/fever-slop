from __future__ import annotations

from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts
from feverslop.composition.arg_parser import PipelineStage, build_arg_parser
from feverslop.composition.config_loader import (
    PipelineRunContext,
    PipelineRunResult,
    build_run_context,
    collect_render_plan_scene_clips,
    convert_to_safe_file_stem,
    count_render_plan_items,
    resolve_runner_path,
    rewrite_concat_list,
    runner_root,
)
from feverslop.composition.pipeline_runner import main, run
from feverslop.composition.stage_runners import (
    RenderProgressReporter,
    run_unittest_suite,
    write_step,
)

__all__ = [
    "PipelineRunContext",
    "PipelineRunResult",
    "PipelineStage",
    "RenderProgressReporter",
    "build_arg_parser",
    "build_run_context",
    "collect_render_plan_scene_clips",
    "convert_to_safe_file_stem",
    "count_render_plan_items",
    "enrich_render_plan_with_msr_prompts",
    "main",
    "resolve_runner_path",
    "rewrite_concat_list",
    "run",
    "run_unittest_suite",
    "runner_root",
    "write_step",
]


if __name__ == "__main__":
    main()
