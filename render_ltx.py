# ruff: noqa: F401
import argparse

from feverslop.cli import render_ltx as _cli
from feverslop.cli.render_ltx import (
    AppConfig,
    BarColumn,
    Console,
    Panel,
    Path,
    Progress,
    ProjectConfig,
    RenderVideoScenesRequest,
    ResolvedLoraConfig,
    ROLLING_FRAME_PROFILES,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    WorkflowAnchorConfig,
    build_arg_parser,
    build_render_video_scenes_use_case,
    coerce_local_path,
    console,
    final_concat_paths,
    load_render_plan_subset,
    main as _package_main,
    namespace_to_options,
    parse_scene_list,
    resolve_composition_rolling_frames,
    resolve_project_config_defaults,
    resolve_rolling_frames,
    rewrite_concat_list,
    sanitize_file_stem,
    safe_file_stem,
    write_media_concat_list,
)


def main():
    for name in (
        "AppConfig", "ProjectConfig", "ResolvedLoraConfig", "RenderVideoScenesRequest",
        "ROLLING_FRAME_PROFILES", "WorkflowAnchorConfig", "build_render_video_scenes_use_case",
        "coerce_local_path", "console", "namespace_to_options", "resolve_composition_rolling_frames",
        "load_render_plan_subset", "parse_scene_list", "safe_file_stem", "write_media_concat_list",
        "Panel", "Progress", "BarColumn", "TaskProgressColumn", "TextColumn",
        "TimeElapsedColumn", "TimeRemainingColumn",
    ):
        setattr(_cli, name, globals()[name])
    return _package_main()


if __name__ == "__main__":
    main()
