# ruff: noqa: F401
import argparse

from feverslop.cli import render_storyboard as _cli
from feverslop.cli.render_storyboard import (
    AppConfig,
    Console,
    Panel,
    RenderStoryboardRequest,
    WorkflowAnchorConfig,
    build_arg_parser,
    build_progress,
    build_render_storyboard_use_case,
    coerce_local_path,
    console,
    load_render_plan_subset,
    main as _package_main,
    parse_scene_list,
)


def main():
    for name in (
        "AppConfig", "WorkflowAnchorConfig", "RenderStoryboardRequest", "build_progress",
        "build_render_storyboard_use_case", "coerce_local_path", "console",
        "load_render_plan_subset", "parse_scene_list", "Panel", "Console",
    ):
        setattr(_cli, name, globals()[name])
    return _package_main()


if __name__ == "__main__":
    main()
