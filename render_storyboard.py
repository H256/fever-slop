# ruff: noqa: F401
"""Compatibility facade for the ``render_storyboard`` CLI; prefer
``feverslop.cli.render_storyboard``.

Re-exports the canonical entry points so the public ``render_storyboard.py``
interface keeps working. It no longer copies names back into the canonical
module: the two entry points are literally the same objects, so tests patch the
canonical module directly.
"""
from feverslop.cli.render_storyboard import (
    build_arg_parser,
    build_progress,
    build_render_storyboard_use_case,
    coerce_local_path,
    console,
    load_render_plan_subset,
    main,
    parse_scene_list,
)

__all__ = [
    "build_arg_parser",
    "build_progress",
    "build_render_storyboard_use_case",
    "coerce_local_path",
    "console",
    "load_render_plan_subset",
    "main",
    "parse_scene_list",
]

if __name__ == "__main__":
    main()
