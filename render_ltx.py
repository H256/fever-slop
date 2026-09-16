# ruff: noqa: F401
"""Compatibility facade for the ``render_ltx`` CLI; prefer
``feverslop.cli.render_ltx``.

This file re-exports the canonical entry points so the public ``render_ltx.py``
interface keeps working. It no longer copies names back into the canonical
module: the two entry points are literally the same objects, and nothing at
runtime (or in the test suite) needs the root module to shadow the package
module.
"""
from feverslop.cli.render_ltx import (
    ROLLING_FRAME_PROFILES,
    build_arg_parser,
    build_render_video_scenes_use_case,
    coerce_local_path,
    console,
    final_concat_paths,
    load_render_plan_subset,
    main,
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

__all__ = [
    "ROLLING_FRAME_PROFILES",
    "build_arg_parser",
    "build_render_video_scenes_use_case",
    "coerce_local_path",
    "console",
    "final_concat_paths",
    "load_render_plan_subset",
    "main",
    "namespace_to_options",
    "parse_scene_list",
    "resolve_composition_rolling_frames",
    "resolve_project_config_defaults",
    "resolve_rolling_frames",
    "rewrite_concat_list",
    "sanitize_file_stem",
    "safe_file_stem",
    "write_media_concat_list",
]

if __name__ == "__main__":
    main()
