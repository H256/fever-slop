# ruff: noqa: F401
"""Compatibility facade for the ``fix_ltx_prompt_anchors`` CLI; prefer
``feverslop.cli.fix_ltx_prompt_anchors``.

Re-exports the canonical entry points so the public ``fix_ltx_prompt_anchors.py``
interface keeps working. It no longer copies names back into the canonical
module: the two entry points are literally the same objects.
"""
from feverslop.cli.fix_ltx_prompt_anchors import (
    LTXPromptAnchorFixer,
    build_arg_parser,
    coerce_local_path,
    console,
    main,
    validate_anchor_file,
)

__all__ = [
    "LTXPromptAnchorFixer",
    "build_arg_parser",
    "coerce_local_path",
    "console",
    "main",
    "validate_anchor_file",
]

if __name__ == "__main__":
    main()
