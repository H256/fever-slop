"""Compatibility facade for the root CLI; prefer ``feverslop.cli.app``.

This file used to be a forked copy of the installed CLI that drifted over
time (it lost the ``full-auto`` and ``profiles`` subcommands and the
``resolve_workflow_reference`` step in the render path, so ``python main.py``
and the installed ``feverslop`` command exposed different feature sets).

It is now a thin re-export of the canonical ``feverslop.cli.app`` module so
both entry points are literally the same code and cannot drift.
"""

from __future__ import annotations

from feverslop.cli.app import (
    _run_render,
    build_arg_parser,
    console,
    main,
)

__all__ = [
    "main",
    "build_arg_parser",
    "console",
    "_run_render",
]

if __name__ == "__main__":
    main()
