# ruff: noqa: F401
"""Compatibility facade for the ``compact_relay_prompts`` CLI; prefer
``feverslop.cli.compact_relay_prompts``.

Re-exports the canonical entry points so the public ``compact_relay_prompts.py``
interface keeps working. It no longer copies names back into the canonical
module: the two entry points are literally the same objects.
"""
from feverslop.cli.compact_relay_prompts import (
    build_arg_parser,
    coerce_local_path,
    console,
    main,
)

__all__ = [
    "build_arg_parser",
    "coerce_local_path",
    "console",
    "main",
]

if __name__ == "__main__":
    main()
