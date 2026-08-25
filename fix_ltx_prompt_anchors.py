# ruff: noqa: F401
import argparse

from feverslop.cli import fix_ltx_prompt_anchors as _cli
from feverslop.cli.fix_ltx_prompt_anchors import (
    Console,
    Panel,
    LTXPromptAnchorFixer,
    validate_anchor_file,
    build_arg_parser,
    coerce_local_path,
    console,
    main as _package_main,
)


def main():
    for name in "LTXPromptAnchorFixer", "validate_anchor_file", "coerce_local_path", "console", "Panel", "Console":
        setattr(_cli, name, globals()[name])
    return _package_main()


if __name__ == "__main__":
    main()
