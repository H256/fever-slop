# ruff: noqa: F401
import argparse

from feverslop.cli import compact_relay_prompts as _cli
from feverslop.cli.compact_relay_prompts import (
    AppConfig,
    BarColumn,
    Console,
    OpenAICompatibleLLMClient,
    Panel,
    Progress,
    RelayDirectionBuilder,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    build_arg_parser,
    coerce_local_path,
    console,
    main as _package_main,
)


def main():
    for name in (
        "AppConfig", "OpenAICompatibleLLMClient", "RelayDirectionBuilder", "coerce_local_path", "console",
        "Panel", "Progress", "SpinnerColumn", "TextColumn", "BarColumn", "TimeElapsedColumn", "TimeRemainingColumn",
    ):
        setattr(_cli, name, globals()[name])
    return _package_main()


if __name__ == "__main__":
    main()
