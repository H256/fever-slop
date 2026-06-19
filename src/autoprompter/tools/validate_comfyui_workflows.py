from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich.console import Console

from autoprompter.adapters.comfyui_client import ComfyUIClient
from autoprompter.adapters.comfyui_model_resolver import (
    ComfyUIModelOverride,
    ComfyUIModelResolutionError,
    ComfyUIModelResolver,
)
from autoprompter.config.app_config import AppConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate ComfyUI workflow model references against the configured server dropdown values."
    )
    parser.add_argument("--app-config", default="app_config.json")
    parser.add_argument("--workflows-dir", default="workflows")
    return parser


def validate_comfyui_workflows(
    *,
    client,
    workflows_dir: str | Path,
    overrides: list[ComfyUIModelOverride],
) -> list[dict]:
    resolver = ComfyUIModelResolver(client, overrides=overrides)
    return resolver.validate_workflow_directory(workflows_dir)


def main() -> None:
    args = build_arg_parser().parse_args()
    console = Console()
    app_config = AppConfig.load(args.app_config)
    client = ComfyUIClient(base_url=app_config.comfyui.base_url)

    try:
        reports = validate_comfyui_workflows(
            client=client,
            workflows_dir=args.workflows_dir,
            overrides=app_config.comfyui.model_overrides,
        )
    except ComfyUIModelResolutionError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        sys.exit(1)

    for report in reports:
        console.print(
            f"[green]OK[/green] {report['workflow']}: "
            f"{report['patched_count']} model reference(s) would be patched"
        )
        for patch in report["patched"]:
            console.print(f"  {patch['node_id']} {patch['input']}: {patch['from']} -> {patch['to']}")


if __name__ == "__main__":
    main()
