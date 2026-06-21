from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich.console import Console

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.config.app_config import AppConfig
from feverslop.config.comfyui import ComfyUIModelOverride


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
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )

    reports = validate_comfyui_workflows(
        client=client,
        workflows_dir=args.workflows_dir,
        overrides=app_config.comfyui.model_overrides,
    )

    has_errors = False
    for report in reports:
        if report["errors"]:
            has_errors = True
            console.print(f"[red]ERROR[/red] {report['workflow']}")
            for error in report["errors"]:
                console.print(f"  {error}")
            continue

        console.print(
            f"[green]OK[/green] {report['workflow']}: "
            f"{report['patched_count']} model reference(s) would be patched"
        )
        for patch in report["patched"]:
            console.print(f"  {patch['node_id']} {patch['input']}: {patch['from']} -> {patch['to']}")

    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
