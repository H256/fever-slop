from __future__ import annotations

import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.path_utils import coerce_local_path
from feverslop.pipeline.scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_srt_scenes,
    validate_scene_durations,
)

console = Console()


def ensure_output_writable(output_srt):
    output_srt = coerce_local_path(output_srt)
    target = output_srt if output_srt.exists() else output_srt.parent
    if not target.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_srt.parent}")
    if not os.access(target, os.W_OK):
        raise PermissionError(f"Output SRT is not writable: {output_srt}")


def main():
    parser = argparse.ArgumentParser(
        description="Repair beat scene SRT so every scene respects min/max duration.",
    )
    parser.add_argument("--input-srt", required=True)
    parser.add_argument("--output-srt", required=True)
    parser.add_argument("--min-duration", type=float, required=True)
    parser.add_argument("--max-duration", type=float, required=True)
    args = parser.parse_args()

    input_srt = coerce_local_path(args.input_srt)
    output_srt = coerce_local_path(args.output_srt)
    ensure_output_writable(output_srt)
    before = parse_srt_scenes(input_srt)
    output = enforce_scene_srt_file(
        input_srt=input_srt,
        output_srt=output_srt,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        artifact_store=JsonArtifactStore(),
    )
    after = parse_srt_scenes(output)

    errors = validate_scene_durations(
        scenes=after,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    table = Table(title="Scene SRT Repair")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="yellow")
    table.add_row("Input scenes", str(len(before)))
    table.add_row("Output scenes", str(len(after)))
    table.add_row("Min duration", f"{args.min_duration:.2f}s")
    table.add_row("Max duration", f"{args.max_duration:.2f}s")
    table.add_row("Output", str(output))
    console.print(table)

    if errors:
        console.print(Panel("\n".join(errors), title="Validation warnings", border_style="yellow"))
    else:
        console.print(Panel.fit("[green]All scene durations are within constraints.[/green]", border_style="green"))


if __name__ == "__main__":
    main()
