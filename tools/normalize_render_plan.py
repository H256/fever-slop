from __future__ import annotations

import argparse
from pathlib import Path
import json

from rich.console import Console
from rich.panel import Panel

from feverslop.path_utils import coerce_local_path
from tools.render_plan_normalizer import normalize_render_plan_file


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Repair render-plan scenes using scene_generation min/max duration."
    )
    parser.add_argument("--input-render-plan", required=True)
    parser.add_argument("--output-render-plan", required=True)
    parser.add_argument("--min-duration", type=float, required=True)
    parser.add_argument("--max-duration", type=float, required=True)
    parser.add_argument("--keep-original-scene-numbers", action="store_true")

    args = parser.parse_args()

    input_render_plan = coerce_local_path(args.input_render_plan)
    output_render_plan = coerce_local_path(args.output_render_plan)
    input_plan = json.loads(input_render_plan.read_text(encoding="utf-8"))
    before = len(input_plan)

    output = normalize_render_plan_file(
        input_render_plan=input_render_plan,
        output_render_plan=output_render_plan,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        renumber=not args.keep_original_scene_numbers,
    )

    output_plan = json.loads(Path(output).read_text(encoding="utf-8"))
    after = len(output_plan)

    console.print(Panel.fit(
        f"[bold]Render Plan Normalized[/bold]\n\n"
        f"Input scenes: [yellow]{before}[/yellow]\n"
        f"Output scenes: [yellow]{after}[/yellow]\n"
        f"Min duration: [yellow]{args.min_duration:.2f}s[/yellow]\n"
        f"Max duration: [yellow]{args.max_duration:.2f}s[/yellow]\n"
        f"Output: [cyan]{output}[/cyan]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
