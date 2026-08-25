from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from feverslop.path_utils import coerce_local_path
from feverslop.prompting.ltx_prompt_anchor_fixer import (
    LTXPromptAnchorFixer,
    validate_anchor_file,
)

console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fix render_plan LTX prompts so I2V preserves the Z-Image startframe composition.",
    )
    parser.add_argument("--input-render-plan", required=True)
    parser.add_argument("--output-render-plan", required=True)
    parser.add_argument(
        "--subject-anchor",
        default=(
            "the old weary warrior man with a weathered scarred face, salt-and-pepper beard, "
            "tattered leather armor, and a heavy frayed cloak"
        ),
    )
    parser.add_argument("--max-base-prompt-chars", type=int, default=1200)
    parser.add_argument("--max-relay-chars", type=int, default=260)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    fixer = LTXPromptAnchorFixer(
        subject_anchor=args.subject_anchor,
        max_base_prompt_chars=args.max_base_prompt_chars,
        max_relay_chars=args.max_relay_chars,
    )

    input_render_plan = coerce_local_path(args.input_render_plan)
    output_render_plan = coerce_local_path(args.output_render_plan)

    output = fixer.fix_file(input_render_plan=input_render_plan, output_render_plan=output_render_plan)

    warnings = validate_anchor_file(output, subject_hint=args.subject_anchor)

    console.print(Panel.fit(
        f"[bold]LTX Prompt Anchors Fixed[/bold]\n\n"
        f"Input: [cyan]{input_render_plan}[/cyan]\n"
        f"Output: [cyan]{output}[/cyan]\n"
        f"Warnings: [yellow]{len(warnings)}[/yellow]",
        title="Done",
        border_style="green",
    ))

    for warning in warnings[:30]:
        console.print(f"[yellow]![/yellow] {warning}")

    if len(warnings) > 30:
        console.print(f"[yellow]... {len(warnings) - 30} more warnings[/yellow]")


if __name__ == "__main__":
    main()

