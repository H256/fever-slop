from __future__ import annotations

from pathlib import Path
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from autoprompter.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from autoprompter.config.app_config import AppConfig
from relay_direction_builder import RelayDirectionBuilder


console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact verbose LTX PromptRelay prompts into short direction prompts."
    )
    parser.add_argument("--app-config", default="./app_config.json")
    parser.add_argument("--input-render-plan", required=True)
    parser.add_argument("--output-render-plan", required=True)
    parser.add_argument("--max-words", type=int, default=28)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    app_config = AppConfig.load(args.app_config)

    console.print(Panel.fit(
        f"[bold]Relay Direction Builder[/bold]\n\n"
        f"Input: [cyan]{args.input_render_plan}[/cyan]\n"
        f"Output: [cyan]{args.output_render_plan}[/cyan]\n"
        f"LLM: [yellow]{app_config.llm.model}[/yellow] @ [cyan]{app_config.llm.base_url}[/cyan]\n"
        f"Max words: [yellow]{args.max_words}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        model=app_config.llm.model,
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
    )

    builder = RelayDirectionBuilder(
        llm=llm,
        max_words=args.max_words,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Compacting relay prompts", total=None)
        output = builder.compact_render_plan_file(
            input_render_plan=args.input_render_plan,
            output_render_plan=args.output_render_plan,
        )
        progress.update(task, completed=1)

    console.print(f"[green]✓[/green] Compact render plan: [cyan]{output}[/cyan]")


if __name__ == "__main__":
    main()
