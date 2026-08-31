from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.config.app_config import AppConfig
from feverslop.path_utils import coerce_local_path
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder

console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact verbose LTX PromptRelay prompts into short direction prompts.",
    )
    parser.add_argument("--app-config", default="./app_config.json")
    parser.add_argument("--input-render-plan", required=True)
    parser.add_argument("--output-render-plan", required=True)
    parser.add_argument("--max-words", type=int, default=28)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    input_render_plan = coerce_local_path(args.input_render_plan)
    output_render_plan = coerce_local_path(args.output_render_plan)
    app_config = AppConfig.load(coerce_local_path(args.app_config))

    console.print(Panel.fit(
        f"[bold]Relay Direction Builder[/bold]\n\n"
        f"Input: [cyan]{input_render_plan}[/cyan]\n"
        f"Output: [cyan]{output_render_plan}[/cyan]\n"
        f"LLM: [yellow]{app_config.llm.model}[/yellow] @ [cyan]{app_config.llm.base_url}[/cyan]\n"
        f"Max words: [yellow]{args.max_words}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        api_key=app_config.llm.api_key,
        model=app_config.llm.model,
        temperature=app_config.llm.temperature,
        dspy_temperature=app_config.llm.dspy_temperature,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        chat_template_kwargs=app_config.llm.chat_template_kwargs,
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
            input_render_plan=input_render_plan,
            output_render_plan=output_render_plan,
        )
        progress.update(task, completed=1)

    console.print(f"[green]OK[/green] Compact render plan: [cyan]{output}[/cyan]")


if __name__ == "__main__":
    main()

