from __future__ import annotations

from pathlib import Path
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, TimeElapsedColumn

from feverslop.config.app_config import AppConfig
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.ports.rendering import WorkflowAnchorConfig


console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Z-Image storyboard startframes from render_plan.json."
    )
    parser.add_argument("--app-config", default="./app_config.json")
    parser.add_argument("--render-plan", required=True)
    parser.add_argument("--workflow", required=True, help="ComfyUI Z-Image API workflow JSON")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenes", default=None, help="Example: 1,2,5-8")
    parser.add_argument("--no-skip-existing", action="store_true")

    parser.add_argument("--character-lora-strength", type=float, default=None)
    parser.add_argument("--negative-prompt", default="")

    parser.add_argument("--positive-title", default="#PROMPT_POSITIVE")
    parser.add_argument("--negative-title", default="#PROMPT_NEGATIVE")
    parser.add_argument("--save-title", default="#SAVE_IMAGE")
    parser.add_argument("--character-lora-title", default="#LORA_1")
    return parser


def parse_scene_list(value: str | None) -> set[int] | None:
    if not value:
        return None

    result = set()

    for part in value.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            result.update(range(start, end + 1))
        else:
            result.add(int(part))

    return result


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    app_config = AppConfig.load(args.app_config)

    console.print(Panel.fit(
        f"[bold]Storyboard Renderer[/bold]\n\n"
        f"ComfyUI: [cyan]{app_config.comfyui.base_url}[/cyan]\n"
        f"Render plan: [cyan]{args.render_plan}[/cyan]\n"
        f"Workflow: [cyan]{args.workflow}[/cyan]\n"
        f"Output: [cyan]{args.output_dir}[/cyan]\n"
        f"LoRA 1 strength: [yellow]{args.character_lora_strength if args.character_lora_strength is not None else 'workflow default'}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    scene_numbers = parse_scene_list(args.scenes)
    use_case = build_render_storyboard_use_case(
        app_config=app_config,
        workflow_path=args.workflow,
        output_dir=args.output_dir,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task("Rendering storyboard startframes...", total=None)

        rendered = use_case.execute(
            RenderStoryboardRequest(
                render_plan_path=Path(args.render_plan),
                workflow_path=Path(args.workflow),
                output_dir=Path(args.output_dir),
                limit=args.limit,
                scene_numbers=scene_numbers,
                skip_existing=not args.no_skip_existing,
                negative_prompt=args.negative_prompt,
                character_lora_strength=args.character_lora_strength,
                anchors=WorkflowAnchorConfig(
                    positive_prompt_title=args.positive_title,
                    negative_prompt_title=args.negative_title,
                    save_image_title=args.save_title,
                    character_lora_title=args.character_lora_title,
                ),
            )
        )

    console.print(
        f"[green]âœ“[/green] Rendered/available storyboard frames: "
        f"[yellow]{len(rendered)}[/yellow]"
    )

    for path in rendered:
        console.print(f"[cyan]{path}[/cyan]")


if __name__ == "__main__":
    main()
