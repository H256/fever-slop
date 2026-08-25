from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.config.app_config import AppConfig
from feverslop.path_utils import coerce_local_path
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.utils.rich_progress import build_progress
from feverslop.utils.render_plan_selection import load_render_plan_subset, parse_scene_list

console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Z-Image storyboard startframes from render_plan.json.",
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


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    app_config_path = coerce_local_path(args.app_config)
    render_plan_path = coerce_local_path(args.render_plan)
    workflow_path = coerce_local_path(args.workflow)
    output_dir = coerce_local_path(args.output_dir)
    app_config = AppConfig.load(app_config_path)

    console.print(Panel.fit(
        f"[bold]Storyboard Renderer[/bold]\n\n"
        f"ComfyUI: [cyan]{app_config.comfyui.base_url}[/cyan]\n"
        f"Render plan: [cyan]{render_plan_path}[/cyan]\n"
        f"Workflow: [cyan]{workflow_path}[/cyan]\n"
        f"Output: [cyan]{output_dir}[/cyan]\n"
        f"LoRA 1 strength: [yellow]{args.character_lora_strength if args.character_lora_strength is not None else 'workflow default'}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    scene_numbers = parse_scene_list(args.scenes)
    planned = load_render_plan_subset(render_plan_path, scene_numbers, args.limit)
    use_case = build_render_storyboard_use_case(
        app_config=app_config,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )

    with build_progress(console=console) as progress:
        task = progress.add_task("Rendering storyboard startframes", total=len(planned))

        rendered = use_case.execute(
            RenderStoryboardRequest(
                render_plan_path=render_plan_path,
                workflow_path=workflow_path,
                output_dir=output_dir,
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
                on_frame_complete=lambda _output, completed, _total: progress.update(
                    task,
                    completed=completed,
                ),
            ),
        )
        progress.update(task, completed=len(rendered))

    console.print(
        f"[green]OK[/green] Rendered/available storyboard frames: "
        f"[yellow]{len(rendered)}[/yellow]",
    )

    for path in rendered:
        console.print(f"[cyan]{path}[/cyan]")


if __name__ == "__main__":
    main()

