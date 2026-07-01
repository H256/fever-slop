from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
from feverslop.application.reference_bible import ReferenceBibleGenerator, ReferenceLocation, ReferenceSubject
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.ports.rendering import WorkflowAnchorConfig


console = Console()
MSR_ACTOR_VIEW_NAMES = ReferenceBibleGenerator.direct_msr_actor_view_names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render FeverSlop Actor and Location Bible reference sheets.")
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--app-config", default="app_config.json")
    parser.add_argument("--hero-workflow", required=True)
    parser.add_argument("--edit-workflow", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hero-positive-title", default="#PROMPT_POSITIVE")
    parser.add_argument("--edit-positive-title", default="#PROMPT_POSITIVE")
    parser.add_argument("--reference-image-title", default="#IMAGE_1")
    parser.add_argument(
        "--view-set",
        choices=["msr", "full"],
        default="msr",
        help="msr renders direct 4-panel actor sheets and hero location references; full renders hero plus edit views and sheets.",
    )
    return parser


def load_reference_subjects(project_config_path: str | Path) -> tuple[list[ReferenceSubject], list[ReferenceLocation]]:
    config = ProjectConfig.load(project_config_path)
    resolved_context = _load_resolved_context(config)
    actor_source = resolved_context.get("actors") or config.actors
    location_source = resolved_context.get("structured_locations") or config.structured_locations
    subjects = [
        ReferenceSubject(
            id=_item_value(actor, "id"),
            name=_item_value(actor, "name"),
            role=_item_value(actor, "role"),
            visual_description=_item_value(actor, "visual_description"),
            image_prompt=(
                _item_value(actor, "image_prompt")
                or _item_value(actor, "visual_description")
                or _item_value(actor, "name")
            ),
        )
        for actor in actor_source
    ]
    if not subjects and config.subject.strip():
        subjects.append(
            ReferenceSubject(
                id="subject",
                name="Subject",
                visual_description=config.subject,
                image_prompt=config.subject,
            )
        )

    locations = [
        ReferenceLocation(
            id=_item_value(location, "id"),
            name=_item_value(location, "name"),
            visual_description=_item_value(location, "visual_description"),
            image_prompt=(
                _item_value(location, "image_prompt")
                or _item_value(location, "visual_description")
                or _item_value(location, "name")
            ),
        )
        for location in location_source
    ]
    return subjects, locations


def _load_resolved_context(config: ProjectConfig) -> dict:
    prompts_dir = config.project_dir / "output" / "prompts"
    candidates = [
        prompts_dir / f"resolved_context_{config.song_id}.json",
        *sorted(prompts_dir.glob("resolved_context_*.json")),
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def _item_value(item, name: str) -> str:
    if isinstance(item, dict):
        return str(item.get(name, "") or "").strip()
    return str(getattr(item, name, "") or "").strip()


def run(args: argparse.Namespace) -> list[Path]:
    app_config = AppConfig.load(args.app_config)
    project_config = ProjectConfig.load(args.project_config)
    output_dir = Path(args.output_dir) if args.output_dir else project_config.project_dir / "output" / "references"

    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    hero_backend = ComfyUIImageBackend(
        client=client,
        workflow_path=args.hero_workflow,
        output_dir=output_dir,
        model_resolver=model_resolver,
    )
    edit_backend = ComfyUIImageBackend(
        client=client,
        workflow_path=args.edit_workflow,
        output_dir=output_dir,
        model_resolver=model_resolver,
    )
    hero_anchors = WorkflowAnchorConfig(positive_prompt_title=args.hero_positive_title)
    edit_anchors = WorkflowAnchorConfig(
        positive_prompt_title=args.edit_positive_title,
        reference_image_title=args.reference_image_title,
    )
    subjects, locations = load_reference_subjects(args.project_config)
    if not subjects and not locations:
        raise ValueError(
            "No reference actors or locations found. Run the prompt/render-plan step first "
            "so output/prompts/resolved_context_<song>.json exists, or add subject/actors/locations "
            "to the project config."
        )

    actor_view_names, location_view_names = resolve_view_names(args.view_set)
    total_views = (len(subjects) * len(actor_view_names)) + (len(locations) * len(location_view_names))
    console.print(
        "[bold cyan]Reference Bible render plan[/bold cyan]\n"
        f"Project: [cyan]{project_config.project_name}[/cyan]\n"
        f"Output: [cyan]{output_dir}[/cyan]\n"
        f"Actors: [yellow]{len(subjects)}[/yellow]  "
        f"Locations: [yellow]{len(locations)}[/yellow]  "
        f"Actor views: [yellow]{len(actor_view_names)}[/yellow]  "
        f"Location views: [yellow]{len(location_view_names)}[/yellow]  "
        f"Total renders: [yellow]{total_views}[/yellow]"
    )

    manifests: list[Path] = []
    current_task_id = None

    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    run._last_progress_columns = columns
    with Progress(
        *columns,
        console=console,
    ) as progress:
        total_task_id = progress.add_task("Rendering reference views", total=total_views)

        def on_view_complete(event: dict) -> None:
            nonlocal current_task_id
            progress.update(
                total_task_id,
                advance=1,
                description=(
                    f"{event['kind']} {event['id']} "
                    f"{event['item_completed']}/{event['item_total']} {event['view']}"
                ),
            )
            if current_task_id is not None:
                progress.update(current_task_id, advance=1)

        generator = ReferenceBibleGenerator(
            backend=hero_backend,
            edit_backend=edit_backend,
            output_dir=output_dir,
            hero_anchors=hero_anchors,
            edit_anchors=edit_anchors,
            on_view_complete=on_view_complete,
            actor_view_names=actor_view_names,
            location_view_names=location_view_names,
            msr_sheet_size=(project_config.video.width, project_config.video.height),
        )

        for subject in subjects:
            current_task_id = progress.add_task(
                f"Actor {subject.id}",
                total=len(actor_view_names),
            )
            manifest = generator.generate_subject_bible(subject)
            manifests.append(manifest)
            progress.update(current_task_id, completed=len(actor_view_names))
            console.print(f"[green]OK[/green] Actor Bible: [cyan]{manifest}[/cyan]")
        for location in locations:
            current_task_id = progress.add_task(
                f"Location {location.id}",
                total=len(location_view_names),
            )
            manifest = generator.generate_location_bible(location)
            manifests.append(manifest)
            progress.update(current_task_id, completed=len(location_view_names))
            console.print(f"[green]OK[/green] Location Bible: [cyan]{manifest}[/cyan]")
        current_task_id = None
    return manifests


def resolve_view_names(view_set: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if view_set == "msr":
        return MSR_ACTOR_VIEW_NAMES, ("hero",)
    return ReferenceBibleGenerator.view_names, ReferenceBibleGenerator.view_names


def main() -> None:
    try:
        run(build_arg_parser().parse_args())
    except ValueError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
