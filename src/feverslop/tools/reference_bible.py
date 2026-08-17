from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
from feverslop.adapters.sequence_to_sheet_backend import ComfyUISequenceToSheetBackend
from feverslop.application.reference_bible import ReferenceBibleGenerator, ReferenceLocation, ReferenceSubject
from feverslop.application.reference_sheet_planning import ReferenceSheetPlanner
from feverslop.adapters.llm_client import LocalOpenAIClient
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
    parser.add_argument("--reference-generation", choices=["image_views", "sequence_sheet"], default="image_views")
    parser.add_argument("--sequence-workflow", default="workflows/sequence_to_sheet_minimax_h3_i2va_v1.json")
    parser.add_argument("--only-kind", choices=["actor", "location"], default=None)
    parser.add_argument("--only-id", default=None)
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
    sequence_backend = None
    sequence_planner = None
    if args.reference_generation == "sequence_sheet":
        sequence_backend = ComfyUISequenceToSheetBackend(
            client=client,
            workflow_path=args.sequence_workflow,
            backend="minimax",
            model_resolver=model_resolver,
        )
        sequence_planner = ReferenceSheetPlanner(
            llm=LocalOpenAIClient(
                base_url=app_config.llm.base_url,
                api_key=app_config.llm.api_key,
                model=app_config.llm.model_for("structured"),
                temperature=app_config.llm.temperature,
                dspy_temperature=app_config.llm.dspy_temperature,
                max_tokens=app_config.llm.max_tokens,
                request_timeout_seconds=app_config.llm.request_timeout_seconds,
                dspy_cache=app_config.llm.dspy_cache,
                max_concurrent_requests=app_config.llm.max_concurrent_requests,
            )
        )
    hero_anchors = WorkflowAnchorConfig(positive_prompt_title=args.hero_positive_title)
    edit_anchors = WorkflowAnchorConfig(
        positive_prompt_title=args.edit_positive_title,
        reference_image_title=args.reference_image_title,
    )
    subjects, locations = load_reference_subjects(args.project_config)
    if args.only_kind == "actor":
        subjects = [subject for subject in subjects if subject.id == args.only_id]
        locations = []
    elif args.only_kind == "location":
        subjects = []
        locations = [location for location in locations if location.id == args.only_id]
    if not subjects and not locations:
        raise ValueError(
            "No reference actors or locations found. Run the prompt/render-plan step first "
            "so output/prompts/resolved_context_<song>.json exists, or add subject/actors/locations "
            "to the project config."
        )

    actor_view_names, location_view_names = resolve_view_names(args.view_set)
    actor_work = 1 if sequence_backend is not None else len(actor_view_names)
    location_work = 1 if sequence_backend is not None else len(location_view_names)
    total_views = (len(subjects) * actor_work) + (len(locations) * location_work)
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

        if sequence_backend is not None:
            console.print(
                "[cyan]Reference phases:[/cyan] each asset runs the anchor image first, "
                "then the MiniMax sequence."
            )

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

        def on_sequence_phase(event: dict) -> None:
            labels = {
                "anchor_start": "Krea anchor startet",
                "anchor_complete": "Krea anchor fertig",
                "sequence_start": "MiniMax-Sequenz startet",
                "sequence_complete": "MiniMax-Sequenz fertig",
            }
            label = labels.get(event["phase"], event["phase"])
            suffix = f": {event['path']}" if event.get("path") else ""
            console.print(f"[cyan]{event['kind']} {event['id']}[/cyan] {label}{suffix}")

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
            reference_image_size=project_config.reference_images.resolve(project_config.video),
            sequence_backend=sequence_backend,
            sequence_planner=sequence_planner,
            visual_style=project_config.style,
            on_sequence_phase=on_sequence_phase,
        )

        for subject in subjects:
            console.print(f"[cyan]Starting Krea anchor + MiniMax sequence: actor {subject.id}[/cyan]")
            current_task_id = progress.add_task(
                f"Actor {subject.id}",
                total=actor_work,
            )
            manifest = generator.generate_subject_bible(subject)
            manifests.append(manifest)
            progress.update(current_task_id, completed=actor_work)
            console.print(f"[green]OK[/green] Actor Bible: [cyan]{manifest}[/cyan]")
        for location in locations:
            console.print(f"[cyan]Starting Krea anchor + MiniMax sequence: location {location.id}[/cyan]")
            current_task_id = progress.add_task(
                f"Location {location.id}",
                total=location_work,
            )
            manifest = generator.generate_location_bible(location)
            manifests.append(manifest)
            progress.update(current_task_id, completed=location_work)
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
