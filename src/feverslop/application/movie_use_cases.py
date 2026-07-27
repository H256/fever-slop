from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from feverslop.domain.movie import MovieProject
from feverslop.domain.slug_utils import slugify_project_name
from feverslop.application.movie_common import (
    MovieInput,
    MovieProductionResult,
    MovieScaffoldResult,
    _planner_source_text,
    validate_movie_input,
)
from feverslop.application.movie_memory import (
    build_movie_scene_cards,
    build_movie_shot_cards,
    generate_movie_narrative_plan,
    generate_movie_screenplay,
    generate_movie_story_design,
    movie_narrative_plan_to_dict,
    movie_scene_cards_to_dict,
    movie_screenplay_to_dict,
    movie_screenplay_to_markdown,
    movie_shot_cards_to_dict,
    movie_story_design_to_dict,
)
from feverslop.application.movie_bible import (
    _bible_dict,
    _reference_manifest,
    _render_plan,
    augment_movie_bible_from_shot_references,
    constrain_movie_shots_to_bible,
    generate_movie_bible,
    generate_movie_continuity_plan,
    plan_movie_shots_from_bible,
)
from feverslop.application.movie_continuity import (
    apply_movie_continuity_to_shots,
    movie_continuity_plan_to_dict,
)
from feverslop.ports.movie import (
    MovieArtifactWriter,
    ReferenceGenerationPort,
    VisualGenerationPort,
)
from feverslop.ports.reporting import ConsoleReporter, NullReporter, Reporter


class ScaffoldMovieUseCase:
    def __init__(
        self,
        *,
        planner: Any,
        projects_root: Path,
        artifact_writer: MovieArtifactWriter,
        console: Any | None = None,
        reporter: Reporter | None = None,
    ):
        self.planner = planner
        self.projects_root = Path(projects_root)
        self.artifact_writer = artifact_writer
        if reporter is not None:
            self.reporter = reporter
        elif console is not None:
            self.reporter = ConsoleReporter(console)
        else:
            self.reporter = NullReporter()

    def execute(self, request: MovieInput) -> MovieScaffoldResult:
        validate_movie_input(request)
        slug = slugify_project_name(request.name)
        project_dir = self.projects_root / slug
        movie_dir = project_dir / "movie"
        movie_dir.mkdir(parents=True, exist_ok=False)

        config = dict(request.config or {})

        self.reporter.step("[bold cyan]Step 1/7[/] — Generating story architecture...")
        story_arch = self.planner.generate_story_arch(
            title=request.name,
            source_type=request.source_type,
            story_text=_planner_source_text(request, config),
            desired_length=float(request.desired_length),
        )
        self.reporter.message(f"  Story arch: {len(story_arch.beats)} beats")

        self.reporter.step("[bold cyan]Step 2/7[/] — Generating movie bible...")
        bible = generate_movie_bible(
            planner=self.planner,
            request=request,
            story_arch=story_arch,
            config=config,
        )
        self.reporter.message(f"  Movie bible: {len(bible.actors)} actors, {len(bible.locations)} locations")

        self.reporter.step("[bold cyan]Step 3/7[/] — Generating story design...")
        story_design = generate_movie_story_design(
            planner=self.planner,
            request=request,
            bible=bible,
            story_arch=story_arch,
            config=config,
            source_text=_planner_source_text(request, config),
        )
        self.reporter.message(f"  Story design: {len(story_design.act_structure)} acts, {len(story_design.scene_blueprint)} scenes")

        self.reporter.step("[bold cyan]Step 4/7[/] — Generating screenplay...")
        screenplay = generate_movie_screenplay(
            planner=self.planner,
            request=request,
            bible=bible,
            story_arch=story_arch,
            story_design=story_design,
            config=config,
            source_text=_planner_source_text(request, config),
        )
        self.reporter.message(f"  Screenplay: {len(screenplay.scenes)} scenes")

        self.reporter.step("[bold cyan]Step 5/7[/] — Generating narrative plan...")
        narrative_plan = generate_movie_narrative_plan(
            planner=self.planner,
            request=request,
            bible=bible,
            screenplay=screenplay,
            config=config,
        )
        self.reporter.message(f"  Narrative plan: {len(narrative_plan.sequences)} sequences, {len(narrative_plan.causal_chain)} causal links")

        self.reporter.step("[bold cyan]Step 6/7[/] — Planning movie shots...")
        shots = plan_movie_shots_from_bible(
            planner=self.planner,
            bible=bible,
            screenplay=screenplay,
            desired_length=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            min_duration=float(request.min_scene_duration),
            max_duration=float(request.max_scene_duration),
        )
        self.reporter.message(f"  Planned {len(shots)} shots")

        bible = augment_movie_bible_from_shot_references(bible, shots, config=config)
        shots = constrain_movie_shots_to_bible(shots, bible)

        self.reporter.step("[bold cyan]Step 7/7[/] — Generating continuity plan...")
        continuity_plan = generate_movie_continuity_plan(planner=self.planner, request=request, bible=bible, shots=shots, config=config)
        self.reporter.message(f"  Continuity plan: {len(continuity_plan.narrative_chain)} narrative beats, {len(continuity_plan.scene_continuity)} scenes")
        shots = apply_movie_continuity_to_shots(shots, continuity_plan)
        scene_cards = build_movie_scene_cards(screenplay=screenplay, shots=shots)
        shot_cards = build_movie_shot_cards(shots=shots, scene_cards=scene_cards)
        movie = MovieProject(
            slug=slug,
            name=request.name,
            bible=bible,
            story_arch=story_arch,
            shots=shots,
            duration_seconds=float(request.desired_length),
            width=int(request.width),
            height=int(request.height),
            mode=request.mode,
            config=config,
        )

        metadata = {
            "project_type": "movie",
            "display_name": request.name,
            "slug": slug,
            "movie": {
                "source_type": request.source_type,
                "story_text": request.story_text,
                "desired_length": float(request.desired_length),
                "width": int(request.width),
                "height": int(request.height),
                "mode": request.mode,
            },
        }
        writer = self.artifact_writer
        writer.write_json(project_dir / ".studio" / "project.json", metadata)

        story_arch_path = movie_dir / "story_arch.json"
        bible_path = movie_dir / "bible.json"
        story_design_path = movie_dir / "story_design.json"
        screenplay_path = movie_dir / "screenplay.json"
        screenplay_md_path = movie_dir / "screenplay.md"
        narrative_plan_path = movie_dir / "narrative_plan.json"
        scene_cards_path = movie_dir / "scene_cards.json"
        continuity_plan_path = movie_dir / "continuity_plan.json"
        shot_cards_path = movie_dir / "shot_cards.json"
        render_plan_path = movie_dir / "render_plan.json"
        reference_manifest_path = movie_dir / "references" / "manifest.json"
        writer.write_json(story_arch_path, asdict(movie.story_arch))
        writer.write_json(bible_path, _bible_dict(movie.bible))
        writer.write_json(story_design_path, movie_story_design_to_dict(story_design))
        writer.write_json(screenplay_path, movie_screenplay_to_dict(screenplay))
        writer.write_text(screenplay_md_path, movie_screenplay_to_markdown(screenplay))
        writer.write_json(narrative_plan_path, movie_narrative_plan_to_dict(narrative_plan))
        writer.write_json(scene_cards_path, movie_scene_cards_to_dict(scene_cards))
        writer.write_json(continuity_plan_path, movie_continuity_plan_to_dict(continuity_plan))
        writer.write_json(shot_cards_path, movie_shot_cards_to_dict(shot_cards))
        if config:
            writer.write_json(project_dir / "config.json", config)
        writer.write_json(render_plan_path, _render_plan(movie, shot_cards=shot_cards))
        writer.write_json(reference_manifest_path, _reference_manifest(movie))
        return MovieScaffoldResult(
            slug,
            project_dir,
            bible_path,
            story_arch_path,
            render_plan_path,
            reference_manifest_path,
            story_design_path,
            screenplay_path,
            narrative_plan_path,
            scene_cards_path,
            shot_cards_path,
        )


class AutoProduceMovieUseCase:
    def __init__(self, *, scaffold: ScaffoldMovieUseCase, visual_backend: VisualGenerationPort, reference_generator: ReferenceGenerationPort | None = None):
        self.scaffold = scaffold
        self.visual_backend = visual_backend
        self.reference_generator = reference_generator

    def execute(self, request: MovieInput) -> MovieProductionResult:
        scaffolded = self.scaffold.execute(request)
        if self.reference_generator is not None:
            self.reference_generator.generate(project_dir=scaffolded.project_dir)
        final_video = self.visual_backend.render_movie(
            project_dir=scaffolded.project_dir,
            render_plan_path=scaffolded.render_plan_path,
        )
        return MovieProductionResult(
            project_slug=scaffolded.project_slug,
            project_dir=scaffolded.project_dir,
            bible_path=scaffolded.bible_path,
            story_arch_path=scaffolded.story_arch_path,
            render_plan_path=scaffolded.render_plan_path,
            reference_manifest_path=scaffolded.reference_manifest_path,
            story_design_path=scaffolded.story_design_path,
            screenplay_path=scaffolded.screenplay_path,
            narrative_plan_path=scaffolded.narrative_plan_path,
            scene_cards_path=scaffolded.scene_cards_path,
            shot_cards_path=scaffolded.shot_cards_path,
            final_video_path=final_video,
        )



