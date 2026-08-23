#!/usr/bin/env python
"""Scaffold a new FeverSlop movie project from screenplay.md.

Usage:
    uv run python scaffold_movie.py --name "My Film" --screenplay screenplay.md
    uv run python scaffold_movie.py --name "My Film" --story-text "A brief synopsis..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from feverslop.adapters.movie_artifact_writer import LocalMovieArtifactWriter
from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
from feverslop.path_utils import coerce_local_path
from feverslop.composition.project_repository import build_movie_planner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold a new movie project (creates render_plan.json and all planning artefacts).",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Project display name (slug is derived, e.g. 'My Film' -> projects/my-film/).",
    )
    parser.add_argument(
        "--screenplay",
        default=None,
        help="Path to an existing screenplay.md file.",
    )
    parser.add_argument(
        "--story-text",
        default=None,
        help="Short story synopsis (alternative to --screenplay).",
    )
    parser.add_argument(
        "--desired-length",
        type=float,
        default=120.0,
        help="Desired video length in seconds (default: 120).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Output height (default: 704).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum scene duration in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum scene duration in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--planner-backend",
        choices=["llm", "deterministic"],
        default="llm",
        help="Planner backend: llm (default) or deterministic.",
    )
    parser.add_argument(
        "--projects-root",
        default="projects",
        help="Projects root directory (default: projects).",
    )
    parser.add_argument(
        "--refine-actors",
        action="store_true",
        default=False,
        help="Use LLM to refine actor visual descriptions with specific physical details.",
    )
    args = parser.parse_args()

    # Determine source text and type
    story_text = ""
    source_type = "short_story"

    if args.screenplay and args.story_text:
        print("Error: specify only one of --screenplay or --story-text", file=sys.stderr)
        sys.exit(1)

    if args.screenplay:
        sp_path = Path(args.screenplay)
        if not sp_path.exists():
            print(f"Error: screenplay file not found: {sp_path}", file=sys.stderr)
            sys.exit(1)
        story_text = sp_path.read_text(encoding="utf-8-sig")
        source_type = "screenplay"
    elif args.story_text:
        story_text = args.story_text
    else:
        print("Error: specify --screenplay or --story-text", file=sys.stderr)
        sys.exit(1)

    from feverslop.application.movie import slugify_project_name

    projects_root = coerce_local_path(args.projects_root).resolve()
    name = args.name
    slug = slugify_project_name(name)
    target_dir = projects_root / slug

    print(f"Scaffolding movie project: {name}")
    print(f"  Slug:          {slug}")
    print(f"  Project dir:   {target_dir}")
    print(f"  Source type:   {source_type}")
    print(f"  Planner:       {args.planner_backend}")
    print(f"  Length:        {args.desired_length}s")
    print(f"  Resolution:    {args.width}x{args.height}")
    print()

    if target_dir.exists():
        print(f"Error: project directory already exists: {target_dir}", file=sys.stderr)
        sys.exit(1)

    planner = build_movie_planner({"planner_backend": args.planner_backend})

    use_case = ScaffoldMovieUseCase(
        planner=planner,
        projects_root=projects_root,
        artifact_writer=LocalMovieArtifactWriter(),
    )

    result = use_case.execute(
        MovieInput(
            name=name,
            source_type=source_type,
            story_text=story_text,
            desired_length=args.desired_length,
            width=args.width,
            height=args.height,
            mode="scaffold",
            min_scene_duration=args.min_duration,
            max_scene_duration=args.max_duration,
            config={
                "project_name": name,
                "input_audio": "",
                "silent_mode": False,
                "lyrics": "",
                "dialogue_language": "en",
                "video": {
                    "fps": 24,
                    "width": args.width,
                    "height": args.height,
                },
                "video_pipeline": "ltx_msr",
                "scene_generation": {
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "bias": 0.7,
                    "duration_preset": "impact_weighted",
                    "seed": -1,
                },
                "planner_backend": args.planner_backend,
                "refine_actor_prompts": args.refine_actors,
            },
        ),
    )

    print()
    print("Scaffold complete!")
    print(f"  Bible:          {result.bible_path}")
    print(f"  Render plan:    {result.render_plan_path}")
    print(f"  Screenplay:     {result.screenplay_path}")
    print(f"  Screenplay MD:  {result.project_dir / 'movie' / 'screenplay.md'}")
    print(f"  Reference manifest: {result.reference_manifest_path}")
    print()
    print("Next step: run the movie pipeline with:")
    print(f'  uv run python movie_pipeline.py "{result.project_dir}" --movie-video-workflow ingredients')


if __name__ == "__main__":
    main()
