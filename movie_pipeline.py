"""Root facade for the movie pipeline.

Public API preserved for compatibility with existing callers and tests.

New code should import from the canonical locations instead:

- ``feverslop.cli.movie_cli.build_movie_arg_parser``  (argparse)
- ``feverslop.cli.movie_cli.config_from_args``  (CLI -> config wiring)
- ``feverslop.composition.movie_pipeline.run``  (pipeline execution)
- ``feverslop.composition.movie_pipeline.MoviePipelineResult``  (result type)
- ``feverslop.composition.movie_pipeline.main``  (CLI entry point)
"""
from __future__ import annotations

# CLI argument parsing (lives in cli/)
from feverslop.cli.movie_cli import (
    build_movie_arg_parser as build_arg_parser,  # noqa: F401
)
from feverslop.cli.movie_cli import config_from_args  # noqa: F401

# Pipeline execution and result type (lives in composition/)
from feverslop.composition.movie_pipeline import (
    MoviePipelineResult,  # noqa: F401
    main,
    run,  # noqa: F401
)

if __name__ == "__main__":
    main()
