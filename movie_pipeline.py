from __future__ import annotations

from feverslop.composition import movie_pipeline as _movie_pipeline


MoviePipelineResult = _movie_pipeline.MoviePipelineResult
build_arg_parser = _movie_pipeline.build_arg_parser
config_from_args = _movie_pipeline.config_from_args
main = _movie_pipeline.main
run = _movie_pipeline.run


if __name__ == "__main__":
    main()
