from __future__ import annotations

from pathlib import Path


class LocalMovieVisualAdapter:
    """Local placeholder adapter for movie production tests and offline Studio use."""

    def render_movie(self, *, project_dir: Path, render_plan_path: Path) -> Path:
        output_dir = project_dir / "output" / "movie"
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / f"{project_dir.name}.mp4"
        final.write_bytes(b"feverslop movie placeholder\n" + render_plan_path.read_bytes())
        return final
