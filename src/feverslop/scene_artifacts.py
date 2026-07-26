from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SceneArtifactLayout:
    project_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_dir", Path(self.project_dir))

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output"

    @property
    def references_dir(self) -> Path:
        return self.output_dir / "references"

    @property
    def render_dir(self) -> Path:
        return self.output_dir / "render"

    @property
    def plans_dir(self) -> Path:
        return self.render_dir / "plans"

    @property
    def base_plan(self) -> Path:
        return self.plans_dir / "base.json"

    @property
    def references_plan(self) -> Path:
        return self.plans_dir / "references.json"

    @property
    def ingredients_plan(self) -> Path:
        return self.plans_dir / "ingredients.json"

    @property
    def compact_plan(self) -> Path:
        return self.plans_dir / "compact.json"

    @property
    def anchored_plan(self) -> Path:
        return self.plans_dir / "anchored.json"

    @property
    def scenes_dir(self) -> Path:
        return self.render_dir / "scenes"

    def scene_dir(self, scene_number: int) -> Path:
        return self.scenes_dir / f"scene_{scene_number:04d}"

    def scene_manifest(self, scene_number: int) -> Path:
        return self.scene_dir(scene_number) / "manifest.json"

    def scene_workflow(self, scene_number: int) -> Path:
        return self.scene_dir(scene_number) / "workflow.json"

    def scene_raw_video(self, scene_number: int) -> Path:
        return self.scene_dir(scene_number) / "raw.mp4"

    def scene_final_video(self, scene_number: int) -> Path:
        return self.scene_dir(scene_number) / "final.mp4"

    def scene_workflow_facefix(self, scene_number: int) -> Path:
        return self.scene_dir(scene_number) / "workflow_facefix.json"

    def scene_final_facefix_video(self, scene_number: int) -> Path:
        return self.scene_dir(scene_number) / "final_facefix.mp4"

    @property
    def storyboard_dir(self) -> Path:
        return self.render_dir / "storyboard"

    @property
    def final_dir(self) -> Path:
        return self.render_dir / "final"

    @property
    def video_only(self) -> Path:
        return self.final_dir / "video_only.mp4"

    @property
    def movie(self) -> Path:
        return self.final_dir / "movie.mp4"

    def find_scene_final_video(self, scene_number: int, *, legacy_dirs: Iterable[str | Path] = ()) -> Path | None:
        candidates = [self.scene_final_video(scene_number)]
        for directory in legacy_dirs:
            directory = Path(directory)
            candidates.extend(
                (
                    directory / f"scene_{scene_number:04d}.mp4",
                    directory / "final" / f"scene_{scene_number:04d}.mp4",
                )
            )
        return next((path for path in candidates if path.exists()), None)

    def find_plan(self, canonical_path: str | Path, *, legacy_paths: Iterable[str | Path] = ()) -> Path | None:
        candidates = [Path(canonical_path), *(Path(path) for path in legacy_paths)]
        return next((path for path in candidates if path.exists()), None)

    def actor_sheet_images(self) -> list[Path]:
        """Discover actor reference sheet images under output/references/actors/<id>/views/*sheet.png."""
        actors_dir = self.references_dir / "actors"
        if not actors_dir.is_dir():
            return []
        sheets: list[Path] = []
        for actor_dir in sorted(actors_dir.iterdir()):
            if not actor_dir.is_dir():
                continue
            views = actor_dir / "views"
            if not views.is_dir():
                continue
            for p in sorted(views.glob("*sheet.png")):
                sheets.append(p)
        return sheets
