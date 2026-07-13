from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feverslop.application.movie import build_movie_actor_reference_prompt, build_movie_actor_visual_description
from feverslop.application.reference_bible import ReferenceBibleGenerator, ReferenceLocation, ReferenceSubject
from feverslop.ports.rendering import WorkflowAnchorConfig


class MovieReferenceSheetGenerator:
    def __init__(
        self,
        *,
        backend,
        edit_backend=None,
        hero_anchors: WorkflowAnchorConfig = WorkflowAnchorConfig(),
        edit_anchors: WorkflowAnchorConfig = WorkflowAnchorConfig(),
    ):
        self.backend = backend
        self.edit_backend = edit_backend or backend
        self.hero_anchors = hero_anchors
        self.edit_anchors = edit_anchors

    def generate(self, *, project_dir: Path) -> Path:
        project_dir = Path(project_dir)
        manifest_path = project_dir / "movie" / "references" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_dir = project_dir / "movie" / "references"
        generator = ReferenceBibleGenerator(
            backend=self.backend,
            edit_backend=self.edit_backend,
            output_dir=output_dir,
            hero_anchors=self.hero_anchors,
            edit_anchors=self.edit_anchors,
            actor_view_names=ReferenceBibleGenerator.direct_msr_actor_view_names,
            location_view_names=("hero",),
            msr_sheet_size=_movie_reference_size(project_dir),
            direct_msr_sheet_prompt_builder=build_movie_direct_msr_sheet_prompt,
        )

        for actor in manifest.get("actors") or []:
            subject = _subject_from_manifest(actor)
            actor["visual_description"] = subject.visual_description
            actor["image_prompt"] = build_movie_actor_reference_prompt(subject.name, subject.visual_description)
            actor["prompt"] = actor["image_prompt"]
            path = generator.generate_subject_bible(subject)
            actor_manifest = json.loads(path.read_text(encoding="utf-8"))
            actor["msr_sheet_path"] = _project_reference_path(actor_manifest.get("msr_input_path") or actor_manifest.get("sheet_path") or "")
            actor["sheet_path"] = _project_reference_path(actor_manifest.get("sheet_path") or "")

        for location in manifest.get("locations") or []:
            path = generator.generate_location_bible(_location_from_manifest(location))
            location_manifest = json.loads(path.read_text(encoding="utf-8"))
            location["msr_sheet_path"] = _project_reference_path(location_manifest.get("msr_background_path") or location_manifest.get("sheet_path") or "")
            location["sheet_path"] = _project_reference_path(location_manifest.get("sheet_path") or "")

        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def _subject_from_manifest(actor: dict[str, Any]) -> ReferenceSubject:
    visual_description = str(actor.get("visual_description") or actor.get("prompt") or actor.get("image_prompt") or actor.get("name") or "").strip()
    visual_description = build_movie_actor_visual_description(visual_description)
    image_prompt = str(actor.get("image_prompt") or actor.get("prompt") or "").strip()
    return ReferenceSubject(
        id=str(actor["id"]),
        name=str(actor.get("name") or actor["id"]),
        role=str(actor.get("role") or ""),
        visual_description=visual_description,
        image_prompt=image_prompt,
    )


def _location_from_manifest(location: dict[str, Any]) -> ReferenceLocation:
    prompt = str(location.get("prompt") or location.get("image_prompt") or location.get("visual_description") or location.get("name") or "").strip()
    return ReferenceLocation(
        id=str(location["id"]),
        name=str(location.get("name") or location["id"]),
        visual_description=prompt,
        image_prompt=prompt,
    )


def build_movie_direct_msr_sheet_prompt(subject: ReferenceSubject) -> str:
    base = subject.visual_description or subject.name
    return build_movie_actor_reference_prompt(subject.name, base)


def _movie_reference_size(project_dir: Path) -> tuple[int, int]:
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        return (1280, 704)
    plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
    resolution = plan.get("resolution") or {}
    return (int(resolution.get("width") or 1280), int(resolution.get("height") or 704))


def _project_reference_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    if not path or Path(path).is_absolute() or path.startswith("movie/"):
        return path
    return f"movie/references/{path}"


def _resolve_reference_path(value: str) -> str:
    return _project_reference_path(value)
