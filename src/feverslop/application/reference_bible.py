from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
from tempfile import NamedTemporaryFile
import time
import uuid
from typing import Callable, Any

from PIL import Image, ImageDraw, ImageOps

from feverslop.errors import FeverSlopValidationError
from feverslop.domain.prepared_workflow import sha256_file
from feverslop.domain.movie_utils import transition_from_previous
from feverslop.domain.visual_consistency import (
    ReferenceAnchor,
    SceneConsistencyContract,
)
from feverslop.domain.visual_consistency_runtime import (
    ingredients_sheet_signature as _ingredients_sheet_signature,
)
from feverslop.ports.rendering import ImageRenderBackend, ImageRenderRequest, WorkflowAnchorConfig
from feverslop.application.sequence_reference_pipeline import SequenceReferencePipeline, SequenceReferenceRequest
from feverslop.utils.io import atomic_write_json


INGREDIENTS_SHEET_LAYOUT_VERSION = "scene-reference-grid/v1"
_INGREDIENTS_CACHE_LOCK_TIMEOUT_SECONDS = 30.0
_MAX_INGREDIENTS_SOURCE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class IngredientsSourceSnapshot:
    id: str
    type: str
    path: str
    suffix: str
    content: bytes


@dataclass(frozen=True)
class ReferenceSubject:
    id: str
    name: str
    role: str = ""
    visual_description: str = ""
    image_prompt: str = ""


@dataclass(frozen=True)
class ReferenceLocation:
    id: str
    name: str
    visual_description: str = ""
    image_prompt: str = ""


class ReferenceBibleGenerator:
    view_names = ("hero", "front", "left", "right", "closeup")
    actor_hero_size = (1088, 1920)
    location_hero_size = (1920, 1088)
    msr_actor_view_names = ("hero_closeup", "front", "left", "back")
    direct_msr_actor_view_names = ("msr_sheet",)

    def __init__(
        self,
        *,
        backend: ImageRenderBackend,
        output_dir: str | Path,
        edit_backend: ImageRenderBackend | None = None,
        hero_anchors: WorkflowAnchorConfig = WorkflowAnchorConfig(),
        edit_anchors: WorkflowAnchorConfig = WorkflowAnchorConfig(),
        on_view_complete: Callable[[dict[str, Any]], None] | None = None,
        view_names: tuple[str, ...] | None = None,
        actor_view_names: tuple[str, ...] | None = None,
        location_view_names: tuple[str, ...] | None = None,
        msr_sheet_size: tuple[int, int] = (1280, 704),
        reference_image_size: tuple[int, int] | None = None,
        direct_msr_sheet_prompt_builder: Callable[[ReferenceSubject], str] | None = None,
        sequence_backend: Any | None = None,
        sequence_planner: Any | None = None,
        visual_style: str = "",
        on_sequence_phase: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.backend = backend
        self.edit_backend = edit_backend or backend
        self.hero_anchors = hero_anchors
        self.edit_anchors = edit_anchors
        self.on_view_complete = on_view_complete
        self.output_dir = Path(output_dir)
        self.artifact_base_dir = _infer_reference_artifact_base_dir(self.output_dir)
        self.view_names = tuple(view_names or self.view_names)
        self.actor_view_names = tuple(actor_view_names or view_names or self.view_names)
        self.location_view_names = tuple(location_view_names or view_names or self.view_names)
        self.msr_sheet_size = (int(msr_sheet_size[0]), int(msr_sheet_size[1]))
        reference_width, reference_height = reference_image_size or (1920, 1088)
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError("reference_image_size must contain positive dimensions")
        self.actor_hero_size = (int(reference_height), int(reference_width))
        self.location_hero_size = (int(reference_width), int(reference_height))
        self.direct_msr_sheet_prompt_builder = direct_msr_sheet_prompt_builder
        self.sequence_backend = sequence_backend
        self.sequence_planner = sequence_planner
        self.visual_style = str(visual_style or "").strip()
        self.on_sequence_phase = on_sequence_phase

    def generate_subject_bible(self, subject: ReferenceSubject) -> Path:
        if self.sequence_backend is not None:
            return self._generate_sequence_subject_bible(subject)
        final_dir = self.output_dir / "actors" / subject.id
        staging_dir = self.output_dir / ".staging" / f"actor-{subject.id}-{uuid.uuid4().hex}"
        try:
            manifest_path = self._generate_subject_bible(subject, staging_dir)
            return _commit_staged_reference(staging_dir, final_dir, manifest_path.name)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _generate_sequence_subject_bible(self, subject: ReferenceSubject) -> Path:
        result = SequenceReferencePipeline(
            anchor_backend=self.backend,
            sequence_backend=self.sequence_backend,
            planner=self.sequence_planner,
            on_phase=self.on_sequence_phase,
        ).generate(
            SequenceReferenceRequest(
                kind="character",
                asset_id=subject.id,
                name=subject.name,
                description=subject.visual_description or subject.name,
                image_prompt=subject.image_prompt,
                visual_style=self.visual_style,
                asset_context=asdict(subject),
                output_dir=self.output_dir,
                reference_image_size=self.actor_hero_size,
            )
        )
        manifest = {
            **asdict(subject),
            "kind": "actor",
            "views": [],
            "anchor_path": self._artifact_path(result.anchor_path),
            "sequence_path": self._artifact_path(result.sequence_path),
            "contact_sheet_path": self._artifact_path(result.contact_sheet_path),
            "msr_input_path": self._artifact_path(result.sheet_path),
            "sheet_path": self._artifact_path(result.sheet_path),
            "planning_profile": result.planning_profile,
            "prompt_revision": result.prompt_revision,
            "planner_source": result.planner_source,
            "fallback_reason": result.fallback_reason,
            "semantic_plan_hash": result.semantic_plan_hash,
            "prompt_hash": result.prompt_hash,
            "workflow_profile": result.workflow_profile,
            "seed": result.seed,
            "frames": result.frames,
            "anchor_prompt": result.anchor_prompt,
        }
        manifest_path = self.output_dir / "actors" / subject.id / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path

    def _generate_subject_bible(self, subject: ReferenceSubject, subject_dir: Path) -> Path:
        subject_dir.mkdir(parents=True, exist_ok=True)

        if self.actor_view_names == self.direct_msr_actor_view_names:
            return self._generate_direct_msr_subject_bible(subject, subject_dir)

        views = []
        hero_path = None
        for index, view_name in enumerate(self.actor_view_names, start=1):
            view_dir = subject_dir / "views"
            is_first_reference = index == 1
            backend = self.backend if is_first_reference else self.edit_backend
            anchors = self.hero_anchors if is_first_reference else self.edit_anchors
            width, height = self._actor_view_size(view_name) if is_first_reference else (None, None)
            rendered = backend.render_image(
                ImageRenderRequest(
                    scene={"reference_id": subject.id, "view": view_name},
                    scene_number=index,
                    prompt=(
                        self._view_prompt(subject, view_name)
                        if is_first_reference
                        else self._edit_view_prompt(subject, view_name)
                    ),
                    workflow_path=Path(""),
                    output_dir=view_dir,
                    width=width,
                    height=height,
                    reference_image=hero_path,
                    anchors=anchors,
                )
            )
            target = view_dir / f"{view_name}.png"
            if Path(rendered) != target:
                Path(rendered).replace(target)
            if is_first_reference:
                hero_path = target
            views.append({"name": view_name, "path": self._artifact_path(target)})
            self._report_view_complete(
                kind="actor",
                item_id=subject.id,
                item_name=subject.name,
                view=view_name,
                index=index,
                path=target,
            )

        view_paths = [subject_dir / "views" / f"{view_name}.png" for view_name in self.actor_view_names]
        if len(view_paths) == 1:
            sheet_path = msr_sheet_path = view_paths[0]
        else:
            sheet_path = subject_dir / "sheet.png"
            compose_reference_sheet(view_paths, sheet_path, labels=False)
            msr_sheet_path = subject_dir / "msr_sheet.png"
            compose_msr_reference_sheet(
                view_paths,
                msr_sheet_path,
                size=self.msr_sheet_size,
            )
        manifest = {
            **asdict(subject),
            "kind": "actor",
            "views": views,
            "msr_input_path": self._artifact_path(msr_sheet_path if len(views) > 1 else hero_path),
            "sheet_path": self._artifact_path(sheet_path),
        }
        manifest_path = subject_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def _generate_direct_msr_subject_bible(self, subject: ReferenceSubject, subject_dir: Path) -> Path:
        view_name = self.direct_msr_actor_view_names[0]
        view_dir = subject_dir / "views"
        rendered = self.backend.render_image(
            ImageRenderRequest(
                scene={"reference_id": subject.id, "view": view_name},
                scene_number=1,
                prompt=(
                    self.direct_msr_sheet_prompt_builder(subject)
                    if self.direct_msr_sheet_prompt_builder is not None
                    else self._direct_msr_sheet_prompt(subject)
                ),
                workflow_path=Path(""),
                output_dir=view_dir,
                width=self.location_hero_size[0],
                height=self.location_hero_size[1],
                reference_image=None,
                anchors=self.hero_anchors,
            )
        )
        target = view_dir / f"{view_name}.png"
        if Path(rendered) != target:
            Path(rendered).replace(target)
        self._report_view_complete(
            kind="actor",
            item_id=subject.id,
            item_name=subject.name,
            view=view_name,
            index=1,
            path=target,
        )
        manifest = {
            **asdict(subject),
            "kind": "actor",
            "views": [{"name": view_name, "path": self._artifact_path(target)}],
            "msr_input_path": self._artifact_path(target),
            "sheet_path": self._artifact_path(target),
        }
        manifest_path = subject_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def generate_location_bible(self, location: ReferenceLocation) -> Path:
        if self.sequence_backend is not None:
            return self._generate_sequence_location_bible(location)
        final_dir = self.output_dir / "locations" / location.id
        staging_dir = self.output_dir / ".staging" / f"location-{location.id}-{uuid.uuid4().hex}"
        try:
            manifest_path = self._generate_location_bible(location, staging_dir)
            return _commit_staged_reference(staging_dir, final_dir, manifest_path.name)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _generate_sequence_location_bible(self, location: ReferenceLocation) -> Path:
        result = SequenceReferencePipeline(
            anchor_backend=self.backend,
            sequence_backend=self.sequence_backend,
            planner=self.sequence_planner,
            on_phase=self.on_sequence_phase,
        ).generate(
            SequenceReferenceRequest(
                kind="location",
                asset_id=location.id,
                name=location.name,
                description=location.visual_description or location.name,
                image_prompt=location.image_prompt,
                visual_style=self.visual_style,
                asset_context=asdict(location),
                output_dir=self.output_dir,
                reference_image_size=self.location_hero_size,
            )
        )
        manifest = {
            **asdict(location),
            "kind": "location",
            "views": [],
            "anchor_path": self._artifact_path(result.anchor_path),
            "sequence_path": self._artifact_path(result.sequence_path),
            "contact_sheet_path": self._artifact_path(result.contact_sheet_path),
            "msr_background_path": self._artifact_path(result.anchor_path),
            "sheet_path": self._artifact_path(result.sheet_path),
            "planning_profile": result.planning_profile,
            "prompt_revision": result.prompt_revision,
            "planner_source": result.planner_source,
            "fallback_reason": result.fallback_reason,
            "semantic_plan_hash": result.semantic_plan_hash,
            "prompt_hash": result.prompt_hash,
            "workflow_profile": result.workflow_profile,
            "seed": result.seed,
            "frames": result.frames,
            "anchor_prompt": result.anchor_prompt,
        }
        manifest_path = self.output_dir / "locations" / location.id / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path

    def _generate_location_bible(self, location: ReferenceLocation, location_dir: Path) -> Path:
        location_dir.mkdir(parents=True, exist_ok=True)

        views = []
        hero_path = None
        for index, view_name in enumerate(self.location_view_names, start=1):
            view_dir = location_dir / "views"
            backend = self.backend if view_name == "hero" else self.edit_backend
            anchors = self.hero_anchors if view_name == "hero" else self.edit_anchors
            rendered = backend.render_image(
                ImageRenderRequest(
                    scene={"reference_id": location.id, "view": view_name},
                    scene_number=index,
                    prompt=self._location_view_prompt(location, view_name),
                    workflow_path=Path(""),
                    output_dir=view_dir,
                    width=self.location_hero_size[0] if view_name == "hero" else None,
                    height=self.location_hero_size[1] if view_name == "hero" else None,
                    reference_image=hero_path,
                    anchors=anchors,
                )
            )
            target = view_dir / f"{view_name}.png"
            if Path(rendered) != target:
                Path(rendered).replace(target)
            if view_name == "hero":
                hero_path = target
            views.append({"name": view_name, "path": self._artifact_path(target)})
            self._report_view_complete(
                kind="location",
                item_id=location.id,
                item_name=location.name,
                view=view_name,
                index=index,
                path=target,
            )

        view_paths = [location_dir / "views" / f"{view_name}.png" for view_name in self.location_view_names]
        if len(view_paths) == 1:
            sheet_path = view_paths[0]
        else:
            sheet_path = location_dir / "sheet.png"
            compose_reference_sheet(view_paths, sheet_path, labels=False)
        manifest = {
            **asdict(location),
            "kind": "location",
            "views": views,
            "msr_background_path": self._artifact_path(hero_path),
            "sheet_path": self._artifact_path(sheet_path),
        }
        manifest_path = location_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _view_prompt(subject: ReferenceSubject, view_name: str) -> str:
        base = subject.image_prompt or subject.visual_description or subject.name
        if view_name in {"closeup", "hero_closeup"}:
            return (
                f"{base}. Create a character reference closeup of {subject.name}: "
                "head and shoulders only, square portrait crop, same identity, same face, "
                "same hairstyle, same outfit details at the neckline, plain white seamless studio background, "
                "even reference-sheet lighting, no environment, no scenery, no props, no text, no extra characters."
            )

        view_direction = {
            "hero": "neutral three-quarter hero reference view",
            "front": "straight front view",
            "left": "clean left side profile view",
            "right": "clean right side profile view",
            "back": "clean full-body back view",
        }.get(view_name, f"{view_name} view")
        return (
            f"{base}. Create a full-body character reference of {subject.name}: "
            f"{view_direction}, portrait reference frame, head to toe visible, feet visible, "
            "centered standing pose, same identity, same face, same hairstyle, same body proportions, "
            "same outfit, same colors and materials, empty margin around the full body, plain white seamless studio background, "
            "even reference-sheet lighting, no environment, no scenery, no props, no text, no extra characters."
        )

    @staticmethod
    def _edit_view_prompt(subject: ReferenceSubject, view_name: str) -> str:
        view_direction = {
            "front": "straight front view",
            "left": "left-side view",
            "right": "right-side view",
            "back": "full-body back view",
            "closeup": "head-and-shoulders closeup view",
            "hero_closeup": "head-and-shoulders closeup view",
        }.get(view_name, f"{view_name} view")
        return (
            f"Create a {view_direction} of the character from the reference image. "
            "Keep the same identity, face, hairstyle, body proportions, outfit, colors, and materials from the reference image. "
            "Use a plain white seamless studio background, even reference-sheet lighting, no environment, no scenery, no props, no text, no extra characters."
        )

    @staticmethod
    def _direct_msr_sheet_prompt(subject: ReferenceSubject) -> str:
        base = subject.image_prompt or subject.visual_description or subject.name
        return (
            "vertical four panel character sheet photos.\n\n"
            f"{base}\n\n"
            "1st panel is a closeup,\n"
            "2nd panel is front view,\n"
            "3rd panel is left view,\n"
            "4th panel is back view.\n\n"
            "the panel background is white"
        )

    def _actor_view_size(self, view_name: str) -> tuple[int, int]:
        return self.actor_hero_size

    @staticmethod
    def _location_view_prompt(location: ReferenceLocation, view_name: str) -> str:
        base = location.visual_description or location.image_prompt or location.name
        return (
            f"{base}. Wide {view_name} view of {location.name}, single continuous image, "
            "no collage, no split screen, no panels."
        )

    def _artifact_path(self, path: str | Path | None) -> str:
        if path is None:
            return ""
        path = Path(path)
        try:
            return path.relative_to(self.artifact_base_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _report_view_complete(
        self,
        *,
        kind: str,
        item_id: str,
        item_name: str,
        view: str,
        index: int,
        path: Path,
    ) -> None:
        if self.on_view_complete is None:
            return
        self.on_view_complete(
            {
                "kind": kind,
                "id": item_id,
                "name": item_name,
                "view": view,
                "item_completed": index,
                "item_total": len(self.actor_view_names if kind == "actor" else self.location_view_names),
                "path": path,
            }
        )


def compose_msr_reference_sheet(image_paths: list[Path], output_path: Path, *, size: tuple[int, int]) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("Cannot compose an empty MSR reference sheet")

    width, height = int(size[0]), int(size[1])
    sheet = Image.new("RGB", (width, height), "white")
    columns = len(images)
    cell_width = max(1, width // columns)
    for index, image in enumerate(images):
        x0 = index * cell_width
        x1 = width if index == columns - 1 else (index + 1) * cell_width
        fitted = _fit_image_cover(image, (x1 - x0, height))
        sheet.paste(fitted, (x0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _fit_image_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(resized_size)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _fit_contain_image(image: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    fitted = ImageOps.contain(image, size, Image.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def ingredients_sheet_size(
    width: int,
    height: int,
    scale: float = 2.0,
) -> tuple[int, int]:
    """Return a 12:7 canvas that is at least ``scale`` times the source size."""
    minimum_width = max(1, math.ceil(width * scale))
    minimum_height = max(1, math.ceil(height * scale))
    units = math.ceil(max(minimum_width / 12, minimum_height / 7))
    return units * 12, units * 7


def compose_reference_sheet(image_paths: list[Path], output_path: Path, *, labels: bool = True) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("Cannot compose an empty reference sheet")

    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 24 if labels else 0
    columns = reference_sheet_columns(width=width, height=height, image_count=len(images))
    rows = math.ceil(len(images) / columns)
    cell_height = height + label_height
    sheet = Image.new("RGB", (columns * width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, image in enumerate(images):
        column = index % columns
        row = index // columns
        x = column * width
        y = row * cell_height
        sheet.paste(image.resize((width, height)), (x, y))
        if labels:
            draw.text((x + 4, y + height + 4), image_paths[index].stem, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def reference_sheet_columns(*, width: int, height: int, image_count: int) -> int:
    if image_count <= 1:
        return 1
    if width > height:
        return min(3, image_count)
    return image_count


def compose_scene_reference_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    size: tuple[int, int] = (1280, 704),
    columns: int | None = None,
) -> Path:
    if not image_paths:
        raise ValueError("Cannot compose a scene reference sheet from an empty list")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    return _compose_scene_reference_images(images, output_path, size=size)


def _compose_scene_reference_snapshots(
    snapshots: list[IngredientsSourceSnapshot],
    output_path: Path,
    *,
    size: tuple[int, int],
) -> Path:
    images = []
    for snapshot in snapshots:
        with Image.open(BytesIO(snapshot.content)) as source:
            images.append(source.convert("RGB"))
    return _compose_scene_reference_images(images, output_path, size=size)


def _compose_scene_reference_images(
    images: list[Image.Image],
    output_path: Path,
    *,
    size: tuple[int, int],
) -> Path:
    width, height = int(size[0]), int(size[1])
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    gap = 16
    cell_w = (width - gap * (cols + 1)) // cols
    cell_h = (height - gap * (rows + 1)) // rows
    sheet = Image.new("RGB", (width, height), (0, 0, 0))

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        fitted = _fit_contain_image(image, (cell_w, cell_h))
        sheet.paste(fitted, (gap + col * (cell_w + gap), gap + row * (cell_h + gap)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def snapshot_ingredients_sources(
    images: list[dict],
    *,
    project_base: Path,
) -> list[IngredientsSourceSnapshot]:
    root = Path(project_base).resolve()
    snapshots = []
    total_bytes = 0
    for image in images:
        raw = Path(str(image["path"]))
        path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Ingredients signature source must be a project file")
        size = path.stat().st_size
        if size > _MAX_INGREDIENTS_SOURCE_BYTES:
            raise ValueError(f"Ingredients source is too large to snapshot: {path}")
        total_bytes += size
        if total_bytes > _MAX_INGREDIENTS_SOURCE_BYTES:
            raise ValueError("Ingredients sources are too large to snapshot")
        content = path.read_bytes()
        snapshots.append(IngredientsSourceSnapshot(
            id=str(image.get("id") or "").strip(),
            type=str(image.get("type") or "").strip(),
            path=path.relative_to(root).as_posix(),
            suffix=path.suffix.lower(),
            content=content,
        ))
    return snapshots


def ingredients_signature_references(
    images: list[dict],
    *,
    project_base: Path,
    snapshots: list[IngredientsSourceSnapshot] | None = None,
) -> list[dict[str, str]]:
    snapshots = snapshots or snapshot_ingredients_sources(
        images,
        project_base=project_base,
    )
    return [
        {
            "id": snapshot.id,
            "type": snapshot.type,
            "sha256": hashlib.sha256(snapshot.content).hexdigest(),
        }
        for snapshot in snapshots
    ]


def ingredients_signature_sources(
    images: list[dict],
    *,
    project_base: Path,
    snapshots: list[IngredientsSourceSnapshot] | None = None,
) -> list[dict[str, str]]:
    snapshots = snapshots or snapshot_ingredients_sources(
        images,
        project_base=project_base,
    )
    return [
        {"id": snapshot.id, "type": snapshot.type, "path": snapshot.path}
        for snapshot in snapshots
    ]


def visual_consistency_sources(
    images: list[dict],
    *,
    project_base: Path,
) -> dict:
    root = Path(project_base).resolve()
    actors: list[dict[str, str]] = []
    location: dict[str, str] | None = None
    for image in images:
        raw = Path(str(image.get("contract_path") or image["path"]))
        path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Visual consistency source must be a project file")
        source = {
            "id": str(image.get("id") or "").strip(),
            "path": path.relative_to(root).as_posix(),
        }
        if image.get("type") == "actor":
            actors.append(source)
        elif image.get("type") == "location":
            location = source
    return {"actors": actors, "location": location}


def ingredients_sheet_signature(
    references: list[dict[str, str]],
    *,
    size: tuple[int, int],
    layout_version: str = INGREDIENTS_SHEET_LAYOUT_VERSION,
) -> str:
    return _ingredients_sheet_signature(
        references,
        size=size,
        layout_version=layout_version,
    )


def compose_cached_ingredients_sheet(
    image_paths: list[Path],
    *,
    cache_dir: Path,
    references: list[dict[str, str]],
    size: tuple[int, int],
    layout_version: str = INGREDIENTS_SHEET_LAYOUT_VERSION,
    snapshots: list[IngredientsSourceSnapshot] | None = None,
) -> tuple[Path, str]:
    if snapshots is not None:
        snapshot_references = [
            {
                "id": snapshot.id,
                "type": snapshot.type,
                "sha256": hashlib.sha256(snapshot.content).hexdigest(),
            }
            for snapshot in snapshots
        ]
        if snapshot_references != references:
            raise ValueError(
                "Ingredients source snapshots do not match signature references"
            )
    signature = ingredients_sheet_signature(
        references,
        size=size,
        layout_version=layout_version,
    )
    output_path = Path(cache_dir) / f"{signature}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{signature}.lock"
    lock_fd = _acquire_cache_lock(lock_path, output_path)
    try:
        if _valid_cached_ingredients_sheet(output_path, size=size):
            return output_path, signature
        with NamedTemporaryFile(
            suffix=".png",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            if snapshots is None:
                compose_scene_reference_sheet(image_paths, temporary, size=size)
            else:
                _compose_scene_reference_snapshots(
                    snapshots,
                    temporary,
                    size=size,
                )
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        _release_cache_lock(lock_fd)
        os.close(lock_fd)
    return output_path, signature


def _valid_cached_ingredients_sheet(
    output_path: Path,
    *,
    size: tuple[int, int],
) -> bool:
    if not output_path.is_file():
        return False
    try:
        with Image.open(output_path) as image:
            if image.format != "PNG" or image.size != size:
                return False
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def _acquire_cache_lock(lock_path: Path, output_path: Path) -> int:
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    if os.fstat(lock_fd).st_size == 0:
        os.write(lock_fd, b"\0")
    deadline = time.monotonic() + _INGREDIENTS_CACHE_LOCK_TIMEOUT_SECONDS
    while True:
        if _try_cache_lock(lock_fd):
            return lock_fd
        if time.monotonic() >= deadline:
            os.close(lock_fd)
            raise TimeoutError(
                f"Timed out waiting for Ingredients cache entry: {output_path}"
            )
        time.sleep(0.01)


def _try_cache_lock(lock_fd: int) -> bool:
    os.lseek(lock_fd, 0, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _release_cache_lock(lock_fd: int) -> None:
    os.lseek(lock_fd, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_fd, fcntl.LOCK_UN)


def build_runtime_consistency_contract(
    scene: dict,
    *,
    images: list[dict],
    project_base: Path,
    mode: str,
    workflow_profile: str,
) -> SceneConsistencyContract:
    actor_anchors = tuple(
        _runtime_reference_anchor(image, project_base=project_base)
        for image in images
        if image.get("type") == "actor"
    )
    location_anchor = next(
        (
            _runtime_reference_anchor(image, project_base=project_base)
            for image in images
            if image.get("type") == "location"
        ),
        None,
    )
    return SceneConsistencyContract.create(
        scene=int(scene["scene"]),
        mode=mode,
        workflow_profile=workflow_profile,
        actors=actor_anchors,
        location=location_anchor,
        transition_from_previous=transition_from_previous(
            scene.get("transition_from_previous")
        ),
    )


def _runtime_reference_anchor(
    image: dict,
    *,
    project_base: Path,
) -> ReferenceAnchor:
    kind = str(image.get("type") or "").strip()
    semantic_id = str(image.get("id") or "").strip()
    description = " ".join(
        str(
            image.get("visual_description")
            or image.get("image_prompt")
            or image.get("name")
            or semantic_id
        ).split()
    )
    return ReferenceAnchor(
        id=semantic_id,
        kind=kind,
        look_id=str(image.get("look_id") or "default").strip() or "default",
        asset_role=(
            "identity-reference"
            if kind == "actor"
            else "environment-reference"
        ),
        asset_sha256=sha256_file(
            project_base / str(image.get("contract_path") or image["path"])
        ),
        prompt_anchor=(
            f"Reference {kind} `{semantic_id}` (look "
            f"`{str(image.get('look_id') or 'default').strip() or 'default'}`): "
            f"{description}"
        )[:350],
    )


def _panel_position_label(row: int, col: int, num_rows: int, num_cols: int, index: int, total: int) -> str:
    """Return a human-readable grid position label."""
    if total == 1:
        return "Full"

    row_names = ("Top", "Middle", "Bottom")
    col_labels_2 = ("Left", "Right")
    col_labels_3 = ("Left", "Center", "Right")
    col_labels_4 = ("Left", "Center-Left", "Center-Right", "Right")

    is_last_row = row == num_rows - 1
    remaining = total - index - 1

    if total > 1 and is_last_row and remaining == 0 and col == 0:
        if num_rows == 2:
            row_name = "Bottom"
        elif num_rows <= 3:
            row_name = row_names[row]
        else:
            row_name = f"Row {row + 1}"
        return f"{row_name} Row"

    if num_rows == 1:
        row_name = ""
    elif num_rows == 2:
        row_name = "Top" if row == 0 else "Bottom"
    else:
        if row == 0:
            row_name = "Top"
        elif row == num_rows - 1:
            row_name = "Bottom"
        else:
            row_name = "Middle" if num_rows == 3 else f"Row {row + 1}"

    if num_cols == 1:
        return row_name or "Full"
    elif num_cols == 2:
        col_label = col_labels_2[col]
    elif num_cols == 3:
        col_label = col_labels_3[col]
    else:
        col_label = col_labels_4[min(col, 3)]

    if not row_name:
        return col_label
    if row_name.startswith("Row "):
        return f"{row_name} {col_label}"
    return f"{row_name} Row {col_label}"


def _type_label(item_type: str) -> str:
    labels = {
        "actor": "Character",
        "location": "Setting",
        "prop": "Prop",
    }
    return labels.get(item_type, item_type.title())


def generate_scene_sheet_description(images: list[dict], num_cols: int, size: tuple[int, int]) -> str:
    """Generate a structured description of the scene reference sheet layout.

    Each image dict should contain 'type' and 'visual_description'.
    """
    if not images:
        return ""

    num_rows = math.ceil(len(images) / num_cols)
    lines = ["### Reference Sheet Description"]

    for index, img in enumerate(images):
        row = index // num_cols
        col = index % num_cols
        position = _panel_position_label(row, col, num_rows, num_cols, index, len(images))
        type_label = _type_label(img.get("type", "actor"))
        anchor = str(img.get("id") or "").strip()
        description = str(img.get("visual_description") or img.get("name") or "").strip()
        if description:
            label = f"{type_label}, {anchor}" if anchor else type_label
            lines.append(f"**{position} ({label}):** {description}")

    return "\n".join(lines)


def generate_scene_sheet_anchors(images: list[dict], num_cols: int) -> list[dict[str, str]]:
    if not images:
        return []
    num_rows = math.ceil(len(images) / num_cols)
    return [
        {
            "id": str(image.get("id") or "").strip(),
            "type": str(image.get("type") or "").strip(),
            "position": _panel_position_label(
                index // num_cols,
                index % num_cols,
                num_rows,
                num_cols,
                index,
                len(images),
            ),
        }
        for index, image in enumerate(images)
    ]


def build_ingredients_target_binding(anchors: list[dict]) -> str:
    lines = []
    for anchor in anchors:
        item_id = str(anchor.get("id") or "").strip()
        position = str(anchor.get("position") or "").strip()
        item_type = str(anchor.get("type") or "").strip()
        if item_type == "actor":
            lines.append(f"Use Character `{item_id}` from {position} as a visible character.")
        elif item_type == "location":
            lines.append(f"Use Setting `{item_id}` from {position} as the environment.")
    if any(str(anchor.get("type")) == "actor" for anchor in anchors):
        lines.append("Do not add or omit visible characters.")
    return "\n".join(lines) + ("\n" if lines else "")


def _infer_reference_artifact_base_dir(output_dir: Path) -> Path:
    if output_dir.name == "references" and output_dir.parent.name == "output":
        return output_dir.parent.parent
    return output_dir


def _commit_staged_reference(staging_dir: Path, final_dir: Path, manifest_name: str) -> Path:
    """Atomically publish a completed reference directory."""
    manifest_path = staging_dir / manifest_name
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = staging_dir.parent.parent
        staging_token = staging_dir.relative_to(root).as_posix()
        final_token = final_dir.relative_to(root).as_posix()

        def rewrite(value):
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, str):
                return value.replace(staging_token, final_token)
            return value

        manifest_path.write_text(
            json.dumps(rewrite(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = final_dir.with_name(f".{final_dir.name}.previous")
    shutil.rmtree(backup_dir, ignore_errors=True)
    if final_dir.exists():
        final_dir.replace(backup_dir)
    try:
        staging_dir.replace(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.replace(final_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)
    return final_dir / manifest_name


def enrich_render_plan_with_reference_sheets(
    render_plan_path: str | Path,
    references_dir: str | Path,
    output_path: str | Path,
    on_scene_complete: Callable[[int, int, int], None] | None = None,
) -> Path:
    render_plan_path = Path(render_plan_path)
    references_dir = Path(references_dir)
    output_path = Path(output_path)
    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8-sig"))
    actor_manifests = _load_manifests_by_id(references_dir / "actors")
    location_manifests = _load_manifests_by_id(references_dir / "locations")
    project_base = _infer_reference_artifact_base_dir(references_dir)

    required_actor_ids = {
        str(actor_id)
        for scene in render_plan
        for actor_id in scene.get("references", {}).get("actor_ids", [])
    }
    required_location_ids = {
        str(location_id)
        for scene in render_plan
        if (location_id := scene.get("references", {}).get("location_id"))
    }
    missing_actor_ids = sorted(required_actor_ids - actor_manifests.keys())
    missing_location_ids = sorted(required_location_ids - location_manifests.keys())
    if missing_actor_ids or missing_location_ids:
        details = []
        if missing_actor_ids:
            details.append(f"actors: {', '.join(missing_actor_ids)}")
        if missing_location_ids:
            details.append(f"locations: {', '.join(missing_location_ids)}")
        raise FeverSlopValidationError(
            "Missing reference manifests for render plan ("
            + "; ".join(details)
            + "). Run --stage msr_references before --stage msr_reference_sheets."
        )

    total = len(render_plan)
    for index, scene in enumerate(render_plan, start=1):
        references = scene.setdefault("references", {})
        actor_ids = list(references.get("actor_ids") or [])
        if len(actor_ids) > 4:
            raise ValueError(f"Scene {scene.get('scene')} references at most 4 actors for ltx_msr")
        references["actor_sheet_paths"] = [
            _portable_manifest_path(actor_manifests[actor_id], "sheet_path", references_dir)
            for actor_id in actor_ids
        ]
        references["actor_msr_paths"] = [
            _portable_manifest_path(actor_manifests[actor_id], "msr_input_path", references_dir, fallback_key="sheet_path")
            for actor_id in actor_ids
        ]
        references["actor_reference_descriptions"] = [
            _reference_description(actor_manifests[actor_id])
            for actor_id in actor_ids
        ]
        location_id = references.get("location_id")
        if location_id:
            location_manifest = location_manifests[str(location_id)]
            references["location_sheet_path"] = _portable_manifest_path(location_manifest, "sheet_path", references_dir)
            references["location_msr_path"] = _portable_manifest_path(
                location_manifest,
                "msr_background_path",
                references_dir,
                fallback_key="sheet_path",
            )
            references["location_reference_description"] = _reference_description(location_manifests[str(location_id)])
        consistency_images = [
            {"id": actor_id, "type": "actor", "path": path}
            for actor_id, path in zip(actor_ids, references["actor_msr_paths"])
        ]
        if location_id:
            consistency_images.append({
                "id": str(location_id),
                "type": "location",
                "path": references["location_msr_path"],
            })
        scene["visual_consistency_sources"] = visual_consistency_sources(
            consistency_images,
            project_base=project_base,
        )
        if on_scene_complete is not None:
            on_scene_complete(int(scene.get("scene", index)), index, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(render_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _load_manifests_by_id(root: Path) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    if not root.exists():
        return manifests
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifests[str(manifest["id"])] = manifest
    return manifests


def _portable_manifest_path(
    manifest: dict,
    key: str,
    references_dir: Path,
    *,
    fallback_key: str | None = None,
) -> str:
    value = str(manifest.get(key) or manifest.get(fallback_key or "") or "").strip()
    if not value:
        return ""
    normalized = value.replace("\\", "/")
    path = Path(value)
    project_base = _infer_reference_artifact_base_dir(references_dir)
    if path.is_absolute():
        try:
            return path.relative_to(project_base).as_posix()
        except ValueError:
            return path.as_posix()
    return normalized


def _reference_description(manifest: dict) -> dict:
    return {
        key: str(manifest.get(key, "") or "").strip()
        for key in ("id", "name", "role", "visual_description", "image_prompt")
        if str(manifest.get(key, "") or "").strip()
    }
