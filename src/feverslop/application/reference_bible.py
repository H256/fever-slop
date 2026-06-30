from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Callable, Any

from PIL import Image, ImageDraw

from feverslop.ports.rendering import ImageRenderBackend, ImageRenderRequest, WorkflowAnchorConfig


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

    def generate_subject_bible(self, subject: ReferenceSubject) -> Path:
        subject_dir = self.output_dir / "actors" / subject.id
        subject_dir.mkdir(parents=True, exist_ok=True)

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
                target.write_bytes(Path(rendered).read_bytes())
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

        sheet_path = subject_dir / "sheet.png"
        view_paths = [subject_dir / "views" / f"{view_name}.png" for view_name in self.actor_view_names]
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

    def generate_location_bible(self, location: ReferenceLocation) -> Path:
        location_dir = self.output_dir / "locations" / location.id
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
                target.write_bytes(Path(rendered).read_bytes())
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

        sheet_path = location_dir / "sheet.png"
        view_paths = [location_dir / "views" / f"{view_name}.png" for view_name in self.location_view_names]
        compose_reference_sheet(view_paths, sheet_path)
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

    @classmethod
    def _actor_view_size(cls, view_name: str) -> tuple[int, int]:
        return cls.actor_hero_size

    @staticmethod
    def _location_view_prompt(location: ReferenceLocation, view_name: str) -> str:
        base = location.image_prompt or location.visual_description or location.name
        return f"{base}. Environment reference {view_name} view of {location.name}."

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


def _infer_reference_artifact_base_dir(output_dir: Path) -> Path:
    if output_dir.name == "references" and output_dir.parent.name == "output":
        return output_dir.parent.parent
    return output_dir


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
