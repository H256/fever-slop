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

    def __init__(
        self,
        *,
        backend: ImageRenderBackend,
        output_dir: str | Path,
        edit_backend: ImageRenderBackend | None = None,
        hero_anchors: WorkflowAnchorConfig = WorkflowAnchorConfig(),
        edit_anchors: WorkflowAnchorConfig = WorkflowAnchorConfig(),
        on_view_complete: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.backend = backend
        self.edit_backend = edit_backend or backend
        self.hero_anchors = hero_anchors
        self.edit_anchors = edit_anchors
        self.on_view_complete = on_view_complete
        self.output_dir = Path(output_dir)

    def generate_subject_bible(self, subject: ReferenceSubject) -> Path:
        subject_dir = self.output_dir / "actors" / subject.id
        subject_dir.mkdir(parents=True, exist_ok=True)

        views = []
        hero_path = None
        for index, view_name in enumerate(self.view_names, start=1):
            view_dir = subject_dir / "views"
            backend = self.backend if view_name == "hero" else self.edit_backend
            anchors = self.hero_anchors if view_name == "hero" else self.edit_anchors
            rendered = backend.render_image(
                ImageRenderRequest(
                    scene={"reference_id": subject.id, "view": view_name},
                    scene_number=index,
                    prompt=self._view_prompt(subject, view_name),
                    workflow_path=Path(""),
                    output_dir=view_dir,
                    reference_image=hero_path,
                    anchors=anchors,
                )
            )
            target = view_dir / f"{view_name}.png"
            if Path(rendered) != target:
                target.write_bytes(Path(rendered).read_bytes())
            if view_name == "hero":
                hero_path = target
            views.append({"name": view_name, "path": str(target)})
            self._report_view_complete(
                kind="actor",
                item_id=subject.id,
                item_name=subject.name,
                view=view_name,
                index=index,
                path=target,
            )

        sheet_path = subject_dir / "sheet.png"
        compose_reference_sheet([Path(view["path"]) for view in views], sheet_path)
        manifest = {
            **asdict(subject),
            "kind": "actor",
            "views": views,
            "sheet_path": str(sheet_path),
        }
        manifest_path = subject_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def generate_location_bible(self, location: ReferenceLocation) -> Path:
        location_dir = self.output_dir / "locations" / location.id
        location_dir.mkdir(parents=True, exist_ok=True)

        views = []
        hero_path = None
        for index, view_name in enumerate(self.view_names, start=1):
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
                    reference_image=hero_path,
                    anchors=anchors,
                )
            )
            target = view_dir / f"{view_name}.png"
            if Path(rendered) != target:
                target.write_bytes(Path(rendered).read_bytes())
            if view_name == "hero":
                hero_path = target
            views.append({"name": view_name, "path": str(target)})
            self._report_view_complete(
                kind="location",
                item_id=location.id,
                item_name=location.name,
                view=view_name,
                index=index,
                path=target,
            )

        sheet_path = location_dir / "sheet.png"
        compose_reference_sheet([Path(view["path"]) for view in views], sheet_path)
        manifest = {
            **asdict(location),
            "kind": "location",
            "views": views,
            "sheet_path": str(sheet_path),
        }
        manifest_path = location_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _view_prompt(subject: ReferenceSubject, view_name: str) -> str:
        base = subject.image_prompt or subject.visual_description or subject.name
        return f"{base}. Character reference {view_name} view of {subject.name}."

    @staticmethod
    def _location_view_prompt(location: ReferenceLocation, view_name: str) -> str:
        base = location.image_prompt or location.visual_description or location.name
        return f"{base}. Environment reference {view_name} view of {location.name}."

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
                "item_total": len(self.view_names),
                "path": path,
            }
        )


def compose_reference_sheet(image_paths: list[Path], output_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("Cannot compose an empty reference sheet")

    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 24
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


def enrich_render_plan_with_reference_sheets(
    render_plan_path: str | Path,
    references_dir: str | Path,
    output_path: str | Path,
) -> Path:
    render_plan_path = Path(render_plan_path)
    references_dir = Path(references_dir)
    output_path = Path(output_path)
    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8-sig"))
    actor_manifests = _load_manifests_by_id(references_dir / "actors")
    location_manifests = _load_manifests_by_id(references_dir / "locations")

    for scene in render_plan:
        references = scene.setdefault("references", {})
        actor_ids = list(references.get("actor_ids") or [])
        if len(actor_ids) > 4:
            raise ValueError(f"Scene {scene.get('scene')} references at most 4 actors for ltx_msr")
        references["actor_sheet_paths"] = [
            actor_manifests[actor_id]["sheet_path"]
            for actor_id in actor_ids
        ]
        location_id = references.get("location_id")
        if location_id:
            references["location_sheet_path"] = location_manifests[str(location_id)]["sheet_path"]

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
