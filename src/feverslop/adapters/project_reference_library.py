from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from feverslop.config.project_config import ProjectConfig
from feverslop.domain.reference_workspace import (
    PropInteraction,
    ReferenceAsset,
    ReferenceKind,
    ReferenceProvenance,
    ReferenceWorkspaceSnapshot,
    SceneReferenceAssignment,
)
from feverslop.ports.reference_library import (
    ArtifactInvalidationPort,
    ImportReferencePort,
    MovieBiblePort,
    ReferenceLibraryPort,
    SceneCastPort,
)


@contextmanager
def _assignment_file_lock(lock_path: Path):
    """Serialize revision check and replace across threads/processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ProjectReferenceLibrary(
    ReferenceLibraryPort,
    ImportReferencePort,
    MovieBiblePort,
    SceneCastPort,
    ArtifactInvalidationPort,
):
    """Adapts project files to the reference library protocol.

    Reads reference data from the project's movie bible, render plan, and
    generated artifacts.  Saves assignments into
    ``movie/reference_assignments.json`` with atomic revision control.
    """

    def __init__(self, project_root: Path, max_scene_actors: int = 4):
        self._project_root = project_root
        self._max_scene_actors = max_scene_actors

    # -- ReferenceLibraryPort --

    def load(self, project_id: str) -> ReferenceWorkspaceSnapshot:
        root = self._project_root
        assignments_path = root / "movie" / "reference_assignments.json"

        assignments: tuple[SceneReferenceAssignment, ...] = ()
        revision = "r1"
        if assignments_path.exists():
            data = json.loads(assignments_path.read_text(encoding="utf-8"))
            revision = str(data.get("revision", "r1"))
            assignments = tuple(
                SceneReferenceAssignment(
                    scene_number=a["scene_number"],
                    actor_ids=tuple(a.get("actor_ids") or ()),
                    location_ids=tuple(a.get("location_ids") or ()),
                    background_ids=tuple(a.get("background_ids") or ()),
                    style_ids=tuple(a.get("style_ids") or ()),
                    actor_look_ids=dict(a.get("actor_look_ids") or {}),
                    prop_ids=tuple(a.get("prop_ids") or ()),
                    prop_interactions=tuple(PropInteraction.from_dict(item) for item in a.get("prop_interactions") or []),
                )
                for a in data.get("assignments") or []
            )

        assets = self._discover_assets(root)
        return ReferenceWorkspaceSnapshot(
            assets=assets,
            assignments=assignments,
            revision=revision,
            project_id=project_id,
        )

    def save_assignments(
        self,
        project_id: str,
        assignments: tuple[SceneReferenceAssignment, ...],
        expected_revision: str,
    ) -> str:
        root = self._project_root
        assignments_path = root / "movie" / "reference_assignments.json"

        with _assignment_file_lock(assignments_path.with_suffix(".lock")):
            current_revision = "r1"
            if assignments_path.exists():
                data = json.loads(assignments_path.read_text(encoding="utf-8"))
                current_revision = str(data.get("revision", "r1"))

            if current_revision != expected_revision:
                raise ValueError(
                    f"Revision mismatch: expected {expected_revision!r}, got {current_revision!r}"
                )

            new_revision = _next_revision(current_revision)
            payload = {
                "revision": new_revision,
                "assignments": [
                    {
                        "scene_number": a.scene_number,
                        "actor_ids": list(a.actor_ids),
                        "location_ids": list(a.location_ids),
                        "background_ids": list(a.background_ids),
                        "style_ids": list(a.style_ids),
                        "actor_look_ids": dict(a.actor_look_ids or {}),
                        "prop_ids": list(a.prop_ids),
                        "prop_interactions": [item.to_dict() for item in a.prop_interactions],
                    }
                    for a in assignments
                ],
            }
            assignments_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=assignments_path.parent, delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temporary_path = Path(handle.name)
            os.replace(temporary_path, assignments_path)
        return new_revision

    def add_asset(self, project_id: str, asset: ReferenceAsset) -> ReferenceAsset:
        return asset

    # -- ImportReferencePort --

    def import_asset(
        self,
        project_id: str,
        source_path: Path,
        asset: ReferenceAsset,
    ) -> ReferenceAsset:
        source_path = Path(source_path)
        if ".." in source_path.parts:
            raise ValueError(f"Directory traversal detected in source path: {source_path}")

        dest_dir = self._project_root / "movie" / "references" / "imported"
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _safe_filename(asset.id, source_path.suffix)
        dest_path = dest_dir / safe_name

        if dest_path.exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            idx = 2
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{idx}{suffix}"
                idx += 1

        shutil.copy2(source_path, dest_path)

        width, height = _extract_dimensions(dest_path)
        relative_path = dest_path.relative_to(self._project_root).as_posix()

        return ReferenceAsset(
            id=asset.id,
            kind=asset.kind,
            label=asset.label or safe_name,
            path=relative_path,
            width=width,
            height=height,
            provenance=ReferenceProvenance(source="import"),
            looks=asset.looks,
            stale=False,
        )

    # -- MovieBiblePort --

    def get_known_actor_ids(self, project_id: str) -> list[str]:
        manifest = self._read_manifest()
        actors = manifest.get("actors") or []
        if isinstance(actors, dict):
            actors = actors.values()
        return [str(a.get("id", "")) for a in actors if a.get("id")]

    def get_known_location_ids(self, project_id: str) -> list[str]:
        manifest = self._read_manifest()
        locations = manifest.get("locations") or []
        if isinstance(locations, dict):
            locations = locations.values()
        return [str(item.get("id", "")) for item in locations if item.get("id")]

    def get_background_ids(self, project_id: str) -> list[str]:
        return []

    def get_known_prop_ids(self, project_id: str) -> list[str]:
        config_path = self._project_root / "config.json"
        if not config_path.is_file():
            return []
        try:
            config = ProjectConfig.load(config_path)
        except (FileNotFoundError, ValueError, KeyError):
            return []
        return [prop.asset_id for prop in config.global_props]

    # -- SceneCastPort --

    def get_max_scene_actors(self, project_id: str) -> int:
        return self._max_scene_actors

    # -- ArtifactInvalidationPort --

    def get_invalidated_artifacts(
        self,
        project_id: str,
        changed_scenes: list[int],
        changed_actor_ids: list[str] | None = None,
        changed_location_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for sn in changed_scenes:
            scene_num = str(sn)
            patterns = [
                f"movie/references/*scene{scene_num}*msr*",
                f"movie/references/*scene{scene_num}*ingredients*",
                f"movie/references/*scene{scene_num}*scene_msr*",
            ]
            msr_files = set()
            for pattern in patterns:
                parts = pattern.split("*")
                prefix = parts[0]
                suffix = parts[-1] if len(parts) > 1 else ""
                ref_dir = self._project_root / "movie" / "references"
                if ref_dir.exists():
                    for f in ref_dir.rglob("*"):
                        pos = f.relative_to(self._project_root).as_posix()
                        if pos.startswith(prefix) and pos.lower().endswith(suffix.lower()):
                            msr_files.add(pos)
            if msr_files:
                result["msr_sheets"] = sorted(msr_files)

            result.setdefault("ingredients_sheets", [])
            result.setdefault("renders", [])
        return result

    # -- Internal helpers --

    def _read_manifest(self) -> dict:
        manifest_path = self._project_root / "movie" / "references" / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {}

    def _discover_assets(self, root: Path) -> tuple[ReferenceAsset, ...]:
        ref_dir = root / "movie" / "references"
        if not ref_dir.exists():
            return ()

        assets: list[ReferenceAsset] = []
        for path in sorted(ref_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue

            name_lower = path.name.lower()
            kind = self._guess_kind(name_lower)
            if kind is None:
                continue

            width, height = _extract_dimensions(path)
            relative = path.relative_to(root).as_posix()
            label = path.stem.replace("_", " ").replace("-", " ").title()

            assets.append(ReferenceAsset(
                id=path.stem,
                kind=kind,
                label=label,
                path=relative,
                width=width,
                height=height,
                exists=True,
            ))

        return tuple(assets)

    @staticmethod
    def _guess_kind(name: str) -> ReferenceKind | None:
        if any(t in name for t in ("msr", "msr_sheet")):
            return ReferenceKind.MSR_SHEET
        if any(t in name for t in ("ingredients", "ingredient")):
            return ReferenceKind.INGREDIENTS_SHEET
        if any(t in name for t in ("storyboard", "sb_frame", "board")):
            return ReferenceKind.STORYBOARD_FRAME
        if any(t in name for t in ("continuity", "cont")):
            return ReferenceKind.CONTINUITY
        return None


def _next_revision(current: str) -> str:
    match = re.match(r"^r(\d+)$", current)
    if match:
        num = int(match.group(1))
        return f"r{num + 1}"
    return "r1"


def _safe_filename(id: str, suffix: str) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in id)
    clean = clean[:100] or "imported"
    ext = suffix if suffix.startswith(".") else ".png"
    return f"{clean}{ext}"


def _extract_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        pass
    return (0, 0)
