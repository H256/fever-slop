"""Safe filesystem adapter for the canonical global asset library."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset
from feverslop.utils.io import atomic_write_json


class GlobalLibraryAdapter:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or Path.home() / ".feverslop" / "library").expanduser().resolve()

    def _kind(self, kind: AssetKind | str) -> AssetKind:
        try:
            return kind if isinstance(kind, AssetKind) else AssetKind(kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid asset kind: {kind!r}") from exc

    def _directory(self, kind: AssetKind | str, asset_id: str) -> Path:
        resolved_kind = self._kind(kind)
        if not asset_id or Path(asset_id).name != asset_id or asset_id in {".", ".."}:
            raise ValueError("asset id must be a single safe path component")
        return self.root / resolved_kind.value / asset_id

    def _manifest_path(self, kind: AssetKind | str, asset_id: str) -> Path:
        return self._directory(kind, asset_id) / "manifest.json"

    @contextmanager
    def _lock(self, directory: Path, *, shared: bool = False) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".lock"
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                # msvcrt has no shared lock mode; shared degrades to exclusive (documented no-op)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_manifest(path: Path, asset: GlobalAsset) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="manifest-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asset.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, kind: AssetKind | str, asset_id: str) -> GlobalAsset:
        if not self._manifest_path(kind, asset_id).is_file():
            raise FileNotFoundError(f"global asset not found: {self._kind(kind).value}/{asset_id}")
        with self._lock(self._directory(kind, asset_id), shared=True):
            return self._get_locked(kind, asset_id)

    def _get_locked(self, kind: AssetKind | str, asset_id: str) -> GlobalAsset:
        path = self._manifest_path(kind, asset_id)
        if not path.is_file():
            raise FileNotFoundError(f"global asset not found: {self._kind(kind).value}/{asset_id}")
        return GlobalAsset.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, kind: AssetKind | str | None = None) -> tuple[GlobalAsset, ...]:
        kinds = (self._kind(kind),) if kind is not None else tuple(AssetKind)
        assets: list[GlobalAsset] = []
        for resolved_kind in kinds:
            kind_dir = self.root / resolved_kind.value
            if not kind_dir.is_dir():
                continue
            for manifest in sorted(kind_dir.glob("*/manifest.json")):
                assets.append(GlobalAsset.from_dict(json.loads(manifest.read_text(encoding="utf-8"))))
        return tuple(sorted(assets, key=lambda item: (item.kind.value, item.id)))

    def create(self, asset: GlobalAsset) -> GlobalAsset:
        directory = self._directory(asset.kind, asset.id)
        with self._lock(directory):
            if (directory / "manifest.json").exists():
                raise FileExistsError(f"global asset already exists: {asset.kind.value}/{asset.id}")
            self._write_manifest(directory / "manifest.json", asset)
        return asset

    def update(self, asset: GlobalAsset, *, expected_revision: int) -> GlobalAsset:
        directory = self._directory(asset.kind, asset.id)
        with self._lock(directory):
            current = self._get_locked(asset.kind, asset.id)
            if current.revision != expected_revision:
                raise ValueError(
                    f"revision conflict for {asset.kind.value}/{asset.id}: expected {expected_revision}, "
                    f"current is {current.revision}",
                )
            if asset.revision <= current.revision:
                raise ValueError("updated asset revision must increase")
            self._write_manifest(directory / "manifest.json", asset)
        return asset

    def update_look_artifacts(
        self,
        kind: AssetKind | str,
        asset_id: str,
        look_id: str,
        *,
        anchor_image: str | Path | None = None,
        sequence_video: str | Path | None = None,
        selected_frames: tuple[str | Path, ...] = (),
        sheet_image: str | Path | None = None,
        contact_sheet_image: str | Path | None = None,
        provenance: dict[str, str] | None = None,
        expected_revision: int,
    ) -> GlobalAsset:
        """Copy generated media and publish one new manifest revision.

        Files are staged before the manifest is replaced, so a manifest never
        references an artifact that was not successfully copied.
        """
        directory = self._directory(kind, asset_id)
        if not look_id or Path(look_id).name != look_id or look_id in {".", ".."}:
            raise ValueError("look id must be a single safe path component")
        with self._lock(directory):
            current = self._get_locked(kind, asset_id)
            if current.revision != expected_revision:
                raise ValueError(
                    f"revision conflict for {current.kind.value}/{current.id}: expected {expected_revision}, "
                    f"current is {current.revision}",
                )
            look = next((item for item in current.looks if item.id == look_id), None)
            if look is None:
                if current.looks:
                    raise ValueError(f"look not found for {current.kind.value}/{current.id}: {look_id}")
                look = AssetLook(look_id, look_id)

            supplied = {
                "anchor_image": anchor_image,
                "sequence_video": sequence_video,
                "sheet_image": sheet_image,
                "contact_sheet_image": contact_sheet_image,
            }
            frame_sources = tuple(Path(path) for path in selected_frames)
            for path in (*tuple(Path(path) for path in supplied.values() if path is not None), *frame_sources):
                if not path.is_file():
                    raise FileNotFoundError(f"generated artifact not found: {path}")

            stage_dir = Path(tempfile.mkdtemp(prefix="sheet-", dir=directory))
            try:
                staged_look_dir = stage_dir / "looks" / look_id
                staged_look_dir.mkdir(parents=True)

                destinations: dict[str, str] = {}
                for field_name, source in supplied.items():
                    if source is None:
                        continue
                    destination_name = {
                        "anchor_image": "anchor.png",
                        "sequence_video": "sequence.mp4",
                        "sheet_image": "sheet.png",
                        "contact_sheet_image": "contact-sheet.png",
                    }[field_name]
                    shutil.copy2(source, staged_look_dir / destination_name)
                    destinations[field_name] = f"looks/{look_id}/{destination_name}"

                frame_destinations: list[str] = []
                for source in frame_sources:
                    shutil.copy2(source, staged_look_dir / source.name)
                    frame_destinations.append(f"looks/{look_id}/{source.name}")

                target_look_dir = directory / "looks" / look_id
                target_look_dir.mkdir(parents=True, exist_ok=True)
                for staged_file in staged_look_dir.iterdir():
                    os.replace(staged_file, target_look_dir / staged_file.name)

                metadata = dict(look.metadata)
                metadata.update({str(key): str(value) for key, value in (provenance or {}).items()})
                updated_look = replace(
                    look,
                    anchor_image=destinations.get("anchor_image", look.anchor_image),
                    sequence_video=destinations.get("sequence_video", look.sequence_video),
                    selected_frames=tuple(frame_destinations) if selected_frames else look.selected_frames,
                    sheet_image=destinations.get("sheet_image", look.sheet_image),
                    contact_sheet_image=destinations.get("contact_sheet_image", look.contact_sheet_image),
                    metadata=tuple(metadata.items()),
                )
                looks = tuple(updated_look if item.id == look.id else item for item in current.looks)
                if not current.looks:
                    looks = (updated_look,)
                updated = replace(current, looks=looks, revision=current.revision + 1)
                self._write_manifest(directory / "manifest.json", updated)
                return updated
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)

    def delete(self, kind: AssetKind | str, asset_id: str) -> None:
        directory = self._directory(kind, asset_id)
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"global asset not found: {self._kind(kind).value}/{asset_id}")
        # Manifest first, media second: interleaved readers see "not found", never a half-removed asset.
        with self._lock(directory):
            if not manifest.is_file():
                raise FileNotFoundError(f"global asset not found: {self._kind(kind).value}/{asset_id}")
            manifest.unlink()
            for entry in sorted(directory.iterdir()):
                if entry.name == ".lock":
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        remove_lock = False
        with self._lock(directory):
            # A concurrent create may republish a new asset while the old one was removed;
            # leave any newly published state intact.
            if manifest.is_file() or any(entry.name != ".lock" for entry in directory.iterdir()):
                return
            remove_lock = True
        # The lock handle must be closed before removing the lock file on Windows.
        if remove_lock:
            try:
                (directory / ".lock").unlink()
            except FileNotFoundError:
                return
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass

    def materialize(
        self,
        kind: AssetKind | str,
        asset_id: str,
        look_id: str,
        project_reference_dir: str | Path,
    ) -> Path:
        source_dir = self._directory(kind, asset_id)
        with self._lock(source_dir, shared=True):
            asset = self._get_locked(kind, asset_id)
            look = next((item for item in asset.looks if item.id == look_id), None)
            if look is None:
                if asset.looks:
                    raise ValueError(f"look not found for {asset.kind.value}/{asset.id}: {look_id}")
                look = AssetLook(look_id, look_id)
            destination = Path(project_reference_dir).resolve() / "global_assets" / asset.kind.value / asset.id / look.id
            destination.mkdir(parents=True, exist_ok=True)
            files = tuple(
                path
                for path in (
                    look.hero_image,
                    look.sheet_image,
                    look.contact_sheet_image,
                    look.anchor_image,
                    look.sequence_video,
                    *look.selected_frames,
                    *look.references,
                )
                if path
            )
            for relative in files:
                source = (source_dir / relative).resolve()
                if source != source_dir and source_dir not in source.parents:
                    raise ValueError("asset media path escapes asset directory")
                if not source.is_file():
                    raise FileNotFoundError(f"global asset media not found: {relative}")
                target = destination / Path(relative).name
                shutil.copy2(source, target)
            snapshot = {"asset_id": asset.id, "kind": asset.kind.value, "look_id": look.id, "revision": asset.revision}
            atomic_write_json(destination / "manifest.json", snapshot)
            return destination

    def snapshot_revision(self, snapshot_dir: str | Path) -> int:
        payload = json.loads((Path(snapshot_dir) / "manifest.json").read_text(encoding="utf-8"))
        revision = payload.get("revision")
        if type(revision) is not int or revision < 1:
            raise ValueError("invalid global asset snapshot revision")
        return revision

    def is_stale(self, snapshot_dir: str | Path) -> bool:
        snapshot = json.loads((Path(snapshot_dir) / "manifest.json").read_text(encoding="utf-8"))
        current = self.get(snapshot["kind"], snapshot["asset_id"])
        return current.revision != self.snapshot_revision(snapshot_dir)
