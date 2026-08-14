"""Safe filesystem adapter for the canonical global asset library."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset


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
    def _lock(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".lock"
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
            current = self.get(asset.kind, asset.id)
            if current.revision != expected_revision:
                raise ValueError(
                    f"revision conflict for {asset.kind.value}/{asset.id}: expected {expected_revision}, "
                    f"current is {current.revision}"
                )
            if asset.revision <= current.revision:
                raise ValueError("updated asset revision must increase")
            self._write_manifest(directory / "manifest.json", asset)
        return asset

    def delete(self, kind: AssetKind | str, asset_id: str) -> None:
        directory = self._directory(kind, asset_id)
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(f"global asset not found: {self._kind(kind).value}/{asset_id}")
        shutil.rmtree(directory)

    def materialize(
        self,
        kind: AssetKind | str,
        asset_id: str,
        look_id: str,
        project_reference_dir: str | Path,
    ) -> Path:
        asset = self.get(kind, asset_id)
        look = next((item for item in asset.looks if item.id == look_id), None)
        if look is None:
            if asset.looks:
                raise ValueError(f"look not found for {asset.kind.value}/{asset.id}: {look_id}")
            look = AssetLook(look_id, look_id)
        source_dir = self._directory(asset.kind, asset.id)
        destination = Path(project_reference_dir).resolve() / "global_assets" / asset.kind.value / asset.id / look.id
        destination.mkdir(parents=True, exist_ok=True)
        files = tuple(path for path in (look.hero_image, look.sheet_image, *look.references) if path)
        for relative in files:
            source = (source_dir / relative).resolve()
            if source != source_dir and source_dir not in source.parents:
                raise ValueError("asset media path escapes asset directory")
            if not source.is_file():
                raise FileNotFoundError(f"global asset media not found: {relative}")
            target = destination / Path(relative).name
            shutil.copy2(source, target)
        snapshot = {"asset_id": asset.id, "kind": asset.kind.value, "look_id": look.id, "revision": asset.revision}
        (destination / "manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
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
