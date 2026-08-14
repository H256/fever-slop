"""Resolve configured global assets into the existing local-reference shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from feverslop.domain.global_library import AssetKind


@dataclass(frozen=True, slots=True)
class GlobalCastResolution:
    actors: tuple[dict, ...]
    locations: tuple[dict, ...]
    styles: tuple[dict, ...]
    props: tuple[dict, ...]
    snapshots: tuple[dict, ...]


class GlobalCastResolver:
    def __init__(self, library: Any):
        self.library = library

    def _resolve_kind(
        self,
        kind: AssetKind,
        entries: Iterable[Any],
        project_reference_dir: Any,
    ) -> tuple[list[dict], list[dict]]:
        result: list[dict] = []
        snapshots: list[dict] = []
        for entry in entries:
            try:
                asset = self.library.get(kind, entry.asset_id)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"global {kind.value} '{entry.asset_id}' is missing; create or import it first"
                ) from exc
            look = next((item for item in asset.looks if item.id == entry.look_id), None)
            if look is None and asset.looks:
                raise ValueError(
                    f"global {kind.value} '{entry.asset_id}' has no look '{entry.look_id}'; "
                    "create-look or select an existing look"
                )
            snapshot_dir = self.library.materialize(kind, entry.asset_id, entry.look_id, project_reference_dir)
            record = {
                "id": asset.id,
                "name": asset.name,
                "role": entry.role,
                "visual_description": asset.description or (look.description if look else ""),
                "global_asset_id": asset.id,
                "global_look_id": entry.look_id,
                "global_revision": asset.revision,
                "snapshot_path": str(snapshot_dir),
            }
            if look is not None:
                record["hero_path"] = f"{snapshot_dir}/{look.hero_image.rsplit('/', 1)[-1]}" if look.hero_image else ""
                record["sheet_path"] = f"{snapshot_dir}/{look.sheet_image.rsplit('/', 1)[-1]}" if look.sheet_image else ""
            result.append(record)
            snapshots.append({"kind": kind.value, "asset_id": asset.id, "look_id": entry.look_id, "revision": asset.revision, "path": str(snapshot_dir)})
        return result, snapshots

    def resolve(
        self,
        *,
        cast: Iterable[Any],
        locations: Iterable[Any],
        styles: Iterable[Any],
        props: Iterable[Any],
        project_reference_dir: Any,
    ) -> GlobalCastResolution:
        reference_dir = project_reference_dir
        actors, actor_snapshots = self._resolve_kind(AssetKind.CHARACTER, cast, reference_dir)
        resolved_locations, location_snapshots = self._resolve_kind(AssetKind.LOCATION, locations, reference_dir)
        resolved_styles, style_snapshots = self._resolve_kind(AssetKind.STYLE, styles, reference_dir)
        resolved_props, prop_snapshots = self._resolve_kind(AssetKind.PROP, props, reference_dir)
        return GlobalCastResolution(
            actors=tuple(actors), locations=tuple(resolved_locations), styles=tuple(resolved_styles), props=tuple(resolved_props),
            snapshots=tuple(actor_snapshots + location_snapshots + style_snapshots + prop_snapshots),
        )


def materialize_global_assets(project_config, app_config, *, refresh: bool = False) -> GlobalCastResolution:
    """Materialize declarations before reference generation; refresh is explicit."""
    adapter_type = __import__("feverslop.adapters.global_library", fromlist=["GlobalLibraryAdapter"]).GlobalLibraryAdapter
    resolver = GlobalCastResolver(adapter_type(app_config.global_library_path))
    if not refresh:
        # Existing snapshots are intentionally left alone; callers can inspect their staleness.
        pass
    return resolver.resolve(
        cast=project_config.global_cast,
        locations=project_config.global_locations,
        styles=project_config.global_styles,
        props=project_config.global_props,
        project_reference_dir=project_config.project_dir / "output" / "references",
    )
