from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.canonical_plan_regeneration import (
    CanonicalPlanRegenerationService,
)
from feverslop.errors import FeverSlopDataError
from feverslop.ports.reporting import Reporter
from feverslop.utils.io import read_json_document
from feverslop.pipeline.continuation_render_plan import project_continuation_sources


class CanonicalPlanRegenerator:
    def __init__(
        self,
        project_dir: str | Path,
        *,
        selected_scene_numbers: set[int] | None = None,
        reference_plan_path: str | Path | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.store = CanonicalPlanStore(project_dir)
        self.snapshot = self.store.capture_regeneration()
        if selected_scene_numbers is not None and not self.snapshot.exists:
            raise FeverSlopDataError(
                "Selected-scene regeneration requires an existing canonical base plan",
            )
        self.selected_scene_numbers = selected_scene_numbers
        self.reference_scenes = self._read_reference_scenes(reference_plan_path)
        self.reporter = reporter
        self.service = CanonicalPlanRegenerationService()

    def write(self, path: str | Path, scenes: list[dict[str, Any]]) -> Path:
        target = Path(path).resolve()
        if target != self.store.layout.base_plan.resolve():
            raise FeverSlopDataError(
                f"Canonical regenerator cannot write a different artifact: {target}",
            )
        existing = project_continuation_sources(self.snapshot.scenes, scenes)
        references = project_continuation_sources(self.reference_scenes, scenes)
        selected = self.selected_scene_numbers
        if selected is not None:
            selected = set(selected)
            selected.update(
                int(scene["scene"]) for scene in [*existing, *scenes]
                if scene.get("semantic_scene") in self.selected_scene_numbers
            )
        result = self.service.merge(
            existing,
            scenes,
            selected_scene_numbers=selected,
            reference_scenes=references,
        )
        if self.reporter is not None:
            for diagnostic in result.diagnostics:
                identity = diagnostic.scene_id or f"scene {diagnostic.scene_number}"
                self.reporter.message(
                    f"Canonical regeneration {diagnostic.severity}: "
                    f"{diagnostic.code} ({identity}): {diagnostic.message}",
                )
        return self.store.commit_regeneration(self.snapshot, result.scenes)

    @staticmethod
    def _read_reference_scenes(path: str | Path | None) -> tuple[dict[str, Any], ...]:
        if path is None or not Path(path).is_file():
            return ()
        value = read_json_document(path)
        if not isinstance(value, list) or any(not isinstance(scene, dict) for scene in value):
            raise FeverSlopDataError(f"Reference plan must be a list of objects: {path}")
        return tuple(value)
