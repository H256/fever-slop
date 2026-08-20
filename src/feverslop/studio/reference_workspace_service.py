from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feverslop.adapters.project_reference_library import ProjectReferenceLibrary
from feverslop.application.reference_workspace import (
    GenerationJobUseCase,
    ImportReferenceUseCase,
    LoadReferenceWorkspaceUseCase,
    PreviewSceneAssignmentUseCase,
    SaveSceneAssignmentsUseCase,
)
from feverslop.domain.reference_workspace import (
    PropInteraction,
    ReferenceAsset,
    ReferenceKind,
    ReferenceWorkspaceSnapshot,
    SceneReferenceAssignment,
)
from feverslop.ports.reference_library import GenerationJobPort


@dataclass(frozen=True)
class CommandError:
    code: str
    message: str


@dataclass(frozen=True)
class CommandResult:
    success: bool
    error: CommandError | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationCommand:
    action: str
    scene_number: int | None = None
    reference_ids: tuple[str, ...] = ()
    actor_ids: tuple[str, ...] = ()
    location_ids: tuple[str, ...] = ()
    background_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaleState:
    stale_assets: tuple[str, ...]
    invalidated_artifacts: dict[str, Any]


class ReferenceWorkspaceService:
    """Coordinates reference workspace operations without Qt dependency."""

    def __init__(self, *, project_root: Path, max_scene_actors: int = 4, generation_jobs: GenerationJobPort | None = None):
        self._project_root = project_root
        self._library = ProjectReferenceLibrary(project_root, max_scene_actors=max_scene_actors)
        self._load = LoadReferenceWorkspaceUseCase(self._library)
        self._preview = PreviewSceneAssignmentUseCase(
            bible=self._library,
            scene_cast=self._library,
        )
        self._save = SaveSceneAssignmentsUseCase(
            library=self._library,
            bible=self._library,
            scene_cast=self._library,
            invalidation=self._library,
        )
        self._import = ImportReferenceUseCase(importer=self._library)
        self._generation = GenerationJobUseCase(
            jobs=generation_jobs if generation_jobs is not None else FakeGenerationJobRegistry(),
        )
        self._current_project: str | None = None

    def set_project(self, project_id: str) -> None:
        self._current_project = project_id

    def load_library(self, project_id: str) -> ReferenceWorkspaceSnapshot:
        return self._load.load(project_id)

    def filter_library(
        self,
        project_id: str,
        kinds: list[ReferenceKind] | None = None,
        stale_only: bool = False,
        missing_only: bool = False,
    ) -> tuple[ReferenceAsset, ...]:
        return self._load.filter(
            project_id,
            kinds=kinds,
            stale_only=stale_only,
            missing_only=missing_only,
        )

    def get_asset(self, project_id: str, asset_id: str) -> ReferenceAsset | None:
        snap = self.load_library(project_id)
        return snap.get_asset(asset_id)

    def scenes_using_asset(self, project_id: str, asset_id: str) -> tuple[int, ...]:
        snap = self.load_library(project_id)
        return snap.scenes_using_asset(asset_id)

    def preview_assignment(
        self,
        project_id: str,
        scene_number: int,
        actor_ids: tuple[str, ...] = (),
        location_ids: tuple[str, ...] = (),
        background_ids: tuple[str, ...] = (),
        style_ids: tuple[str, ...] = (),
        actor_look_ids: dict[str, str] | None = None,
        prop_ids: tuple[str, ...] = (),
        prop_interactions: tuple[dict[str, Any], ...] = (),
    ) -> CommandResult:
        try:
            assignment = SceneReferenceAssignment(
                scene_number=scene_number,
                actor_ids=actor_ids,
                location_ids=location_ids,
                  background_ids=background_ids or (),
                  style_ids=style_ids,
                  actor_look_ids=actor_look_ids,
                  prop_ids=prop_ids,
                  prop_interactions=tuple(PropInteraction.from_dict(item) for item in prop_interactions),
              )
        except ValueError as e:
            return CommandResult(success=False, error=CommandError("invalid_assignment", str(e)))

        issues = self._preview.preview(assignment, project_id=project_id)
        if issues:
            return CommandResult(
                success=False,
                error=CommandError("validation_errors", "; ".join(issues)),
                data={"issues": issues},
            )
        return CommandResult(success=True, data={"assignment": _assignment_to_dict(assignment)})

    def save_assignments(
        self,
        project_id: str,
        assignments: tuple[dict, ...],
        expected_revision: str,
        old_snapshot: ReferenceWorkspaceSnapshot | None = None,
    ) -> CommandResult:
        try:
            parsed = tuple(
                _assignment_from_dict(a) for a in assignments
            )
        except (ValueError, KeyError) as e:
            return CommandResult(success=False, error=CommandError("invalid_assignment", str(e)))

        try:
            result = self._save.save(project_id, parsed, expected_revision, old_snapshot=old_snapshot)
        except ValueError as e:
            return CommandResult(success=False, error=CommandError("revision_mismatch", str(e)))

        return CommandResult(
            success=not result.issues,
            error=CommandError("validation_errors", "; ".join(result.issues)) if result.issues else None,
            data={
                "new_revision": result.new_revision,
                "affected_scenes": list(result.affected_scenes),
                "invalidated_artifacts": result.invalidated_artifacts,
                "issues": list(result.issues),
            },
        )

    def import_asset(
        self,
        project_id: str,
        source_path: Path,
        asset_data: dict[str, Any],
    ) -> CommandResult:
        try:
            asset = ReferenceAsset(
                id=str(asset_data.get("id", "")),
                kind=ReferenceKind(asset_data.get("kind", "actor")),
                label=str(asset_data.get("label", "")),
            )
        except ValueError as e:
            return CommandResult(success=False, error=CommandError("invalid_kind", str(e)))

        if not asset.id:
            return CommandResult(success=False, error=CommandError("missing_id", "asset id is required"))

        try:
            result = self._import.import_asset(project_id, source_path, asset)
        except ValueError as e:
            return CommandResult(success=False, error=CommandError("import_failed", str(e)))

        return CommandResult(success=True, data={
            "asset": _asset_to_dict(result),
        })

    def queue_generation(self, project_id: str, command: GenerationCommand) -> CommandResult:
        mapping = {
            "storyboard_frame": "queue_storyboard_frame",
            "storyboard_page": "queue_storyboard_frame",
            "msr_sheet": "queue_msr_sheet",
            "ingredients_sheet": "queue_ingredients_sheet",
            "reference_rerender": "queue_reference_rerender",
        }
        method_name = mapping.get(command.action)
        if not method_name and command.action not in PIPELINE_ACTIONS:
            return CommandResult(
                success=False,
                error=CommandError("unknown_action", f"Unknown action: {command.action}"),
            )

        if method_name == "queue_storyboard_frame":
            job_id = self._generation.queue_storyboard_frame(
                project_id, command.scene_number or 1, command.reference_ids,
            )
        elif method_name == "queue_msr_sheet":
            if not command.scene_number:
                return CommandResult(success=False, error=CommandError("missing_scene", "scene_number required"))
            job_id = self._generation.queue_msr_sheet(
                project_id, command.scene_number, command.actor_ids, command.location_ids,
            )
        elif method_name == "queue_ingredients_sheet":
            if not command.scene_number:
                return CommandResult(success=False, error=CommandError("missing_scene", "scene_number required"))
            job_id = self._generation.queue_ingredients_sheet(
                project_id, command.scene_number, command.actor_ids, command.location_ids, command.background_ids,
            )
        elif method_name == "queue_reference_rerender":
            if not command.reference_ids:
                return CommandResult(success=False, error=CommandError("missing_reference", "reference_id required"))
            job_id = self._generation.queue_reference_rerender(project_id, command.reference_ids[0])
        else:
            return CommandResult(
                success=False,
                error=CommandError("unsupported", f"Pipeline action not supported: {command.action}"),
            )

        return CommandResult(success=True, data={"job_id": job_id})

    def check_stale(self, project_id: str) -> StaleState:
        snap = self.load_library(project_id)
        stale = tuple(a.id for a in snap.filter_assets(stale_only=True))
        invalidated: dict[str, Any] = {}
        for a in snap.filter_assets(stale_only=True):
            for sn in snap.scenes_using_asset(a.id):
                invalidated.setdefault(f"scene_{sn}", []).append(a.id)
        return StaleState(stale_assets=stale, invalidated_artifacts=invalidated)


def _assignment_to_dict(a: SceneReferenceAssignment) -> dict[str, Any]:
    return {
        "scene_number": a.scene_number,
        "actor_ids": list(a.actor_ids),
        "location_ids": list(a.location_ids),
        "background_ids": list(a.background_ids),
        "style_ids": list(a.style_ids),
        "actor_look_ids": dict(a.actor_look_ids or {}),
        "prop_ids": list(a.prop_ids),
        "prop_interactions": [item.to_dict() for item in a.prop_interactions],
    }


def _assignment_from_dict(d: dict[str, Any]) -> SceneReferenceAssignment:
    return SceneReferenceAssignment(
        scene_number=int(d["scene_number"]),
        actor_ids=tuple(d.get("actor_ids") or ()),
        location_ids=tuple(d.get("location_ids") or ()),
        background_ids=tuple(d.get("background_ids") or ()),
        style_ids=tuple(d.get("style_ids") or ()),
        actor_look_ids=dict(d.get("actor_look_ids") or {}),
        prop_ids=tuple(d.get("prop_ids") or ()),
        prop_interactions=tuple(PropInteraction.from_dict(item) for item in d.get("prop_interactions") or []),
    )


def _asset_to_dict(a: ReferenceAsset) -> dict[str, Any]:
    prov = a.provenance
    return {
        "id": a.id,
        "kind": a.kind.value,
        "label": a.label,
        "path": a.path,
        "width": a.width,
        "height": a.height,
        "exists": a.exists,
        "stale": a.stale,
        "provenance": {
            "source": prov.source if prov else "",
            "generated_at": prov.generated_at if prov else "",
            "job_action": prov.job_action if prov else "",
        },
    }


PIPELINE_ACTIONS = {
    "anchor-fix",
    "relay-compact",
    "storyboard-frames",
    "storyboard-page",
    "msr-reference-sheets",
    "msr-prompt-enrich",
    "rebuild-plan",
    "rebuild-plan-timeline",
    "generate-storyboard-frames",
    "generate-storyboard-page",
    "generate-ingredients-sheets",
    "generate-reference-sheets",
}


class FakeGenerationJobRegistry:
    """In-memory job registry for testing."""

    def __init__(self):
        self._counter = 0
        self.jobs: list[dict[str, Any]] = []

    def _next_id(self):
        self._counter += 1
        return f"job_{self._counter}"

    def queue_storyboard_frame(self, project_id: str, scene_number: int, reference_ids: tuple[str, ...]) -> str:
        self.jobs.append({"action": "storyboard_frame", "scene": scene_number, "refs": reference_ids})
        return self._next_id()

    def queue_msr_sheet(self, project_id: str, scene_number: int, actor_ids: tuple[str, ...], location_ids: tuple[str, ...] = ()) -> str:
        self.jobs.append({"action": "msr_sheet", "scene": scene_number, "actors": actor_ids, "locations": location_ids})
        return self._next_id()

    def queue_ingredients_sheet(self, project_id: str, scene_number: int, actor_ids: tuple[str, ...], location_ids: tuple[str, ...] = (), background_ids: tuple[str, ...] = ()) -> str:
        self.jobs.append({"action": "ingredients_sheet", "scene": scene_number, "actors": actor_ids, "locations": location_ids, "backgrounds": background_ids})
        return self._next_id()

    def queue_reference_rerender(self, project_id: str, reference_id: str) -> str:
        self.jobs.append({"action": "rerender", "ref": reference_id})
        return self._next_id()
