from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from feverslop.application.scene_workspace import (
    LoadSceneWorkspaceUseCase,
    PatchSceneUseCase,
    SceneWorkspaceSnapshot,
)
from feverslop.ports.scene_documents import (
    SceneDocumentSnapshot,
    SceneLtxPromptField,
)
from feverslop.studio.job_service import StudioJobRequest, StudioJobService


@dataclass(frozen=True)
class _SceneAction:
    job_action: str | None


_SCENE_ACTIONS: Mapping[str, _SceneAction] = MappingProxyType(
    {
        "render": _SceneAction(job_action="ltx-render-scenes"),
        "rerender": _SceneAction(job_action="ltx-render-scenes"),
        "retake": _SceneAction(job_action="ltx-render-scenes"),
        "ltx-render": _SceneAction(job_action="ltx-render-scenes"),
        "stage-1-preview": _SceneAction(job_action=None),
    },
)


def normalize_scene_numbers(scene_numbers: Iterable[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for scene_number in scene_numbers:
        if isinstance(scene_number, bool) or not isinstance(scene_number, int) or scene_number <= 0:
            raise ValueError("Scene numbers must be positive integers")
        normalized.add(scene_number)
    return tuple(sorted(normalized))


class SceneWorkspaceService:
    def __init__(
        self,
        *,
        load_workspace: LoadSceneWorkspaceUseCase,
        patch_scene: PatchSceneUseCase,
        jobs: StudioJobService,
        project_type: Callable[[str], str] | None = None,
    ) -> None:
        self._load_workspace = load_workspace
        self._patch_scene = patch_scene
        self._jobs = jobs
        self._project_type = project_type or (lambda _project_id: "standard_music_video")

    def load(self, project_id: str) -> SceneWorkspaceSnapshot:
        self._ensure_available(project_id)
        return self._load_workspace.execute(project_id)

    def patch_scene(
        self,
        *,
        project_id: str,
        scene_number: int,
        changes: Mapping[str, object],
        expected_revision: str,
        selected_ltx_prompt_field: SceneLtxPromptField | None = None,
    ) -> SceneDocumentSnapshot:
        self._ensure_available(project_id)
        return self._patch_scene.execute(
            project_id=project_id,
            scene_number=scene_number,
            changes=changes,
            expected_revision=expected_revision,
            selected_ltx_prompt_field=selected_ltx_prompt_field,
        )

    def start_action(
        self,
        *,
        project_id: str,
        action: str,
        scene_numbers: Iterable[int],
        preview_stage: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_available(project_id)
        action_spec = _SCENE_ACTIONS.get(action)
        if action_spec is None:
            raise ValueError(f"Unknown scene action: {action}")
        if action_spec.job_action is None or preview_stage is not None:
            if preview_stage not in {None, 1} or isinstance(preview_stage, bool):
                raise ValueError("preview_stage must be 1")
            raise ValueError(
                "Stage-1 preview is unavailable because no studio job action is registered",
            )

        scenes = normalize_scene_numbers(scene_numbers)
        if not scenes:
            raise ValueError("Scene action requires at least one scene")
        return self._jobs.start_job(
            project_id,
            StudioJobRequest(action=action_spec.job_action, scenes=list(scenes)),
        )

    def _ensure_available(self, project_id: str) -> None:
        if self._project_type(project_id) == "movie":
            raise ValueError("Scene workspace is unavailable for movie projects")
