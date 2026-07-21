from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from feverslop.domain.scene_workspace import SceneWorkspace
from feverslop.ports.scene_documents import (
    SceneDocumentPort,
    SceneDocumentSnapshot,
    SceneLtxPromptField,
    SceneMediaPort,
)


@dataclass(frozen=True)
class SceneWorkspaceSnapshot:
    workspace: SceneWorkspace
    revision: str


class ScenePatchRejected(ValueError):
    """Raised when a scene patch reaches outside the Release 1 edit surface."""


class LoadSceneWorkspaceUseCase:
    def __init__(self, *, documents: SceneDocumentPort, media: SceneMediaPort) -> None:
        self._documents = documents
        self._media = media

    def execute(self, project_id: str) -> SceneWorkspaceSnapshot:
        document = self._documents.load(project_id)
        media_by_scene = self._media.load_media(project_id)
        return SceneWorkspaceSnapshot(
            workspace=SceneWorkspace.from_scenes(
                document.scenes,
                media_by_scene=media_by_scene,
            ),
            revision=document.revision,
        )


class PatchSceneUseCase:
    _EDITABLE_FIELDS = frozenset({"shot_description", "z_image.prompt"})

    def __init__(self, *, documents: SceneDocumentPort) -> None:
        self._documents = documents

    def execute(
        self,
        *,
        project_id: str,
        scene_number: int,
        changes: Mapping[str, object],
        selected_ltx_prompt_field: SceneLtxPromptField,
        expected_revision: str,
    ) -> SceneDocumentSnapshot:
        canonical_changes = self._canonical_changes(
            changes,
            selected_ltx_prompt_field=selected_ltx_prompt_field,
        )
        return self._documents.patch_scene(
            project_id,
            scene_number,
            canonical_changes,
            expected_revision,
        )

    @classmethod
    def _canonical_changes(
        cls,
        changes: Mapping[str, object],
        *,
        selected_ltx_prompt_field: SceneLtxPromptField,
    ) -> dict[str, object]:
        editable_fields = set(cls._EDITABLE_FIELDS)
        selected_ltx_key: str | None = None
        if isinstance(selected_ltx_prompt_field, SceneLtxPromptField):
            selected_ltx_key = f"ltx.{selected_ltx_prompt_field.value}"
            editable_fields.add(selected_ltx_key)

        rejected = set(changes) - editable_fields
        if rejected:
            fields = ", ".join(sorted(repr(field) for field in rejected))
            raise ScenePatchRejected(f"Scene field is not editable: {fields}")
        if selected_ltx_key is None:
            raise ScenePatchRejected("Unknown selected LTX prompt field")

        for field_name, value in changes.items():
            if not isinstance(value, str):
                raise ScenePatchRejected(f"Scene field {field_name!r} must be text")

        canonical: dict[str, object] = {}
        if "shot_description" in changes:
            canonical["shot_description"] = changes["shot_description"]
        if "z_image.prompt" in changes:
            canonical["z_image"] = {"prompt": changes["z_image.prompt"]}
        if selected_ltx_key in changes:
            canonical["ltx"] = {
                selected_ltx_prompt_field.value: changes[selected_ltx_key]
            }
        return canonical
