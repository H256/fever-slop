from __future__ import annotations

import unittest
from collections.abc import Mapping

from feverslop.application.scene_workspace import (
    LoadSceneWorkspaceUseCase,
    PatchSceneUseCase,
    ScenePatchRejected,
)
from feverslop.domain.scene_workspace import SceneMedia
from feverslop.ports.scene_documents import (
    SceneDocumentConflict,
    SceneDocumentSnapshot,
    SceneLtxPromptField,
)


class FakeSceneDocuments:
    def __init__(self, snapshot: SceneDocumentSnapshot) -> None:
        self.snapshot = snapshot
        self.patch_calls: list[dict[str, object]] = []
        self.conflict = False

    def load(self, project_id: str) -> SceneDocumentSnapshot:
        return self.snapshot

    def patch_scene(
        self,
        project_id: str,
        scene_number: int,
        changes: Mapping[str, object],
        expected_revision: str,
    ) -> SceneDocumentSnapshot:
        if self.conflict:
            raise SceneDocumentConflict(project_id, expected_revision, "revision-2")
        self.patch_calls.append(
            {
                "project_id": project_id,
                "scene_number": scene_number,
                "changes": dict(changes),
                "expected_revision": expected_revision,
            }
        )
        return SceneDocumentSnapshot(scenes=self.snapshot.scenes, revision="revision-2")


class FakeSceneMedia:
    def __init__(self, media_by_scene: Mapping[int, SceneMedia]) -> None:
        self.media_by_scene = media_by_scene
        self.loaded_projects: list[str] = []

    def load_media(self, project_id: str) -> Mapping[int, SceneMedia]:
        self.loaded_projects.append(project_id)
        return self.media_by_scene


class SceneWorkspaceApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = FakeSceneDocuments(
            SceneDocumentSnapshot(
                scenes=(
                    {
                        "scene": 1,
                        "shot_description": "Wide shot",
                        "z_image": {"prompt": "Amber stage"},
                        "ltx": {"original_style_i2v_prompt": "Slow dolly"},
                    },
                    {"scene": 2, "shot_description": "Close-up"},
                ),
                revision="revision-1",
            )
        )

    def test_load_combines_document_snapshot_with_media_facts(self):
        media = FakeSceneMedia(
            {
                1: SceneMedia(
                    thumbnail_path="scenes/0001/preview.png",
                    video_path="output/final/scene_0001.mp4",
                ),
                2: SceneMedia(workflow_path="scenes/0002/workflow.json"),
            }
        )

        result = LoadSceneWorkspaceUseCase(
            documents=self.documents,
            media=media,
        ).execute("demo")

        self.assertEqual("revision-1", result.revision)
        self.assertEqual((1, 2), tuple(item.scene_number for item in result.workspace.items))
        self.assertEqual("rendered", result.workspace.items[0].status)
        self.assertEqual("scenes/0001/preview.png", result.workspace.items[0].media.thumbnail_path)
        self.assertEqual("planned", result.workspace.items[1].status)
        self.assertEqual(["demo"], media.loaded_projects)

    def test_patch_forwards_only_explicit_release_one_fields(self):
        result = PatchSceneUseCase(documents=self.documents).execute(
            project_id="demo",
            scene_number=1,
            changes={
                "shot_description": "Tracking shot",
                "image_prompt": "Blue stage",
                "ltx_prompt": "Fast dolly",
            },
            selected_ltx_prompt_field=SceneLtxPromptField.ORIGINAL_STYLE_I2V_PROMPT,
            expected_revision="revision-1",
        )

        self.assertEqual("revision-2", result.revision)
        self.assertEqual(
            [
                {
                    "project_id": "demo",
                    "scene_number": 1,
                    "changes": {
                        "shot_description": "Tracking shot",
                        "z_image": {"prompt": "Blue stage"},
                        "ltx": {"original_style_i2v_prompt": "Fast dolly"},
                    },
                    "expected_revision": "revision-1",
                }
            ],
            self.documents.patch_calls,
        )

    def test_patch_writes_ltx_prompt_to_the_selected_field_only(self):
        PatchSceneUseCase(documents=self.documents).execute(
            project_id="demo",
            scene_number=1,
            changes={"ltx_prompt": "Replacement"},
            selected_ltx_prompt_field=SceneLtxPromptField.I2V_PROMPT_FROM_T2I,
            expected_revision="revision-1",
        )

        self.assertEqual(
            {"ltx": {"i2v_prompt_from_t2i": "Replacement"}},
            self.documents.patch_calls[0]["changes"],
        )

    def test_patch_rejects_unknown_ltx_prompt_field(self):
        with self.assertRaisesRegex(ScenePatchRejected, "LTX prompt field"):
            PatchSceneUseCase(documents=self.documents).execute(
                project_id="demo",
                scene_number=1,
                changes={"ltx_prompt": "Replacement"},
                selected_ltx_prompt_field="ltx.arbitrary",  # type: ignore[arg-type]
                expected_revision="revision-1",
            )

        self.assertEqual([], self.documents.patch_calls)

    def test_patch_rejects_protected_and_arbitrary_fields(self):
        invalid_changes = (
            {"scene": 9},
            {"video_path": "elsewhere.mp4"},
            {"output.video_path": "elsewhere.mp4"},
            {"width": 1920},
            {"height": 1080},
            {"z_image": {"prompt": "Nested bypass"}},
            {"z_image.prompt": "Dotted bypass"},
            {"ltx": {"base_prompt": "Nested bypass"}},
            {"ltx.arbitrary.value": "Dotted bypass"},
        )
        use_case = PatchSceneUseCase(documents=self.documents)

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ScenePatchRejected, "not editable"):
                    use_case.execute(
                        project_id="demo",
                        scene_number=1,
                        changes=changes,
                        selected_ltx_prompt_field=SceneLtxPromptField.ORIGINAL_STYLE_I2V_PROMPT,
                        expected_revision="revision-1",
                    )

        self.assertEqual([], self.documents.patch_calls)

    def test_patch_rejects_non_text_values(self):
        with self.assertRaisesRegex(ScenePatchRejected, "must be text"):
            PatchSceneUseCase(documents=self.documents).execute(
                project_id="demo",
                scene_number=1,
                changes={"shot_description": {"nested": "value"}},
                selected_ltx_prompt_field=SceneLtxPromptField.ORIGINAL_STYLE_I2V_PROMPT,
                expected_revision="revision-1",
            )

        self.assertEqual([], self.documents.patch_calls)

    def test_patch_propagates_optimistic_concurrency_conflict(self):
        self.documents.conflict = True

        with self.assertRaises(SceneDocumentConflict) as caught:
            PatchSceneUseCase(documents=self.documents).execute(
                project_id="demo",
                scene_number=1,
                changes={"shot_description": "Tracking shot"},
                selected_ltx_prompt_field=SceneLtxPromptField.ORIGINAL_STYLE_I2V_PROMPT,
                expected_revision="stale-revision",
            )

        self.assertEqual("stale-revision", caught.exception.expected_revision)
        self.assertEqual([], self.documents.patch_calls)


if __name__ == "__main__":
    unittest.main()
