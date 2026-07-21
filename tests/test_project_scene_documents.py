from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.project_scene_documents import ProjectSceneDocuments
from feverslop.ports.scene_documents import SceneDocumentConflict
from feverslop.studio.artifact_catalog import ArtifactCatalog


class CatalogStub:
    def __init__(self, artifacts: dict[str, list[str]]) -> None:
        self.artifacts = artifacts
        self.requests: list[str] = []

    def list_artifacts(self, project_id: str) -> dict[str, list[str]]:
        self.requests.append(project_id)
        return self.artifacts


class ProjectSceneDocumentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.projects_root = Path(self.temporary_directory.name)
        self.project = self.projects_root / "demo"
        self.project.mkdir()

    def _project_root(self, project_id: str) -> Path:
        self.assertEqual("demo", project_id)
        return self.project

    def _write_plan(self, relative_path: str, scenes: object) -> tuple[Path, bytes]:
        path = self.project / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(scenes, ensure_ascii=False, separators=(",", ":")).encode()
        path.write_bytes(payload)
        return path, payload

    def test_load_uses_first_render_plan_from_supplied_catalog_and_hashes_exact_bytes(self):
        legacy, _ = self._write_plan("render_plan_legacy.json", [{"scene": 9}])
        preferred, payload = self._write_plan(
            "output/render/plans/base.json",
            [{"scene": 1, "shot_description": "Überblick"}],
        )
        catalog = CatalogStub(
            {
                "render_plans": [preferred.relative_to(self.project).as_posix(), legacy.name],
                "images": [],
                "videos": [],
                "generated_json": [],
            }
        )

        snapshot = ProjectSceneDocuments(self._project_root, catalog=catalog).load("demo")

        self.assertEqual([{"scene": 1, "shot_description": "Überblick"}], snapshot.to_scenes())
        self.assertEqual(hashlib.sha256(payload).hexdigest(), snapshot.revision)
        self.assertEqual(["demo"], catalog.requests)

    def test_default_catalog_preserves_its_canonical_before_legacy_order(self):
        self._write_plan("output/render/render_plan_song.json", [{"scene": 2}])
        self._write_plan("output/render/plans/base.json", [{"scene": 1}])

        snapshot = ProjectSceneDocuments(
            self._project_root,
            catalog=ArtifactCatalog(self._project_root),
        ).load("demo")

        self.assertEqual([{"scene": 1}], snapshot.to_scenes())

    def test_load_rejects_missing_malformed_and_non_list_render_plans(self):
        missing_catalog = CatalogStub(
            {"render_plans": [], "images": [], "videos": [], "generated_json": []}
        )
        with self.assertRaisesRegex(FileNotFoundError, "render plan"):
            ProjectSceneDocuments(self._project_root, catalog=missing_catalog).load("demo")

        for contents, message in ((b"{broken", "Malformed render plan JSON"), (b'{"scene":1}', "JSON array")):
            with self.subTest(contents=contents):
                path = self.project / "plan.json"
                path.write_bytes(contents)
                catalog = CatalogStub(
                    {"render_plans": ["plan.json"], "images": [], "videos": [], "generated_json": []}
                )
                with self.assertRaisesRegex(ValueError, message):
                    ProjectSceneDocuments(self._project_root, catalog=catalog).load("demo")

    def test_patch_merges_one_prompt_field_without_changing_any_other_decoded_value(self):
        original = [
            {
                "scene": 1,
                "shot_description": "Wide",
                "ltx": {
                    "original_style_i2v_prompt": "old",
                    "base_prompt": "preserve",
                    "weights": [0.25, 1],
                },
                "extension": {"enabled": True, "nullable": None},
            },
            {"scene": 2, "shot_description": "Untouched"},
        ]
        path, payload = self._write_plan("output/render/plans/base.json", original)
        catalog = CatalogStub(
            {"render_plans": [path.relative_to(self.project).as_posix()], "images": [], "videos": [], "generated_json": []}
        )
        adapter = ProjectSceneDocuments(self._project_root, catalog=catalog)

        snapshot = adapter.patch_scene(
            "demo",
            1,
            {"ltx": {"original_style_i2v_prompt": "new"}},
            hashlib.sha256(payload).hexdigest(),
        )

        expected = json.loads(json.dumps(original))
        expected[0]["ltx"]["original_style_i2v_prompt"] = "new"
        decoded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(expected, decoded)
        self.assertEqual(expected, snapshot.to_scenes())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), snapshot.revision)
        self.assertFalse(any(path.parent.glob(f".{path.name}.*.tmp")))

    def test_patch_rejects_missing_scene_without_writing(self):
        path, payload = self._write_plan("plan.json", [{"scene": 1}])
        catalog = CatalogStub(
            {"render_plans": ["plan.json"], "images": [], "videos": [], "generated_json": []}
        )

        with self.assertRaisesRegex(KeyError, "Scene 2"):
            ProjectSceneDocuments(self._project_root, catalog=catalog).patch_scene(
                "demo", 2, {"shot_description": "missing"}, hashlib.sha256(payload).hexdigest()
            )

        self.assertEqual(payload, path.read_bytes())

    def test_patch_detects_revision_conflict_from_bytes_immediately_before_write(self):
        path, payload = self._write_plan("plan.json", [{"scene": 1, "shot_description": "old"}])
        catalog = CatalogStub(
            {"render_plans": ["plan.json"], "images": [], "videos": [], "generated_json": []}
        )
        adapter = ProjectSceneDocuments(self._project_root, catalog=catalog)
        expected_revision = hashlib.sha256(payload).hexdigest()
        changed = b'[{"scene":1,"shot_description":"external"}]'
        path.write_bytes(changed)

        with self.assertRaises(SceneDocumentConflict) as caught:
            adapter.patch_scene("demo", 1, {"shot_description": "ours"}, expected_revision)

        self.assertEqual(hashlib.sha256(changed).hexdigest(), caught.exception.actual_revision)
        self.assertEqual(changed, path.read_bytes())

    def test_patch_rechecks_revision_after_merging_and_immediately_before_replace(self):
        path, payload = self._write_plan("plan.json", [{"scene": 1, "shot_description": "old"}])
        catalog = CatalogStub(
            {"render_plans": ["plan.json"], "images": [], "videos": [], "generated_json": []}
        )
        changed = b'[{"scene":1,"shot_description":"external-during-merge"}]'

        class MutatingChanges(dict[str, object]):
            def items(self):
                path.write_bytes(changed)
                return super().items()

        with self.assertRaises(SceneDocumentConflict) as caught:
            ProjectSceneDocuments(self._project_root, catalog=catalog).patch_scene(
                "demo",
                1,
                MutatingChanges(shot_description="ours"),
                hashlib.sha256(payload).hexdigest(),
            )

        self.assertEqual(hashlib.sha256(changed).hexdigest(), caught.exception.actual_revision)
        self.assertEqual(changed, path.read_bytes())

    def test_catalog_paths_outside_project_root_are_refused(self):
        outside = self.projects_root / "outside.json"
        outside.write_text("[]", encoding="utf-8")
        catalog = CatalogStub(
            {"render_plans": ["../outside.json"], "images": [], "videos": [], "generated_json": []}
        )

        with self.assertRaisesRegex(ValueError, "outside project root"):
            ProjectSceneDocuments(self._project_root, catalog=catalog).load("demo")

    def test_load_media_maps_only_catalog_returned_scene_artifacts(self):
        paths = [
            "output/render/scenes/scene_0001/preview.png",
            "output/render/scenes/scene_0001/final.mp4",
            "output/render/scenes/scene_0002/workflow.json",
        ]
        for relative_path in paths:
            path = self.project / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact")
        uncatalogued = self.project / "output/render/scenes/scene_0003/final.mp4"
        uncatalogued.parent.mkdir(parents=True)
        uncatalogued.write_bytes(b"not catalogued")
        catalog = CatalogStub(
            {
                "render_plans": [],
                "images": [paths[0]],
                "videos": [paths[1]],
                "generated_json": [paths[2]],
            }
        )

        media = ProjectSceneDocuments(self._project_root, catalog=catalog).load_media("demo")

        self.assertEqual(paths[0], media[1].thumbnail_path)
        self.assertEqual(paths[1], media[1].video_path)
        self.assertEqual(paths[2], media[2].workflow_path)
        self.assertNotIn(3, media)
        self.assertEqual(["demo"], catalog.requests)


if __name__ == "__main__":
    unittest.main()
