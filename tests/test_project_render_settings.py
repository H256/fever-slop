import hashlib
import tempfile
import unittest
from pathlib import Path

from feverslop.domain.project_render_settings import (
    ProjectRenderSettings,
    WorkflowSelection,
)


class WorkflowSelectionTests(unittest.TestCase):
    def test_records_repository_relative_path_and_content_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workflow = root / "workflows" / "video.json"
            workflow.parent.mkdir()
            workflow.write_text('{"steps": 8}', encoding="utf-8")

            selection = WorkflowSelection.from_path(workflow, root=root)

        self.assertEqual("workflows/video.json", selection.path)
        self.assertEqual(
            hashlib.sha256(b'{"steps": 8}').hexdigest(),
            selection.sha256,
        )

    def test_missing_workflow_fails_with_selected_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"

            with self.assertRaisesRegex(FileNotFoundError, "missing.json"):
                WorkflowSelection.from_path(missing, root=Path(temp_dir))


class ProjectRenderSettingsTests(unittest.TestCase):
    def _selection(self, root: Path, name: str, content: str) -> WorkflowSelection:
        path = root / "workflows" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return WorkflowSelection.from_path(path, root=root)

    def test_overlay_applies_dimensions_and_workflow_provenance_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = self._selection(root, "video.json", "video")
            hero = self._selection(root, "hero.json", "hero")
            edit = self._selection(root, "edit.json", "edit")
            source = {
                "scene": 1,
                "width": 1280,
                "height": 704,
                "references": {"actor_ids": ["hero"]},
            }
            settings = ProjectRenderSettings(
                width=1024,
                height=576,
                video_workflow=video,
                reference_hero_workflow=hero,
                reference_edit_workflow=edit,
            )

            result = settings.apply_to_scene(source)

        self.assertEqual((1280, 704), (source["width"], source["height"]))
        self.assertEqual((1024, 576), (result["width"], result["height"]))
        self.assertEqual(video.to_dict(), result["render_settings"]["video_workflow"])
        self.assertRegex(result["references"]["generator_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(["hero"], result["references"]["actor_ids"])

    def test_overlay_removes_managed_workflow_markers_when_project_returns_to_defaults(self):
        source = {
            "scene": 1,
            "width": 1280,
            "height": 704,
            "render_settings": {
                "video_workflow": {"path": "custom.json", "sha256": "a" * 64},
                "keep": "value",
            },
            "references": {
                "generator_fingerprint": "b" * 64,
                "actor_ids": ["hero"],
            },
        }

        result = ProjectRenderSettings(width=1280, height=704).apply_to_scene(source)

        self.assertEqual({"keep": "value"}, result["render_settings"])
        self.assertEqual({"actor_ids": ["hero"]}, result["references"])

    def test_reference_fingerprint_changes_when_workflow_content_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hero_path = root / "workflows" / "hero.json"
            hero_path.parent.mkdir()
            edit = self._selection(root, "edit.json", "edit")
            hero_path.write_text("first", encoding="utf-8")
            first = ProjectRenderSettings(
                width=1280,
                height=704,
                reference_hero_workflow=WorkflowSelection.from_path(hero_path, root=root),
                reference_edit_workflow=edit,
            ).apply_to_scene({"scene": 1})
            hero_path.write_text("second", encoding="utf-8")
            second = ProjectRenderSettings(
                width=1280,
                height=704,
                reference_hero_workflow=WorkflowSelection.from_path(hero_path, root=root),
                reference_edit_workflow=edit,
            ).apply_to_scene({"scene": 1})

        self.assertNotEqual(
            first["references"]["generator_fingerprint"],
            second["references"]["generator_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
