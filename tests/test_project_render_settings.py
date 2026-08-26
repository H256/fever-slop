import hashlib
import tempfile
import unittest
from pathlib import Path

from feverslop.composition.project_render_settings import resolve_project_render_settings
from feverslop.composition.config_loader import resolve_runner_path
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

    def test_project_reference_generation_overrides_runner_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.wav").write_bytes(b"")
            (root / "config.json").write_text(
                '{"input_audio":"song.wav","reference_generation":"sequence_sheet",'
                '"workflows":{"reference_sequence":"workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json"}}',
                encoding="utf-8",
            )

            resolved = resolve_project_render_settings(root, video_pipeline="minimax-h3-r2v")

        self.assertEqual("sequence_sheet", resolved.runner_overrides["reference_generation"])
        self.assertEqual(
            str(resolve_runner_path("workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json").resolve()),
            resolved.runner_overrides["sequence_to_sheet_workflow"],
        )

    def test_h3_r2v_selects_two_pass_workflow_without_project_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.wav").write_bytes(b"")
            (root / "config.json").write_text('{"input_audio":"song.wav"}', encoding="utf-8")

            resolved = resolve_project_render_settings(root, video_pipeline="minimax-h3-r2v")

        self.assertEqual(
            str(resolve_runner_path("workflows/video/minimax_h3/r2v_audio_two_pass.json").resolve()),
            resolved.runner_overrides["single_prompt_workflow"],
        )

    def test_explicit_reference_generation_overrides_project_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.wav").write_bytes(b"")
            (root / "config.json").write_text(
                '{"input_audio":"song.wav","reference_generation":"sequence_sheet"}',
                encoding="utf-8",
            )

            resolved = resolve_project_render_settings(
                root,
                video_pipeline="minimax-h3-r2v",
                explicit_runner_options={"reference_generation"},
                reference_generation="image_views",
            )

        self.assertEqual("image_views", resolved.settings.reference_generation)


if __name__ == "__main__":
    unittest.main()
