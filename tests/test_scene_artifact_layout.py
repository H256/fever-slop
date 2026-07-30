import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory

from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.config.project_config import ProjectConfig, ProjectPaths


class SceneArtifactLayoutTests(unittest.TestCase):
    def test_ingredients_sheet_cache_has_canonical_signature_directory(self):
        project = Path("C:/projects/demo")
        layout = SceneArtifactLayout(project)

        self.assertEqual(
            project
            / "output"
            / "references"
            / "ingredients_sheets"
            / "by_signature",
            layout.ingredients_sheet_cache_dir,
        )

    def test_exposes_canonical_render_artifact_paths(self):
        project = Path("/project")
        layout = SceneArtifactLayout(project)

        self.assertEqual(project / "output" / "render" / "plans" / "base.json", layout.base_plan)
        self.assertEqual(project / "output" / "render" / "plans" / "references.json", layout.references_plan)
        self.assertEqual(project / "output" / "render" / "plans" / "ingredients.json", layout.ingredients_plan)
        self.assertEqual(project / "output" / "render" / "plans" / "compact.json", layout.compact_plan)
        self.assertEqual(project / "output" / "render" / "plans" / "anchored.json", layout.anchored_plan)
        self.assertEqual(project / "output" / "render" / "scenes" / "scene_0005", layout.scene_dir(5))
        self.assertEqual(layout.scene_dir(5) / "manifest.json", layout.scene_manifest(5))
        self.assertEqual(layout.scene_dir(5) / "workflow.json", layout.scene_workflow(5))
        self.assertEqual(layout.scene_dir(5) / "raw.mp4", layout.scene_raw_video(5))
        self.assertEqual(layout.scene_dir(5) / "final.mp4", layout.scene_final_video(5))
        self.assertEqual(project / "output" / "render" / "storyboard", layout.storyboard_dir)
        self.assertEqual(project / "output" / "render" / "final" / "video_only.mp4", layout.video_only)
        self.assertEqual(project / "output" / "render" / "final" / "movie.mp4", layout.movie)

        with self.assertRaises(FrozenInstanceError):
            layout.project_dir = Path("/elsewhere")

    def test_resolves_new_scene_video_before_legacy_fallbacks(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            layout = SceneArtifactLayout(project)
            legacy = layout.render_dir / "ltx_msr"
            legacy.mkdir(parents=True)
            old_clip = legacy / "scene_0005.mp4"
            old_clip.touch()

            self.assertEqual(old_clip, layout.find_scene_final_video(5, legacy_dirs=[legacy]))

            new_clip = layout.scene_final_video(5)
            new_clip.parent.mkdir(parents=True)
            new_clip.touch()
            self.assertEqual(new_clip, layout.find_scene_final_video(5, legacy_dirs=[legacy]))

    def test_legacy_fallback_supports_final_subdirectory(self):
        with TemporaryDirectory() as tmp:
            layout = SceneArtifactLayout(Path(tmp))
            legacy = layout.render_dir / "ltx_relay"
            clip = legacy / "final" / "scene_0003.mp4"
            clip.parent.mkdir(parents=True)
            clip.touch()

            self.assertEqual(clip, layout.find_scene_final_video(3, legacy_dirs=[legacy]))

    def test_project_paths_exposes_the_shared_layout(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            config_path = project / "config.json"
            config_path.write_text('{"project_name":"demo","input_audio":"song.mp3"}', encoding="utf-8")
            paths = ProjectPaths.from_config(ProjectConfig.load(config_path))

            self.assertEqual(SceneArtifactLayout(project), paths.artifact_layout)

    def test_plan_reads_prefer_canonical_path_and_fall_back_to_legacy(self):
        with TemporaryDirectory() as tmp:
            layout = SceneArtifactLayout(Path(tmp))
            legacy = layout.render_dir / "render_plan_song.json"
            legacy.parent.mkdir(parents=True)
            legacy.touch()

            self.assertEqual(legacy, layout.find_plan(layout.base_plan, legacy_paths=[legacy]))

            layout.base_plan.parent.mkdir(parents=True)
            layout.base_plan.touch()
            self.assertEqual(layout.base_plan, layout.find_plan(layout.base_plan, legacy_paths=[legacy]))


if __name__ == "__main__":
    unittest.main()
