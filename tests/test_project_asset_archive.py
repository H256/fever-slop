import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from feverslop.tools.project_asset_archive import (
    ArchiveMember,
    build_archive_manifest,
    build_arg_parser,
    collect_archive_members,
    create_project_archive,
    main,
    resolve_available_zip_path,
    resolve_project_dir,
)


class ProjectAssetArchiveTests(unittest.TestCase):
    def test_resolve_project_dir_from_config_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            project.mkdir()
            config = project / "config.json"
            config.write_text(json.dumps({"input_audio": "input/song.mp3"}), encoding="utf-8")

            self.assertEqual(project, resolve_project_dir(project=config, project_dir=None))

    def test_resolve_project_dir_from_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            project.mkdir()

            self.assertEqual(project, resolve_project_dir(project=None, project_dir=project))

    def test_collect_archive_members_excludes_protected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            (project / "input").mkdir(parents=True)
            (project / "output" / "render" / "ltx").mkdir(parents=True)
            (project / "output" / "render" / "storyboard").mkdir(parents=True)
            (project / "archives").mkdir(parents=True)
            config = project / "config.json"
            config.write_text(json.dumps({"project_name": "demo"}), encoding="utf-8")
            (project / "input" / "song.mp3").write_bytes(b"audio")
            (project / "output" / "render" / "render_plan_song.json").write_text("[]", encoding="utf-8")
            (project / "output" / "render" / "ltx" / "scene_0001_raw.mp4").write_bytes(b"raw")
            (project / "output" / "render" / "ltx" / "demo_video_only.mp4").write_bytes(b"video only")
            (project / "output" / "render" / "ltx" / "demo.mp4").write_bytes(b"final video")
            (project / "output" / "render" / "storyboard" / "index.html").write_text(
                "<html></html>",
                encoding="utf-8",
            )
            (project / "output" / "render" / "storyboard" / "scene_0001.png").write_bytes(b"png")
            (project / "archives" / "old.zip").write_bytes(b"old archive")

            members = collect_archive_members(project, project_config=config, project_name="demo")

            self.assertEqual(
                [
                    ArchiveMember(project / "input" / "song.mp3", "input/song.mp3", 5),
                    ArchiveMember(
                        project / "output" / "render" / "ltx" / "demo_video_only.mp4",
                        "output/render/ltx/demo_video_only.mp4",
                        10,
                    ),
                    ArchiveMember(
                        project / "output" / "render" / "ltx" / "scene_0001_raw.mp4",
                        "output/render/ltx/scene_0001_raw.mp4",
                        3,
                    ),
                    ArchiveMember(
                        project / "output" / "render" / "render_plan_song.json",
                        "output/render/render_plan_song.json",
                        2,
                    ),
                ],
                members,
            )

    def test_collect_archive_members_excludes_sanitized_project_name_final_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            (project / "output" / "render" / "ltx").mkdir(parents=True)
            (project / "output" / "render" / "ltx" / "La_Entity.mp4").write_bytes(b"final")
            (project / "output" / "render" / "ltx" / "La_Entity_video_only.mp4").write_bytes(b"video only")

            members = collect_archive_members(project, project_name="La Entity")

            self.assertEqual(
                [
                    ArchiveMember(
                        project / "output" / "render" / "ltx" / "La_Entity_video_only.mp4",
                        "output/render/ltx/La_Entity_video_only.mp4",
                        10,
                    ),
                ],
                members,
            )

    def test_collect_archive_members_includes_canonical_render_artifacts_in_mixed_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            canonical = project / "output" / "render"
            legacy = canonical / "ltx"
            (canonical / "plans").mkdir(parents=True)
            (canonical / "scenes" / "scene_0001").mkdir(parents=True)
            (canonical / "final").mkdir(parents=True)
            legacy.mkdir()
            (canonical / "plans" / "base.json").write_text("[]", encoding="utf-8")
            (canonical / "scenes" / "scene_0001" / "manifest.json").write_text("{}", encoding="utf-8")
            (canonical / "scenes" / "scene_0001" / "final.mp4").write_bytes(b"scene")
            (canonical / "final" / "video_only.mp4").write_bytes(b"video")
            (canonical / "final" / "movie.mp4").write_bytes(b"movie")
            (legacy / "scene_0001.mp4").write_bytes(b"legacy")
            (legacy / "final_concat.mp4").write_bytes(b"protected")

            arcnames = [member.arcname for member in collect_archive_members(project)]

            self.assertEqual(
                [
                    "output/render/final/movie.mp4",
                    "output/render/final/video_only.mp4",
                    "output/render/ltx/scene_0001.mp4",
                    "output/render/plans/base.json",
                    "output/render/scenes/scene_0001/final.mp4",
                    "output/render/scenes/scene_0001/manifest.json",
                ],
                arcnames,
            )

    def test_build_archive_manifest_records_relative_files_and_total_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            project.mkdir()
            first = project / "input.mp3"
            second = project / "output.json"
            first.write_bytes(b"abc")
            second.write_text("{}", encoding="utf-8")
            members = [
                ArchiveMember(first, "input.mp3", 3),
                ArchiveMember(second, "output.json", 2),
            ]

            manifest = build_archive_manifest(project, members, created_at="2026-06-19T12:00:00")

            self.assertEqual("demo", manifest["project_name"])
            self.assertEqual("2026-06-19T12:00:00", manifest["created_at"])
            self.assertEqual(2, manifest["file_count"])
            self.assertEqual(5, manifest["total_bytes"])
            self.assertEqual(
                [
                    {"path": "input.mp3", "bytes": 3},
                    {"path": "output.json", "bytes": 2},
                ],
                manifest["files"],
            )

    def test_create_project_archive_writes_zip_with_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            (project / "input").mkdir(parents=True)
            config = project / "config.json"
            config.write_text(json.dumps({"project_name": "demo"}), encoding="utf-8")
            (project / "input" / "song.mp3").write_bytes(b"audio")
            (project / "output" / "render" / "ltx").mkdir(parents=True)
            (project / "output" / "render" / "storyboard").mkdir(parents=True)
            (project / "output" / "render" / "ltx" / "demo.mp4").write_bytes(b"final")
            (project / "output" / "render" / "ltx" / "demo_video_only.mp4").write_bytes(b"video only")
            (project / "output" / "render" / "storyboard" / "index.html").write_text(
                "<html></html>",
                encoding="utf-8",
            )
            output_zip = Path(temp_dir) / "demo.zip"

            created = create_project_archive(
                project_dir=project,
                project_config=config,
                project_name="demo",
                output_zip=output_zip,
                created_at="2026-06-19T12:00:00",
            )

            self.assertEqual(output_zip, created)
            with ZipFile(output_zip) as archive:
                self.assertEqual(
                    ["archive_manifest.json", "input/song.mp3", "output/render/ltx/demo_video_only.mp4"],
                    sorted(archive.namelist()),
                )
                manifest = json.loads(archive.read("archive_manifest.json").decode("utf-8"))
                self.assertEqual(2, manifest["file_count"])
                self.assertEqual("input/song.mp3", manifest["files"][0]["path"])

    def test_resolve_available_zip_path_adds_suffix_when_file_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_zip = Path(temp_dir) / "demo.zip"
            output_zip.write_bytes(b"existing")
            (Path(temp_dir) / "demo-2.zip").write_bytes(b"existing")

            available = resolve_available_zip_path(output_zip)

            self.assertEqual(Path(temp_dir) / "demo-3.zip", available)

    def test_arg_parser_accepts_project_and_dry_run(self):
        args = build_arg_parser().parse_args(["--project", "config.json", "--dry-run"])

        self.assertEqual("config.json", args.project)
        self.assertTrue(args.dry_run)
        self.assertIsNone(args.project_dir)

    def test_dry_run_prints_members_without_creating_zip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            project.mkdir()
            config = project / "config.json"
            config.write_text("{}", encoding="utf-8")
            output_zip = Path(temp_dir) / "demo.zip"

            exit_code = main(
                [
                    "--project",
                    str(config),
                    "--output",
                    str(output_zip),
                    "--dry-run",
                ],
            )

            self.assertEqual(0, exit_code)
            self.assertFalse(output_zip.exists())

    def _symlink_layout(self, temp_dir: str) -> tuple[Path, Path, Path]:
        """Project with a regular file plus a symlink pointing outside the project."""
        project = Path(temp_dir) / "demo"
        (project / "input").mkdir(parents=True)
        outside = Path(temp_dir) / "outside"
        outside.mkdir()
        target = outside / "big_model.bin"
        target.write_bytes(b"X" * 1000)
        (project / "input" / "song.mp3").write_bytes(b"audio")
        link = project / "input" / "linked_model.bin"
        link.symlink_to(target)
        return project, link, target

    def test_collect_archive_members_skips_file_symlinks_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project, link, _ = self._symlink_layout(temp_dir)

            members = collect_archive_members(project)

            self.assertEqual(["input/song.mp3"], [member.arcname for member in members])
            self.assertFalse(any(member.source == link for member in members))

    def test_collect_archive_members_follows_symlinks_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project, link, target = self._symlink_layout(temp_dir)

            members = collect_archive_members(project, follow_symlinks=True)

            arcnames = {member.arcname for member in members}
            self.assertIn("input/linked_model.bin", arcnames)
            linked = next(member for member in members if member.source == link)
            self.assertEqual(target.stat().st_size, linked.size)

    def test_collect_archive_members_skips_broken_symlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "demo"
            (project / "input").mkdir(parents=True)
            (project / "input" / "song.mp3").write_bytes(b"audio")
            (project / "input" / "broken.bin").symlink_to(Path(temp_dir) / "missing.bin")

            members = collect_archive_members(project, follow_symlinks=True)

            self.assertEqual(["input/song.mp3"], [member.arcname for member in members])

    def test_create_project_archive_excludes_symlink_targets_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project, link, _ = self._symlink_layout(temp_dir)
            output_zip = Path(temp_dir) / "demo.zip"

            created = create_project_archive(project_dir=project, output_zip=output_zip)

            with ZipFile(created) as archive:
                names = archive.namelist()
            self.assertNotIn("input/linked_model.bin", names)
            self.assertIn("input/song.mp3", names)

    def test_arg_parser_accepts_follow_symlinks(self):
        args = build_arg_parser().parse_args(["--follow-symlinks"])

        self.assertTrue(args.follow_symlinks)


if __name__ == "__main__":
    unittest.main()
