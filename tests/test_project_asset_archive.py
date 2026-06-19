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
                    )
                ],
                members,
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
                ]
            )

            self.assertEqual(0, exit_code)
            self.assertFalse(output_zip.exists())


if __name__ == "__main__":
    unittest.main()
