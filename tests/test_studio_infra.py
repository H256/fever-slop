"""Tests for studio infrastructure fixes (Issue #280)."""

import json
import os
import tempfile
import unittest
from pathlib import Path


class TestArtifactCatalogOSError(unittest.TestCase):
    """INFRA-003: Handle OSError on path.stat()."""

    def test_catalog_snapshot_handles_broken_symlink(self):
        from feverslop.studio.artifact_catalog import ArtifactCatalog

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create a file
            (root / "config.json").write_text("{}")
            # Create a broken symlink
            target = root / "broken_link"
            try:
                target.symlink_to("nonexistent_file")
            except OSError as exc:
                if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                    self.skipTest("Windows symlink privilege is unavailable")
                raise
            # catalog_snapshot should not crash
            catalog = ArtifactCatalog(lambda _: root)
            snapshot = catalog.catalog_snapshot("")
            self.assertIn("artifacts", snapshot)
            self.assertIn("artifact_sizes", snapshot)


class TestRenderLogLinesNarrowExcept(unittest.TestCase):
    """INFRA-004: Narrow except Exception in render_log_lines."""

    def test_render_log_lines_fallback_on_value_error(self):
        from feverslop.studio.logging import render_log_lines

        class BadValue:
            def __rich_console__(self, console, options):
                raise ValueError("test error")

            def __str__(self):
                return "BadValue str fallback"

        result = render_log_lines(BadValue())
        self.assertIn("BadValue str fallback", result)

    def test_logger_is_configured(self):
        from feverslop.studio import logging as log_module

        self.assertTrue(hasattr(log_module, "_logger"))


class TestPipelineStateStoreTempFile(unittest.TestCase):
    """INFRA-005: Guard temp file unlink in finally."""

    def test_temp_file_guard_uses_except_not_finally(self):
        import inspect

        from feverslop.studio.pipeline_state_store import PipelineStateStore

        source = inspect.getsource(PipelineStateStore.record_pipeline_run)
        self.assertIn("except BaseException", source)
        self.assertNotIn("finally:", source)


class TestDownstreamStagesNaming(unittest.TestCase):
    """INFRA-009: Standardize kebab/snake naming in downstream stages."""

    def test_snake_and_kebab_aliases_present(self):
        from feverslop.studio.pipeline_state_store import (
            _MAIN_PIPELINE_DOWNSTREAM_STAGES,
        )

        # Verify that both conventions exist for known stages
        self.assertIn("full-pipeline", _MAIN_PIPELINE_DOWNSTREAM_STAGES)
        self.assertIn("full_pipeline", _MAIN_PIPELINE_DOWNSTREAM_STAGES)
        self.assertIn("main_pipeline", _MAIN_PIPELINE_DOWNSTREAM_STAGES)
        self.assertIn("main-pipeline", _MAIN_PIPELINE_DOWNSTREAM_STAGES)


class TestArtifactRevisionBomNormalization(unittest.TestCase):
    """INFRA-011: BOM-consistent artifact revision hashing."""

    def test_bom_and_nonbom_produce_same_revision(self):
        from feverslop.studio.projects import _artifact_revision

        content = b'{"key": "value"}\n'
        content_with_bom = b"\xef\xbb\xbf" + content
        self.assertEqual(
            _artifact_revision(content),
            _artifact_revision(content_with_bom),
        )


class TestReadJsonFileHandlesBadJson(unittest.TestCase):
    """INFRA-013: Handle JSON decode errors in _read_json_file."""

    def test_returns_default_on_corrupt_json(self):
        from feverslop.studio.projects import ProjectStore

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("this is not valid json {{{")
            f.flush()
            path = Path(f.name)

        try:
            with self.assertLogs("feverslop.studio.projects", level="WARNING") as logs:
                result = ProjectStore._read_json_file(path, default="fallback_value")
            self.assertEqual("fallback_value", result)
            self.assertIn(str(path), "\n".join(logs.output))
            self.assertIn("invalid JSON", "\n".join(logs.output))
        finally:
            path.unlink()

    def test_returns_parsed_data_on_valid_json(self):
        from feverslop.studio.projects import ProjectStore

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"hello": "world"}, f)
            f.flush()
            path = Path(f.name)

        try:
            result = ProjectStore._read_json_file(path, default="fallback_value")
            self.assertEqual({"hello": "world"}, result)
        finally:
            path.unlink()

    def test_returns_default_on_missing_file(self):
        from feverslop.studio.projects import ProjectStore

        result = ProjectStore._read_json_file(Path("/nonexistent/path"), default="fallback_value")
        self.assertEqual("fallback_value", result)


class MediaStoreAtomicWriteTests(unittest.TestCase):
    """INFRA-2001/INFRA-2002: media store writes route through atomic writers."""

    def test_write_media_data_url_writes_bytes_and_leaves_no_tmp(self):
        import base64

        from feverslop.studio.media_store import MediaStore

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "project"
            store = MediaStore(
                project_root=lambda project_id: root,
                resolve_project_path=lambda project_id, rel: root / rel,
                read_json_file=lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
            )
            data_url = "data:image/png;base64," + base64.b64encode(b"\x89PNGfake").decode()
            result = store.write_media_data_url("p1", "sub/a.png", data_url)

            self.assertEqual({"path": "sub/a.png"}, result)
            self.assertEqual(b"\x89PNGfake", (root / "sub" / "a.png").read_bytes())
            self.assertEqual([], [p for p in root.rglob("*") if p.name.endswith(".tmp")])

    def test_store_audio_upload_updates_config_and_leaves_no_tmp(self):
        import io

        from feverslop.studio.media_store import MediaStore

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "project"
            root.mkdir(parents=True)
            (root / "config.json").write_text(json.dumps({"existing": 1}), encoding="utf-8")
            store = MediaStore(
                project_root=lambda project_id: root,
                resolve_project_path=lambda project_id, rel: root / rel,
                read_json_file=lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
            )
            result = store.store_audio_upload("p1", "song.wav", "audio/wav", io.BytesIO(b"RIFF-bytes"))

            self.assertEqual({"path": "input/song.wav"}, result)
            self.assertTrue((root / "input" / "song.wav").is_file())
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual({"existing": 1, "input_audio": "input/song.wav"}, config)
            self.assertEqual([], [p for p in root.rglob("*") if p.name.endswith(".tmp")])


class EnsureMovieConfigAtomicTests(unittest.TestCase):
    """INFRA-2003: ensure_movie_config writes are atomic and non-destructive."""

    def test_ensure_movie_config_creates_config_and_leaves_no_tmp(self):
        from feverslop.studio.project_repository import ProjectRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "project"
            root.mkdir(parents=True)
            metadata = {"title": "T"}

            ProjectRepository.ensure_movie_config(root, metadata)

            self.assertTrue((root / "config.json").is_file())
            self.assertIsInstance(json.loads((root / "config.json").read_text(encoding="utf-8")), dict)

            (root / "config.json").write_text(json.dumps({"keep": True}), encoding="utf-8")
            ProjectRepository.ensure_movie_config(root, metadata)

            self.assertEqual({"keep": True}, json.loads((root / "config.json").read_text(encoding="utf-8")))
            self.assertEqual([], [p for p in root.rglob("*") if p.name.endswith(".tmp")])


if __name__ == "__main__":
    unittest.main()
