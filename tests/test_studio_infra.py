"""Tests for studio infrastructure fixes (Issue #280)."""

import json
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
            target.symlink_to("nonexistent_file")
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
        from feverslop.studio.pipeline_state_store import PipelineStateStore

        import inspect

        source = inspect.getsource(PipelineStateStore.record_pipeline_run)
        self.assertIn("except BaseException", source)
        self.assertNotIn("finally:", source)


class TestDownstreamStagesNaming(unittest.TestCase):
    """INFRA-009: Standardize kebab/snake naming in downstream stages."""

    def test_snake_and_kebab_aliases_present(self):
        from feverslop.studio.pipeline_state_store import _MAIN_PIPELINE_DOWNSTREAM_STAGES

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


if __name__ == "__main__":
    unittest.main()
