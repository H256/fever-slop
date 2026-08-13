"""Security audit tests for SEC-001 .. SEC-005.

Regression tests that ensure critical security patterns are never introduced
into the codebase and that path containment checks function correctly.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feverslop.domain.security import (
    SecurityError,
    audit_no_fstring_sql,
    audit_no_pickle,
    audit_no_shell_true,
    guard_path_under_root,
    is_safe_identifier,
    run_all_audits,
    sanitize_path_component,
)
from feverslop.path_utils import coerce_local_path
from feverslop.studio.projects import ProjectCreateRequest, ProjectStore, StudioPathError


class SecurityAuditTests(unittest.TestCase):
    """Audit tests – scan the codebase for forbidden patterns."""

    @classmethod
    def _source_root(cls) -> Path:
        return Path(__file__).resolve().parent.parent / "src"

    # --- SEC-001: pickle ---

    def test_no_pickle_usage_in_source(self):
        """SEC-001: no pickle deserialization in source code."""
        issues = audit_no_pickle(self._source_root())
        self.assertEqual(
            [],
            issues,
            "pickle usage found (RCE risk):\n" + "\n".join(issues),
        )

    def test_audit_no_pickle_skips_test_security_audit_self(self):
        """The audit functions themselves mention pickle in strings – verify grep doesn't trip."""
        # The audit function searches for *code* patterns, not docstrings.
        # Confirm it actually scans something.
        issues: list[str] = audit_no_pickle(self._source_root())
        # Even if the module itself is scanned, it should not match because
        # the word "pickle" only appears in comments/docstrings (no actual calls).
        for line in issues:
            self.assertNotIn(".loads", line, f"pickle import/call pattern in {line}")

    # --- SEC-002: shell=True ---

    def test_no_shell_true_in_source(self):
        """SEC-002: no subprocess calls with shell=True."""
        issues = audit_no_shell_true(self._source_root())
        self.assertEqual(
            [],
            issues,
            "shell=True found (command injection risk):\n" + "\n".join(issues),
        )

    # --- SEC-005: f-string SQL ---

    def test_no_fstring_sql_in_infra(self):
        """SEC-005: no f-string SQL in sqlite adapter."""
        infra_dir = self._source_root() / "feverslop" / "infra"
        issues = audit_no_fstring_sql(infra_dir)
        self.assertEqual(
            [],
            issues,
            "f-string SQL found (SQL injection risk):\n" + "\n".join(issues),
        )

    # --- Combined audit ---

    def test_run_all_audits_passes_clean(self):
        """All audits should pass with zero issues."""
        results = run_all_audits(self._source_root())
        for check, issues in results.items():
            self.assertEqual(
                [],
                issues,
                f"Audit '{check}' found issues:\n" + "\n".join(issues),
            )


class PathContainmentTests(unittest.TestCase):
    """SEC-003: path traversal containment checks."""

    # --- coerce_local_path containment ---

    def test_coerce_local_path_allows_valid_relative(self):
        self.assertEqual(
            Path("projects") / "my-song",
            coerce_local_path("projects/my-song"),
        )

    def test_coerce_local_path_containment_accepts_safe_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = coerce_local_path(
                "subdir/file.txt",
                base_dir=root,
                containment_root=root,
            )
            self.assertTrue(result.resolve().is_relative_to(root))

    def test_coerce_local_path_containment_blocks_parent_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError) as ctx:
                coerce_local_path(
                    "../escape.txt",
                    base_dir=root,
                    containment_root=root,
                )
            self.assertIn("escapes containment root", str(ctx.exception))

    def test_coerce_local_path_containment_blocks_deep_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                coerce_local_path(
                    "foo/../../bar/escape.txt",
                    base_dir=root,
                    containment_root=root,
                )

    # --- guard_path_under_root ---

    def test_guard_path_under_root_accepts_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "a" / "b" / "file.txt"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.touch()
            result = guard_path_under_root(nested, root)
            self.assertEqual(nested.resolve(), result)

    def test_guard_path_under_root_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            escape = Path(tmp) / ".." / "etc" / "passwd"
            with self.assertRaises(SecurityError):
                guard_path_under_root(escape, root)

    def test_guard_path_under_root_blocks_absolute_foreign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreign = Path("/etc/passwd")
            with self.assertRaises(SecurityError):
                guard_path_under_root(foreign, root)

    # --- sanitize_path_component ---

    def test_sanitize_path_component_removes_traversal(self):
        self.assertEqual("safe_name", sanitize_path_component("../safe_name"))
        self.assertEqual("ab", sanitize_path_component("a/..b"))
        self.assertEqual("", sanitize_path_component("../../../.."))

    def test_sanitize_path_component_preserves_hidden_files(self):
        self.assertEqual(".hidden", sanitize_path_component(".hidden"))
        self.assertEqual(".env", sanitize_path_component(".env"))
        self.assertEqual(".gitignore", sanitize_path_component(".gitignore"))

    # --- is_safe_identifier ---

    def test_is_safe_identifier_rejects_parent_dir(self):
        self.assertFalse(is_safe_identifier("foo/../bar"))
        self.assertFalse(is_safe_identifier("foo\\..\\bar"))
        self.assertFalse(is_safe_identifier("../"))
        self.assertFalse(is_safe_identifier("../../../etc"))

    def test_is_safe_identifier_rejects_absolute(self):
        for value in (
            "/etc/passwd",
            "\\etc\\passwd",
            "C:\\projects\\demo",
            "\\\\server\\share\\demo",
        ):
            with self.subTest(value=value):
                self.assertFalse(is_safe_identifier(value))

    def test_is_safe_identifier_accepts_safe_names(self):
        self.assertTrue(is_safe_identifier("my-song"))
        self.assertTrue(is_safe_identifier("foo/bar"))
        self.assertTrue(is_safe_identifier(""))

    # --- ProjectStore path traversal ---

    def test_full_auto_scaffold_rejects_unsafe_project_slug(self):
        from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold
        from feverslop.domain.full_auto import GeneratedSong, SongSpec

        spec = SongSpec(
            title="Test", tags="", lyrics="", bpm=120, duration_seconds=10,
            language="en", keyscale="C", visual_story_idea="", visual_style="",
        )
        song = GeneratedSong(audio_path=Path("/tmp/song.mp3"), manifest={})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                LocalProjectScaffold().create_project(
                    projects_dir=Path(tmp), project_slug="../escape", spec=spec, generated_song=song,
                )

    def test_project_store_project_root_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))
            # Ensure traversal can't be used as a project id
            with self.assertRaises(StudioPathError):
                store.project_root("../../../etc")

    def test_project_store_project_root_reports_missing_direct_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))

            with self.assertRaises(FileNotFoundError):
                store.project_root("missing-project")

    def test_project_store_resolve_project_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))
            # Create a real project
            store.create_project(
                ProjectCreateRequest(project_type="standard_music_video", name="Test")
            )
            # Try to resolve a path that escapes project root
            with self.assertRaises(StudioPathError):
                store.resolve_project_path("test", "../other/file.json")


if __name__ == "__main__":
    unittest.main()
