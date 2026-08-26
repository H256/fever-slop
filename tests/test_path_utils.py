import tempfile
import unittest
import warnings
from pathlib import Path

from feverslop.path_utils import WORKFLOW_PATH_ALIASES, coerce_local_path, resolve_workflow_reference


class PathUtilsTests(unittest.TestCase):
    def test_windows_relative_path_is_coerced_to_current_platform_path(self):
        self.assertEqual(Path("app_config.json"), coerce_local_path(".\\app_config.json"))

    def test_windows_relative_path_is_resolved_against_base_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            path = coerce_local_path("input\\song.mp3", base_dir=base_dir)

        self.assertEqual(base_dir / "input" / "song.mp3", path)

    def test_posix_relative_path_remains_valid(self):
        self.assertEqual(
            Path("workflows") / "video_ltxv_i2v_v1.json",
            coerce_local_path("workflows/video_ltxv_i2v_v1.json"),
        )

    def test_absolute_path_stays_absolute(self):
        absolute = Path.cwd().resolve() / "app_config.json"

        self.assertEqual(absolute, coerce_local_path(absolute))

    def test_known_legacy_workflow_path_warns_and_resolves_exactly(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_workflow_reference(
                r".\workflows\image_t2i_startframe_krea_v1.json",
            )

        self.assertEqual("workflows/image/image-model/image_t2i_startframe_krea_v1.json", resolved)
        self.assertEqual(1, len(caught))
        self.assertIs(caught[0].category, DeprecationWarning)
        self.assertIn("deprecated", str(caught[0].message))

    def test_aliases_do_not_guess_by_basename(self):
        value = "image_t2i_startframe_krea_v1.json"

        self.assertEqual(value, resolve_workflow_reference(value))

    def test_maintained_sources_do_not_add_flat_workflow_literals(self):
        allowed = {
            Path("src/feverslop/path_utils.py"),
            Path("documentation/workflow-path-map.md"),
        }
        candidates = [
            path
            for root in (Path("src"), Path("tests"), Path("documentation"))
            for path in root.rglob("*")
            if path.suffix in {".md", ".py", ".json"} and path not in allowed
        ]
        violations = [
            f"{path}:{legacy}"
            for path in candidates
            for legacy in WORKFLOW_PATH_ALIASES
            if legacy in path.read_text(encoding="utf-8")
        ]

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
