import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from feverslop.adapters.project_visual_consistency import (
    ProjectReferenceManifestAdapter,
    validate_project_scene_artifacts,
)
from feverslop.domain.visual_consistency import PreflightMode


class ProjectReferenceManifestAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.projects_root = Path(self.temporary_directory.name)
        self.project = self.projects_root / "demo"
        self.project.mkdir()
        self.adapter = ProjectReferenceManifestAdapter(self._project_root)

    def _project_root(self, project_id: str) -> Path:
        self.assertEqual("demo", project_id)
        return self.project

    def _asset(self, relative_path: str, contents: bytes = b"reference") -> Path:
        path = self.project / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def _legacy_manifest(self, kind: str, item_id: str, payload: dict) -> Path:
        path = (
            self.project
            / "output"
            / "references"
            / f"{kind}s"
            / item_id
            / "manifest.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"id": item_id, **payload}, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _movie_manifest(self, payload: dict) -> Path:
        path = self.project / "movie" / "references" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_legacy_defaults_and_looks_with_relative_domain_values(self):
        hero = self._asset("output/references/actors/hero/sheet.png", b"hero")
        winter = self._asset("output/references/actors/hero/winter.png", b"winter")
        stage = self._asset("output/references/locations/stage/sheet.png", b"stage")
        self._legacy_manifest(
            "actor",
            "hero",
            {
                "sheet_path": hero.relative_to(self.project).as_posix(),
                "visual_description": "  silver-haired singer " + "x" * 400,
                "looks": [
                    {
                        "id": "winter",
                        "sheet_path": winter.relative_to(self.project).as_posix(),
                        "visual_description": "heavy white coat",
                    }
                ],
            },
        )
        self._legacy_manifest(
            "location",
            "stage",
            {
                "msr_sheet_path": stage.relative_to(self.project).as_posix(),
                "visual_description": "mirrored black stage",
            },
        )

        snapshot = self.adapter.load("demo")

        default = snapshot.actors[("hero", "default")]
        self.assertEqual("identity-reference", default.asset_role)
        self.assertEqual(hashlib.sha256(b"hero").hexdigest(), default.asset_sha256)
        self.assertTrue(default.prompt_anchor.startswith("Reference actor `hero`"))
        self.assertLessEqual(len(default.prompt_anchor), 350)
        self.assertNotIn(str(self.project), default.prompt_anchor)
        self.assertEqual(
            hashlib.sha256(b"winter").hexdigest(),
            snapshot.actors[("hero", "winter")].asset_sha256,
        )
        self.assertIn(("stage", "default"), snapshot.locations)
        self.assertEqual(64, len(snapshot.revision))

    def test_loads_current_movie_manifest_and_revision_is_canonical_json(self):
        asset = self._asset("movie/references/actors/hero/msr.png", b"hero")
        decoded = {
            "project_type": "movie",
            "actors": [
                {
                    "id": "hero",
                    "msr_sheet_path": asset.relative_to(self.project).as_posix(),
                    "visual_description": "red leather jacket",
                }
            ],
            "locations": [],
        }
        self._movie_manifest(decoded)

        first = self.adapter.load("demo")
        self._movie_manifest(
            {
                "locations": [],
                "actors": decoded["actors"],
                "project_type": "movie",
            }
        )
        second = self.adapter.load("demo")

        self.assertIn(("hero", "default"), first.actors)
        self.assertEqual(first.revision, second.revision)

    def test_validates_contained_ingredients_artifact_paths(self):
        self._asset("sheets/scene.png")

        issues = validate_project_scene_artifacts(
            self.project,
            [{"scene": 1, "ingredients": {"sheet_path": "sheets/scene.png"}}],
            mode="ingredients",
            preflight_mode=PreflightMode.STRICT,
        )

        self.assertEqual((), issues)

    def test_reports_missing_and_external_ingredients_artifacts(self):
        outside = self.projects_root / "outside.png"
        outside.write_bytes(b"outside")

        missing = validate_project_scene_artifacts(
            self.project,
            [{"scene": 1, "ingredients": {"sheet_path": "missing.png"}}],
            mode="ingredients",
            preflight_mode=PreflightMode.STRICT,
        )
        external = validate_project_scene_artifacts(
            self.project,
            [{"scene": 2, "ingredients": {"sheet_path": str(outside)}}],
            mode="ingredients",
            preflight_mode=PreflightMode.STRICT,
        )

        self.assertEqual("missing_ingredients_sheet_file", missing[0].code)
        self.assertEqual("error", missing[0].severity)
        self.assertEqual("invalid_ingredients_sheet_path", external[0].code)

    def test_reports_missing_and_external_msr_artifacts(self):
        outside = self.projects_root / "outside.png"
        outside.write_bytes(b"outside")
        self._asset("refs/hero.png")

        issues = validate_project_scene_artifacts(
            self.project,
            [{
                "scene": 1,
                "references": {
                    "actor_msr_paths": ["refs/hero.png", "missing.png"],
                    "location_msr_path": str(outside),
                },
            }],
            mode="msr",
            preflight_mode=PreflightMode.WARN,
        )

        self.assertEqual(
            ["missing_msr_actor_file", "invalid_msr_location_path"],
            [issue.code for issue in issues],
        )
        self.assertTrue(all(issue.severity == "warning" for issue in issues))

    def test_rejects_symlink_escape_when_supported(self):
        outside = self.projects_root / "outside.png"
        outside.write_bytes(b"outside")
        link = self.project / "linked.png"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        issues = validate_project_scene_artifacts(
            self.project,
            [{"scene": 1, "ingredients": {"sheet_path": "linked.png"}}],
            mode="ingredients",
            preflight_mode=PreflightMode.STRICT,
        )

        self.assertEqual("invalid_ingredients_sheet_path", issues[0].code)

    def test_rejects_missing_external_and_directory_assets(self):
        outside = self.projects_root / "outside.png"
        outside.write_bytes(b"outside")
        directory = self.project / "directory"
        directory.mkdir()
        cases = (
            ("missing.png", "does not exist"),
            (outside, "outside project"),
            ("directory", "not a file"),
        )
        for index, (path, message) in enumerate(cases):
            with self.subTest(path=path):
                item_id = f"hero-{index}"
                self._legacy_manifest(
                    "actor",
                    item_id,
                    {
                        "sheet_path": str(path),
                        "visual_description": "red jacket",
                    },
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter.load("demo")
                (
                    self.project
                    / "output"
                    / "references"
                    / "actors"
                    / item_id
                    / "manifest.json"
                ).unlink()

    def test_rejects_empty_descriptions_and_missing_ids(self):
        asset = self._asset("output/references/actors/hero/sheet.png")
        cases = (
            (
                {"sheet_path": asset.relative_to(self.project).as_posix()},
                "visual description",
            ),
            (
                {
                    "id": " ",
                    "sheet_path": asset.relative_to(self.project).as_posix(),
                    "visual_description": "red jacket",
                },
                "id",
            ),
        )
        for index, (payload, message) in enumerate(cases):
            with self.subTest(payload=payload):
                manifest = self._legacy_manifest("actor", f"case-{index}", payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter.load("demo")
                manifest.unlink()

    def test_rejects_conflicting_duplicate_keys(self):
        first = self._asset("output/references/actors/hero/sheet.png", b"one")
        second = self._asset("movie/references/actors/hero/sheet.png", b"two")
        self._legacy_manifest(
            "actor",
            "hero",
            {
                "sheet_path": first.relative_to(self.project).as_posix(),
                "visual_description": "red jacket",
            },
        )
        self._movie_manifest(
            {
                "actors": [
                    {
                        "id": "hero",
                        "sheet_path": second.relative_to(self.project).as_posix(),
                        "visual_description": "blue jacket",
                    }
                ],
                "locations": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "Conflicting duplicate.*actor.*hero.*default"):
            self.adapter.load("demo")

    def test_rejects_identical_duplicate_keys(self):
        asset = self._asset("movie/references/actors/hero/sheet.png", b"hero")
        item = {
            "id": "hero",
            "sheet_path": asset.relative_to(self.project).as_posix(),
            "visual_description": "red jacket",
        }
        self._movie_manifest(
            {
                "actors": [item, dict(item)],
                "locations": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "Duplicate.*actor.*hero.*default"):
            self.adapter.load("demo")

    def test_rejects_malformed_manifest_scalar_types_with_field_context(self):
        asset = self._asset("output/references/actors/hero/sheet.png")
        cases = (
            (
                {"id": 7, "sheet_path": "unused.png", "visual_description": "hero"},
                "id.*string",
            ),
            (
                {"id": "hero", "sheet_path": {"path": "sheet.png"}, "visual_description": "hero"},
                "asset path.*string",
            ),
            (
                {
                    "id": "hero",
                    "sheet_path": asset.relative_to(self.project).as_posix(),
                    "visual_description": ["hero"],
                },
                "visual description.*string",
            ),
            (
                {
                    "id": "hero",
                    "looks": [
                        {
                            "id": 2,
                            "sheet_path": asset.relative_to(self.project).as_posix(),
                            "visual_description": "winter hero",
                        }
                    ],
                },
                "look id.*string",
            ),
        )
        for payload, message in cases:
            with self.subTest(payload=payload):
                path = self._movie_manifest({"actors": [payload], "locations": []})
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter.load("demo")
                path.unlink()

    def test_cross_layout_duplicate_diagnostic_names_both_manifest_sources(self):
        asset = self._asset("output/references/actors/hero/sheet.png", b"hero")
        legacy = self._legacy_manifest(
            "actor",
            "hero",
            {
                "sheet_path": asset.relative_to(self.project).as_posix(),
                "visual_description": "red jacket",
            },
        )
        movie = self._movie_manifest(
            {
                "actors": [
                    {
                        "id": "hero",
                        "sheet_path": asset.relative_to(self.project).as_posix(),
                        "visual_description": "red jacket",
                    }
                ],
                "locations": [],
            }
        )

        with self.assertRaises(ValueError) as caught:
            self.adapter.load("demo")

        message = str(caught.exception)
        self.assertIn(legacy.relative_to(self.project).as_posix(), message)
        self.assertIn(movie.relative_to(self.project).as_posix(), message)
        self.assertNotIn(str(self.project), message)

    def test_rejects_present_non_array_collections_even_when_falsy(self):
        cases = (
            ({"actors": {}, "locations": []}, "actors"),
            ({"actors": [], "locations": ""}, "locations"),
        )
        for payload, field in cases:
            with self.subTest(field=field):
                path = self._movie_manifest(payload)
                with self.assertRaisesRegex(ValueError, f"{field}.*JSON array"):
                    self.adapter.load("demo")
                path.unlink()

        asset = self._asset("output/references/actors/hero/sheet.png")
        manifest = self._legacy_manifest(
            "actor",
            "hero",
            {
                "sheet_path": asset.relative_to(self.project).as_posix(),
                "visual_description": "red jacket",
                "looks": "",
            },
        )
        with self.assertRaisesRegex(ValueError, "looks.*JSON array"):
            self.adapter.load("demo")
        manifest.unlink()


if __name__ == "__main__":
    unittest.main()
