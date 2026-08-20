from __future__ import annotations

import json
import unittest
from pathlib import Path

from feverslop.adapters.project_reference_library import (
    ProjectReferenceLibrary,
    _next_revision,
    _safe_filename,
)
from feverslop.domain.reference_workspace import (
    ReferenceAsset,
    ReferenceKind,
    SceneReferenceAssignment,
)


class _FixtureProject:
    """Create a temporary project directory with movie reference fixtures."""

    def __init__(self, tmpdir: Path):
        self.root = tmpdir / "project"
        self.root.mkdir(parents=True, exist_ok=True)

    def add_manifest(self, actors=None, locations=None):
        manifest_dir = self.root / "movie" / "references"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if actors:
            data["actors"] = [{"id": a} for a in actors]
        if locations:
            data["locations"] = [{"id": loc} for loc in locations]
        manifest_path = manifest_dir / "manifest.json"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")

    def add_assignment_file(self, assignments=None, revision="r1"):
        path = self.root / "movie" / "reference_assignments.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "revision": revision,
            "assignments": list(assignments or []),
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def add_ref_image(self, name: str, content: bytes = b"fake-image"):
        ref_dir = self.root / "movie" / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / f"{name}.png").write_bytes(content)

    def add_config(self, props=None):
        data = {"input_audio": "song.mp3"}
        if props:
            data["global_props"] = [{"asset_id": p} for p in props]
        (self.root / "config.json").write_text(json.dumps(data), encoding="utf-8")


class NextRevisionTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual("r2", _next_revision("r1"))
        self.assertEqual("r10", _next_revision("r9"))

    def test_fallback(self):
        self.assertEqual("r1", _next_revision(""))
        self.assertEqual("r1", _next_revision("a"))


class SafeFilenameTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual("hero_portrait.png", _safe_filename("hero_portrait", ".png"))

    def test_special_chars(self):
        self.assertEqual("my_actor_v2__awesome.jpg", _safe_filename("my actor v2!!awesome", ".jpg"))

    def test_empty_id(self):
        self.assertEqual("imported.png", _safe_filename("", ".png"))


class ProjectReferenceLibraryLoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__file__).parent / "tmp_test_project_ref"
        self.fixture = _FixtureProject(self._tmp)

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self._tmp)
        except FileNotFoundError:
            pass

    def test_empty_project(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        snap = lib.load("proj")
        self.assertEqual((), snap.assets)
        self.assertEqual((), snap.assignments)
        self.assertEqual("r1", snap.revision)

    def test_load_assignments(self):
        self.fixture.add_assignment_file([
            {"scene_number": 3, "actor_ids": ["hero"], "location_ids": ["lab"]},
        ])
        lib = ProjectReferenceLibrary(self.fixture.root)
        snap = lib.load("proj")
        self.assertEqual(1, len(snap.assignments))
        self.assertEqual(3, snap.assignments[0].scene_number)

    def test_load_discovered_assets(self):
        self.fixture.add_ref_image("scene3_msr_sheet")
        lib = ProjectReferenceLibrary(self.fixture.root)
        snap = lib.load("proj")
        self.assertTrue(any(a.kind == ReferenceKind.MSR_SHEET for a in snap.assets))

    def test_discover_ingredients_sheet(self):
        self.fixture.add_ref_image("scene3_ingredients")
        lib = ProjectReferenceLibrary(self.fixture.root)
        snap = lib.load("proj")
        self.assertTrue(any(a.kind == ReferenceKind.INGREDIENTS_SHEET for a in snap.assets))

    def test_movie_bible_actor_ids(self):
        self.fixture.add_manifest(actors=["hero", "villain"])
        lib = ProjectReferenceLibrary(self.fixture.root)
        self.assertEqual(["hero", "villain"], lib.get_known_actor_ids("proj"))

    def test_movie_bible_location_ids(self):
        self.fixture.add_manifest(locations=["lab", "office"])
        lib = ProjectReferenceLibrary(self.fixture.root)
        self.assertEqual(["lab", "office"], lib.get_known_location_ids("proj"))

    def test_movie_bible_prop_ids(self):
        self.fixture.add_config(props=["guitar", "mic"])
        lib = ProjectReferenceLibrary(self.fixture.root)
        self.assertEqual(["guitar", "mic"], lib.get_known_prop_ids("proj"))

    def test_movie_bible_prop_ids_without_config(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        self.assertEqual([], lib.get_known_prop_ids("proj"))

    def test_movie_bible_prop_ids_invalid_config(self):
        (self.fixture.root / "config.json").write_text(
            json.dumps({"input_audio": "song.mp3", "global_props": "not-a-list"}),
            encoding="utf-8",
        )
        lib = ProjectReferenceLibrary(self.fixture.root)
        self.assertEqual([], lib.get_known_prop_ids("proj"))

    def test_max_scene_actors(self):
        lib = ProjectReferenceLibrary(self.fixture.root, max_scene_actors=3)
        self.assertEqual(3, lib.get_max_scene_actors("proj"))


class ProjectReferenceLibrarySaveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__file__).parent / "tmp_test_project_ref_save"
        self.fixture = _FixtureProject(self._tmp)

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self._tmp)
        except FileNotFoundError:
            pass

    def test_save_creates_file(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        revision = lib.save_assignments("proj", assignments, "r1")
        self.assertEqual("r2", revision)
        self.assertTrue((self.fixture.root / "movie" / "reference_assignments.json").exists())

    def test_save_revision_mismatch(self):
        self.fixture.add_assignment_file(revision="r2")
        lib = ProjectReferenceLibrary(self.fixture.root)
        assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",)),)
        with self.assertRaisesRegex(ValueError, "Revision mismatch"):
            lib.save_assignments("proj", assignments, "r1")

    def test_save_roundtrip(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        assignments = (SceneReferenceAssignment(scene_number=3, actor_ids=("hero",), location_ids=("lab",)),)
        lib.save_assignments("proj", assignments, "r1")
        snap = lib.load("proj")
        self.assertEqual(1, len(snap.assignments))
        self.assertEqual(("hero",), snap.assignments[0].actor_ids)
        self.assertEqual(("lab",), snap.assignments[0].location_ids)


class ProjectReferenceLibraryImportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__file__).parent / "tmp_test_project_ref_import"
        self.fixture = _FixtureProject(self._tmp)
        self.external_src = self._tmp / "external" / "hero_shot.png"
        self.external_src.parent.mkdir(parents=True, exist_ok=True)
        self.external_src.write_bytes(b"fake-image-123")

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self._tmp)
        except FileNotFoundError:
            pass

    def test_import_copies_file(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        asset = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR, label="Hero")
        result = lib.import_asset("proj", self.external_src, asset)
        dest = self.fixture.root / "movie" / "references" / "imported" / "hero.png"
        self.assertTrue(dest.exists())
        self.assertEqual("hero", result.id)
        self.assertIn("movie/references/imported/hero.png", result.path)

    def test_import_preserves_source(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        asset = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR)
        lib.import_asset("proj", self.external_src, asset)
        self.assertEqual(b"fake-image-123", self.external_src.read_bytes())

    def test_import_avoid_collision(self):
        existing = self.fixture.root / "movie" / "references" / "imported" / "hero.png"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"already-here")
        lib = ProjectReferenceLibrary(self.fixture.root)
        asset = ReferenceAsset(id="hero", kind=ReferenceKind.ACTOR)
        result = lib.import_asset("proj", self.external_src, asset)
        self.assertIn("hero_2", result.path)


class ProjectReferenceLibraryInvalidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(__file__).parent / "tmp_test_project_ref_inval"
        self.fixture = _FixtureProject(self._tmp)
        self.fixture.add_ref_image("scene3_msr_sheet")

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self._tmp)
        except FileNotFoundError:
            pass

    def test_invalidation_returns_msrs(self):
        lib = ProjectReferenceLibrary(self.fixture.root)
        result = lib.get_invalidated_artifacts("proj", changed_scenes=[3])
        self.assertTrue("msr_sheets" in result or "ingredients_sheets" in result)


if __name__ == "__main__":
    unittest.main()
