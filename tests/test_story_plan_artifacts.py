"""Tests for the story plan artifact manifest contract (issue #1384)."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from feverslop.domain.artifact_hash import sha256_bytes
from feverslop.domain.story_plan import StoryPlan
from feverslop.domain.story_plan_artifacts import (
    ArtifactClass,
    ArtifactDependency,
    RegenerationPolicy,
    StoryPlanArtifactManifest,
    manifest_is_stale,
    read_manifest,
    read_story_plan,
    write_manifest,
    write_story_plan,
)
from feverslop.errors import FeverSlopDataError


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _make_manifest(**overrides) -> StoryPlanArtifactManifest:
    payload = {
        "artifact_class": "authoritative",
        "regeneration_policy": "never",
        "input_fingerprint": _sha("input"),
        "dependencies": [{"path": "input/stage1.json", "sha256": _sha("stage1")}],
        "plan_fingerprint": _sha("plan"),
    }
    payload.update(overrides)
    return StoryPlanArtifactManifest.model_validate(payload, strict=False)


def _make_plan() -> StoryPlan:
    return StoryPlan.model_validate(
        {
            "mode": "music_video",
            "source_fingerprint": _sha("source"),
            "provenance": {"producer": "test-planner"},
            "segments": [
                {
                    "id": "brief-1",
                    "target": "seg-1",
                    "audio_ref": {"segment_id": "seg-1", "fingerprint": _sha("seg-1")},
                }
            ],
        },
        strict=False,
    )


class ArtifactManifestModelTests(unittest.TestCase):
    def test_manifest_construction(self):
        manifest = _make_manifest()
        self.assertEqual(manifest.artifact_class, ArtifactClass.authoritative)
        self.assertEqual(manifest.regeneration_policy, RegenerationPolicy.never)
        self.assertEqual(len(manifest.dependencies), 1)

    def test_artifact_class_enum_values(self):
        self.assertEqual(
            [member.value for member in ArtifactClass],
            ["authoritative", "resume_cache", "review_export"],
        )

    def test_regeneration_policy_enum_values(self):
        self.assertEqual(
            [member.value for member in RegenerationPolicy],
            ["never", "on_input_change", "on_request"],
        )

    def test_rejects_unsupported_schema_version(self):
        with self.assertRaises(ValidationError):
            _make_manifest(schema_version="story-plan-artifact/v999")

    def test_rejects_non_sha256_input_fingerprint(self):
        with self.assertRaises(ValidationError):
            _make_manifest(input_fingerprint="xyz")

    def test_rejects_non_sha256_dependency_digest(self):
        with self.assertRaises(ValidationError):
            _make_manifest(dependencies=[{"path": "input/stage1.json", "sha256": "xyz"}])

    def test_rejects_duplicate_dependency_paths(self):
        with self.assertRaises(ValidationError):
            _make_manifest(
                dependencies=[
                    {"path": "input/a.json", "sha256": _sha("a")},
                    {"path": "input/a.json", "sha256": _sha("b")},
                ]
            )

    def test_rejects_extra_fields(self):
        with self.assertRaises(ValidationError):
            _make_manifest(surprise="value")


class ManifestIoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_write_then_read_round_trip(self):
        manifest = _make_manifest()
        path = self.root / "manifest.json"
        written = write_manifest(path, manifest)
        self.assertEqual(written, path)
        self.assertEqual(read_manifest(path), manifest)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_write_creates_parent_dirs(self):
        manifest = _make_manifest()
        path = self.root / "nested" / "deep" / "manifest.json"
        write_manifest(path, manifest)
        self.assertTrue(path.is_file())
        self.assertEqual(read_manifest(path), manifest)

    def test_read_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_manifest(self.root / "missing.json")

    def test_read_rejects_malformed_json(self):
        path = self.root / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(FeverSlopDataError):
            read_manifest(path)

    def test_read_rejects_non_object_json(self):
        path = self.root / "list.json"
        path.write_text("[1,2]", encoding="utf-8")
        with self.assertRaises(FeverSlopDataError):
            read_manifest(path)

    def test_read_rejects_unsupported_schema_version(self):
        payload = _make_manifest().model_dump(mode="json")
        payload["schema_version"] = "story-plan-artifact/v999"
        path = self.root / "version.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(FeverSlopDataError):
            read_manifest(path)

    def test_read_rejects_malformed_manifest(self):
        payload = _make_manifest().model_dump(mode="json")
        payload["input_fingerprint"] = "xyz"
        path = self.root / "malformed.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(FeverSlopDataError):
            read_manifest(path)

    def test_write_read_story_plan_round_trip(self):
        plan = _make_plan()
        path = self.root / "story_plan.json"
        write_story_plan(path, plan)
        self.assertEqual(read_story_plan(path), plan)

    def test_read_story_plan_rejects_unsupported_schema_version(self):
        payload = _make_plan().model_dump(mode="json")
        payload["schema_version"] = "story-plan/v999"
        path = self.root / "story_plan.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(FeverSlopDataError):
            read_story_plan(path)


class StaleManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dep = self.root / "dep.json"
        self.dep_content = b'{"a": 1}'
        self.dep.write_bytes(self.dep_content)
        self.manifest = StoryPlanArtifactManifest(
            artifact_class=ArtifactClass.authoritative,
            regeneration_policy=RegenerationPolicy.on_input_change,
            input_fingerprint=_sha("input"),
            dependencies=[
                ArtifactDependency(path="dep.json", sha256=sha256_bytes(self.dep_content))
            ],
        )

    def test_fresh_manifest_is_not_stale(self):
        self.assertFalse(
            manifest_is_stale(
                self.manifest,
                input_fingerprint=_sha("input"),
                dependency_digests={"dep.json": sha256_bytes(self.dep_content)},
            )
        )

    def test_changed_input_fingerprint_is_stale(self):
        self.assertTrue(
            manifest_is_stale(
                self.manifest,
                input_fingerprint=_sha("other"),
                dependency_digests={"dep.json": sha256_bytes(self.dep_content)},
            )
        )

    def test_changed_dependency_digest_is_stale(self):
        new_content = b'{"a": 2}'
        self.dep.write_bytes(new_content)
        self.assertTrue(
            manifest_is_stale(
                self.manifest,
                input_fingerprint=_sha("input"),
                dependency_digests={"dep.json": sha256_bytes(new_content)},
            )
        )

    def test_missing_dependency_is_stale(self):
        self.assertTrue(
            manifest_is_stale(
                self.manifest,
                input_fingerprint=_sha("input"),
                dependency_digests={},
            )
        )


if __name__ == "__main__":
    unittest.main()
