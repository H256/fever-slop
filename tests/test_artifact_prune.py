"""Tests for the manifest-governed artifact prune application service (issue #1388).

Covers ``scan_project`` (machine-readable report, no writes, per-candidate
manifest problems as ineligibility reasons), ``create_prune_archive``
(zip + ``archive_manifest.json``, collision-safe paths, failure cleanup),
and ``apply_prune`` (archive-first, delete exactly the archived eligible
candidates).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from feverslop.application.artifact_prune import (
    PRUNE_REPORT_SCHEMA,
    apply_prune,
    create_prune_archive,
    scan_project,
)
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies, project_effective_plan
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.tools.project_asset_archive import ArchiveMember


class _FailingZip:
    """ZipFile stand-in that fails on the first member write (after the manifest)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> "_FailingZip":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def writestr(self, name: str, data: object) -> None:
        del data
        if name != "archive_manifest.json":
            raise OSError("injected zip write failure")

    def write(self, source: object, arcname: str) -> None:
        del source, arcname
        raise OSError("injected zip write failure")


class ArtifactPruneApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()
        self.layout = SceneArtifactLayout(self.project)

    def _scene_entry(self, scene_number: int) -> dict:
        canonical = build_canonical_scene(
            segment_id=f"segment-{scene_number}",
            generated_roles={PromptRole.Z_IMAGE: f"generated prompt {scene_number}"},
        )
        return {
            "scene": scene_number,
            "canonical": canonical,
            "z_image": {"prompt": f"generated prompt {scene_number}"},
        }

    def _write_base(self, scene_numbers: tuple[int, ...]) -> dict[int, dict]:
        scenes = {number: self._scene_entry(number) for number in scene_numbers}
        self.layout.plans_dir.mkdir(parents=True, exist_ok=True)
        self.layout.base_plan.write_text(
            json.dumps([scenes[number] for number in scene_numbers]), encoding="utf-8"
        )
        return scenes

    def _write_scene_manifest(
        self,
        scene_number: int,
        scene: dict,
        *,
        first_frame_path: Path | None = None,
    ) -> None:
        scene_dir = self.layout.scene_dir(scene_number)
        scene_dir.mkdir(parents=True, exist_ok=True)
        workflow = self.layout.scene_workflow(scene_number)
        workflow.write_text("{}", encoding="utf-8")
        template = self.project / "workflow-template.json"
        template.write_text("{}", encoding="utf-8")
        projected = project_effective_plan([scene])[0]
        dependencies = CanonicalSceneDependencies.from_dict(
            projected["canonical_projection"]["dependencies"]
        )
        SceneWorkflowManifest.create(
            project_dir=self.project,
            scene=scene_number,
            pipeline="ltx_i2v",
            workflow_path=workflow,
            template_path=template,
            render_plan_path=self.layout.base_plan,
            assets=[],
            seed=1,
            fps=24,
            frame_count=25,
            width=1280,
            height=704,
            canonical_dependencies=dependencies,
            first_frame_path=first_frame_path,
        ).write(self.layout.scene_manifest(scene_number))

    def _write_complete_scene(self, scene_number: int, scene: dict) -> None:
        self._write_scene_manifest(scene_number, scene)
        (self.layout.scene_dir(scene_number) / "raw.mp4").write_bytes(b"raw-video-bytes")
        self.layout.scene_final_video(scene_number).write_bytes(b"final-video-bytes")

    def _write_blocked_scene(self, scene_number: int) -> None:
        scene_dir = self.layout.scene_dir(scene_number)
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "workflow.json").write_text("{}", encoding="utf-8")
        (scene_dir / "raw.mp4").write_bytes(b"raw-video-bytes")

    def _write_protected_files(self) -> None:
        (self.project / "config.json").write_text(
            json.dumps({"upscale": {"enabled": False}}), encoding="utf-8"
        )
        references = self.project / "output" / "references"
        (references / "actors").mkdir(parents=True)
        (references / "actors" / "a.png").write_bytes(b"reference-bytes")
        (references / "manifest.json").write_text(json.dumps({"assets": []}), encoding="utf-8")
        (self.project / "output" / "review.md").write_text("# review", encoding="utf-8")
        final_dir = self.project / "output" / "render" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "video_only.mp4").write_bytes(b"assembled-final")
        prompts = self.project / "output" / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "story_plan_song-a.json").write_text(json.dumps({"cast": []}), encoding="utf-8")

    def _build_two_scene_project(self) -> None:
        scenes = self._write_base((1, 2))
        self._write_complete_scene(1, scenes[1])
        self._write_blocked_scene(2)
        self._write_protected_files()
        derived = project_effective_plan([scenes[1]], [scenes[1]])
        self.layout.references_plan.write_text(json.dumps(derived), encoding="utf-8")

    def _tree_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.project.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.project).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _candidate_entry(self, report, relative_path: str) -> dict:
        return next(entry for entry in report.candidates if entry["path"] == relative_path)


class ScanProjectTests(ArtifactPruneApplicationTests):
    def test_scan_produces_machine_readable_report(self):
        self._build_two_scene_project()

        report = scan_project(self.project)

        payload = report.to_dict()
        self.assertEqual(PRUNE_REPORT_SCHEMA, payload["schema_version"])
        self.assertEqual("safe", payload["mode"])
        self.assertIsNone(payload["archive_path"])
        self.assertEqual([], payload["errors"])
        self.assertEqual([], payload["deleted"])
        self.assertTrue(payload["created_at"])
        # Candidate set: the eligible derived plan + both scenes' workflow/raw.
        self.assertEqual(
            {
                "output/render/plans/references.json",
                "output/render/scenes/scene_0001/workflow.json",
                "output/render/scenes/scene_0001/raw.mp4",
                "output/render/scenes/scene_0002/workflow.json",
                "output/render/scenes/scene_0002/raw.mp4",
            },
            {entry["path"] for entry in report.candidates},
        )
        eligible = {entry["path"] for entry in report.candidates if entry["eligible"]}
        self.assertEqual(
            {
                "output/render/plans/references.json",
                "output/render/scenes/scene_0001/workflow.json",
                "output/render/scenes/scene_0001/raw.mp4",
            },
            eligible,
        )
        # Sizes are real file sizes; blocked scene reports every blocker.
        workflow_entry = self._candidate_entry(report, "output/render/scenes/scene_0001/workflow.json")
        self.assertEqual(len(b"{}"), workflow_entry["size_bytes"])
        self.assertEqual("resume_cache", workflow_entry["class"])
        self.assertEqual([], workflow_entry["reasons"])
        blocked_entry = self._candidate_entry(report, "output/render/scenes/scene_0002/workflow.json")
        self.assertFalse(blocked_entry["eligible"])
        self.assertEqual(["manifest_missing", "no_final_successor"], blocked_entry["reasons"])
        # Protected entries carry class + reason. The scan enumerates the
        # layout-anchored set only (config, base plan, story plans, scene
        # manifests/finals); other protected classes (references, review
        # exports, assembled finals) are never candidates and are left
        # untouched.
        protected_paths = {entry["path"]: entry for entry in report.protected}
        self.assertEqual("authoritative", protected_paths["config.json"]["class"])
        self.assertNotEqual("", protected_paths["config.json"]["reason"])
        self.assertEqual(
            {
                "config.json",
                "output/render/plans/base.json",
                "output/prompts/story_plan_song-a.json",
                "output/render/scenes/scene_0001/manifest.json",
                "output/render/scenes/scene_0001/final.mp4",
            },
            set(protected_paths),
        )
        # The JSON form round-trips the same payload.
        self.assertEqual(payload, json.loads(report.to_json()))

    def test_scan_performs_no_writes(self):
        self._build_two_scene_project()
        before = self._tree_hash()

        scan_project(self.project)

        self.assertEqual(before, self._tree_hash())

    def test_scan_missing_project_raises(self):
        with self.assertRaises(NotADirectoryError):
            scan_project(self.project / "missing")

    def test_scan_records_corrupt_scene_manifest_as_ineligibility(self):
        scenes = self._write_base((1,))
        self._write_complete_scene(1, scenes[1])
        self._write_protected_files()
        self.layout.scene_manifest(1).write_text("{", encoding="utf-8")

        report = scan_project(self.project)

        entry = self._candidate_entry(report, "output/render/scenes/scene_0001/workflow.json")
        self.assertFalse(entry["eligible"])
        self.assertIn("manifest_invalid", entry["reasons"])
        self.assertEqual([], report.errors)

    def test_scene_referenced_by_unfinished_scene_blocks_deletion(self):
        scenes = self._write_base((1, 2))
        self._write_complete_scene(1, scenes[1])
        self._write_blocked_scene(2)
        self._write_protected_files()
        # Scene 2's manifest references scene 1's raw clip; scene 2's final
        # chain is incomplete, so the referenced candidate is blocked.
        self._write_scene_manifest(
            2,
            scenes[2],
            first_frame_path=self.layout.scene_raw_video(1),
        )

        report = scan_project(self.project)

        raw_entry = self._candidate_entry(report, "output/render/scenes/scene_0001/raw.mp4")
        self.assertFalse(raw_entry["eligible"])
        self.assertIn("referenced_by_unfinished", raw_entry["reasons"])
        # The unreferenced scene 1 workflow stays eligible.
        workflow_entry = self._candidate_entry(report, "output/render/scenes/scene_0001/workflow.json")
        self.assertTrue(workflow_entry["eligible"])
        self.assertEqual([], workflow_entry["reasons"])


class CreatePruneArchiveTests(ArtifactPruneApplicationTests):
    def _members(self) -> list[ArchiveMember]:
        workflow = self.layout.scene_workflow(1)
        raw = self.layout.scene_raw_video(1)
        return [
            ArchiveMember(
                source=workflow,
                arcname=workflow.relative_to(self.project).as_posix(),
                size=workflow.stat().st_size,
            ),
            ArchiveMember(
                source=raw,
                arcname=raw.relative_to(self.project).as_posix(),
                size=raw.stat().st_size,
            ),
        ]

    def test_create_prune_archive_writes_zip_with_manifest(self):
        self._build_two_scene_project()
        members = self._members()
        output = Path(self.temp.name) / "archive.zip"

        created = create_prune_archive(self.project, members, output)

        self.assertEqual(output, created)
        self.assertTrue(created.is_file())
        with zipfile.ZipFile(created) as archive:
            self.assertEqual(
                {
                    "archive_manifest.json",
                    "output/render/scenes/scene_0001/workflow.json",
                    "output/render/scenes/scene_0001/raw.mp4",
                },
                set(archive.namelist()),
            )
            manifest = json.loads(archive.read("archive_manifest.json"))
            self.assertEqual(2, manifest["file_count"])
            self.assertEqual(
                members[0].size + members[1].size, manifest["total_bytes"]
            )
            self.assertEqual(
                [
                    {"path": "output/render/scenes/scene_0001/workflow.json", "bytes": members[0].size},
                    {"path": "output/render/scenes/scene_0001/raw.mp4", "bytes": members[1].size},
                ],
                manifest["files"],
            )
            self.assertEqual(
                members[0].source.read_bytes(),
                archive.read("output/render/scenes/scene_0001/workflow.json"),
            )

    def test_create_prune_archive_is_collision_safe(self):
        self._build_two_scene_project()
        members = self._members()
        output = Path(self.temp.name) / "archive.zip"

        first = create_prune_archive(self.project, members, output)
        second = create_prune_archive(self.project, members, output)

        self.assertEqual(output, first)
        self.assertNotEqual(first, second)
        self.assertEqual("archive-2.zip", second.name)
        self.assertTrue(second.is_file())

    def test_create_prune_archive_failure_removes_partial_zip(self):
        self._build_two_scene_project()
        members = self._members()
        output = Path(self.temp.name) / "fail.zip"

        with patch("feverslop.application.artifact_prune.ZipFile", _FailingZip):
            with self.assertRaises(OSError):
                create_prune_archive(self.project, members, output)

        self.assertFalse(output.exists())
        self.assertTrue(self.layout.scene_workflow(1).is_file())
        self.assertTrue(self.layout.scene_raw_video(1).is_file())


class ApplyPruneTests(ArtifactPruneApplicationTests):
    def test_apply_prune_deletes_exactly_archived_eligible(self):
        self._build_two_scene_project()
        report = scan_project(self.project)
        archive = Path(self.temp.name) / "apply.zip"

        final = apply_prune(self.project, archive, report)

        self.assertEqual("apply", final.mode)
        self.assertEqual(str(archive), final.archive_path)
        # Eligible candidates are gone.
        self.assertFalse(self.layout.scene_workflow(1).exists())
        self.assertFalse(self.layout.scene_raw_video(1).exists())
        self.assertFalse(self.layout.references_plan.exists())
        # Ineligible candidates and protected files survive.
        self.assertTrue(self.layout.scene_workflow(2).is_file())
        self.assertTrue(self.layout.scene_raw_video(2).is_file())
        self.assertTrue(self.layout.base_plan.is_file())
        self.assertTrue(self.layout.scene_manifest(1).is_file())
        self.assertTrue(self.layout.scene_final_video(1).is_file())
        self.assertTrue((self.project / "config.json").is_file())
        self.assertTrue((self.project / "output" / "review.md").is_file())
        self.assertEqual(
            {
                "output/render/plans/references.json",
                "output/render/scenes/scene_0001/workflow.json",
                "output/render/scenes/scene_0001/raw.mp4",
            },
            {entry["path"] for entry in final.deleted},
        )
        # Deleted entries carry the pre-delete sizes from the scan report.
        expected_sizes = {
            entry["path"]: entry["size_bytes"] for entry in report.candidates if entry["eligible"]
        }
        self.assertEqual(
            expected_sizes,
            {entry["path"]: entry["size_bytes"] for entry in final.deleted},
        )
        # The archive contains exactly the deleted files plus the manifest.
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as zip_file:
            self.assertEqual(
                {
                    "archive_manifest.json",
                    "output/render/plans/references.json",
                    "output/render/scenes/scene_0001/workflow.json",
                    "output/render/scenes/scene_0001/raw.mp4",
                },
                set(zip_file.namelist()),
            )

    def test_apply_prune_archive_failure_deletes_nothing(self):
        self._build_two_scene_project()
        report = scan_project(self.project)
        archive = Path(self.temp.name) / "fail.zip"

        with patch("feverslop.application.artifact_prune.ZipFile", _FailingZip):
            with self.assertRaises(OSError):
                apply_prune(self.project, archive, report)

        self.assertFalse(archive.exists())
        self.assertTrue(self.layout.scene_workflow(1).is_file())
        self.assertTrue(self.layout.scene_raw_video(1).is_file())
        self.assertTrue(self.layout.references_plan.is_file())
        self.assertTrue(self.layout.scene_workflow(2).is_file())

    def test_apply_prune_with_no_eligible_candidates_deletes_nothing(self):
        self._write_base((2,))
        self._write_blocked_scene(2)
        self._write_protected_files()
        report = scan_project(self.project)
        archive = Path(self.temp.name) / "empty.zip"

        final = apply_prune(self.project, archive, report)

        self.assertEqual([], final.deleted)
        self.assertTrue(archive.is_file())
        self.assertTrue(self.layout.scene_workflow(2).is_file())
        self.assertTrue(self.layout.scene_raw_video(2).is_file())


if __name__ == "__main__":
    unittest.main()
