from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.canonical_plan_migration import analyze_canonical_plan_migration
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.errors import FeverSlopDataError


class CanonicalPlanMigrationAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name)
        self.plans = self.project / "output" / "render" / "plans"
        self.plans.mkdir(parents=True)

    def _scene(self, number: int, segment_id: str) -> dict:
        relay = [{"frame_start": 0, "frame_end": 48, "state": "singing"}]
        timing = [{"start": 0, "end": 48}]
        return {
            "scene": number,
            "metadata": {"segment_id": segment_id},
            "z_image": {"prompt": f"still {segment_id}"},
            "ltx": {
                "base_prompt": f"base {segment_id}",
                "i2v_prompt_from_t2i": f"i2v {segment_id}",
                "prompt_relay": relay,
            },
            "h3": {"prompt": f"h3 {segment_id}"},
            "performance_timing": timing,
            "canonical": build_canonical_scene(
                segment_id=segment_id,
                generated_roles={
                    PromptRole.Z_IMAGE: f"still {segment_id}",
                    PromptRole.LTX_BASE: f"base {segment_id}",
                    PromptRole.LTX_I2V: f"i2v {segment_id}",
                    PromptRole.LTX_RELAY: relay,
                    PromptRole.H3_VIDEO: f"h3 {segment_id}",
                    PromptRole.PERFORMANCE_TIMING: timing,
                },
            ),
        }

    def _write(self, name: str, value: object) -> Path:
        path = self.plans / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def _analyze(self):
        return analyze_canonical_plan_migration(CanonicalPlanStore(self.project).load())

    def test_base_legacy_edits_are_importable_for_every_supported_role(self):
        scene = self._scene(1, "segment-a")
        scene["z_image"]["prompt"] = "edited still"
        scene["ltx"]["base_prompt"] = "edited base"
        scene["ltx"]["i2v_prompt_from_t2i"] = "edited i2v"
        scene["ltx"]["prompt_relay"] = [{"state": "edited relay"}]
        scene["h3"]["prompt"] = "edited h3"
        scene["performance_timing"] = [{"start": 2, "end": 46}]
        self._write("base.json", [scene])

        before = {path: path.read_bytes() for path in self.project.rglob("*") if path.is_file()}
        report = self._analyze()

        self.assertEqual(
            {
                PromptRole.Z_IMAGE,
                PromptRole.LTX_BASE,
                PromptRole.LTX_I2V,
                PromptRole.LTX_RELAY,
                PromptRole.H3_VIDEO,
                PromptRole.PERFORMANCE_TIMING,
            },
            {finding.role for finding in report.importable},
        )
        self.assertFalse(report.unresolved)
        self.assertEqual(
            before,
            {path: path.read_bytes() for path in self.project.rglob("*") if path.is_file()},
        )

    def test_reordered_derived_scenes_match_by_stable_identity(self):
        first = self._scene(1, "segment-a")
        second = self._scene(2, "segment-b")
        self._write("base.json", [first, second])
        anchored = json.loads(json.dumps([first, second]))
        references = json.loads(json.dumps([second, first]))
        references[1]["z_image"]["prompt"] = "human edit after reorder"
        self._write("anchored.json", anchored)
        self._write("references.json", references)

        report = self._analyze()

        finding = next(item for item in report.importable if item.role == PromptRole.Z_IMAGE)
        self.assertEqual(first["canonical"]["scene_id"], finding.scene_id)
        self.assertEqual("scene_id", finding.matched_by)

    def test_conflicting_candidates_for_one_role_are_unresolved(self):
        scene = self._scene(1, "segment-a")
        scene["z_image"]["prompt"] = "edit from base"
        self._write("base.json", [scene])
        anchored = json.loads(json.dumps([scene]))
        anchored[0]["z_image"]["prompt"] = "still segment-a"
        references = json.loads(json.dumps([scene]))
        references[0]["z_image"]["prompt"] = "edit from references"
        self._write("anchored.json", anchored)
        self._write("references.json", references)

        report = self._analyze()

        self.assertFalse([item for item in report.importable if item.role == PromptRole.Z_IMAGE])
        self.assertTrue(any(item.reason == "conflicting candidate values" for item in report.unresolved))

    def test_orphan_and_duplicate_identity_are_unresolved(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        duplicate = json.loads(json.dumps(scene))
        duplicate["scene"] = 2
        orphan = self._scene(3, "segment-orphan")
        self._write("references.json", [scene, duplicate, orphan])

        report = self._analyze()

        reasons = {item.reason for item in report.unresolved}
        self.assertIn("duplicate scene identity", reasons)
        self.assertIn("orphan scene", reasons)

    def test_missing_and_contradictory_scene_identity_are_unresolved(self):
        first = self._scene(1, "segment-a")
        second = self._scene(2, "segment-b")
        self._write("base.json", [first, second])
        contradictory = json.loads(json.dumps(first))
        contradictory["canonical"]["segment_id"] = "segment-b"
        self._write("references.json", [contradictory])

        report = self._analyze()

        reasons = {item.reason for item in report.unresolved}
        self.assertIn("conflicting scene identity", reasons)
        self.assertIn("missing scene", reasons)

    def test_empty_candidate_is_not_imported(self):
        scene = self._scene(1, "segment-a")
        scene["h3"]["prompt"] = ""
        self._write("base.json", [scene])

        report = self._analyze()

        self.assertFalse(report.importable)
        self.assertTrue(any(item.reason == "candidate value is empty" for item in report.unresolved))

    def test_malformed_optional_artifact_is_reported_without_writes(self):
        self._write("base.json", [self._scene(1, "segment-a")])
        references = self.plans / "references.json"
        references.write_text("{broken", encoding="utf-8")
        before = references.read_bytes()

        report = self._analyze()

        self.assertTrue(any(item.reason == "malformed JSON" for item in report.unresolved))
        self.assertEqual(before, references.read_bytes())

    def test_existing_equal_override_is_an_idempotent_no_op(self):
        scene = self._scene(1, "segment-a")
        scene["z_image"]["prompt"] = "already imported"
        scene["canonical"]["roles"][PromptRole.Z_IMAGE]["override"] = {
            "value": "already imported",
            "provenance": {"source": "legacy-plan-migration"},
        }
        self._write("base.json", [scene])

        report = self._analyze()

        self.assertFalse(report.importable)
        self.assertTrue(any(item.role == PromptRole.Z_IMAGE for item in report.no_op))

    def test_existing_different_override_blocks_migration(self):
        scene = self._scene(1, "segment-a")
        scene["z_image"]["prompt"] = "legacy candidate"
        scene["canonical"]["roles"][PromptRole.Z_IMAGE]["override"] = {
            "value": "intentional current override",
            "provenance": {"source": "human"},
        }
        self._write("base.json", [scene])

        report = self._analyze()

        self.assertFalse(report.importable)
        self.assertTrue(any(
            item.reason == "candidate conflicts with existing override"
            for item in report.unresolved
        ))

    def test_legacy_refs_file_uses_anchored_pass_through_baseline(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        self._write("anchored.json", [scene])
        legacy = json.loads(json.dumps([scene]))
        legacy[0]["h3"]["prompt"] = "legacy refs human edit"
        legacy_path = self.project / "output/render/render_plan_song_refs.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

        report = self._analyze()

        finding = next(item for item in report.importable if item.role == PromptRole.H3_VIDEO)
        self.assertTrue(finding.source_path.endswith("render_plan_song_refs.json"))

    def test_references_without_anchored_plan_use_canonical_base_as_baseline(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        self._write("references.json", json.loads(json.dumps([scene])))

        report = self._analyze()

        self.assertFalse(report.unresolved)

    def test_stale_h3_prompt_in_derived_references_plan_is_not_importable(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        references = json.loads(json.dumps([scene]))
        references[0]["h3"]["prompt"] = "h3 prompt from the previous generation"
        self._write("references.json", references)

        report = self._analyze()

        self.assertFalse(report.importable)
        self.assertFalse(report.unresolved)

    def test_ingredients_relay_is_compared_to_msr_relay(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        references = json.loads(json.dumps([scene]))
        references[0]["ltx"]["msr_prompt_relay"] = [{"state": "generated msr relay"}]
        ingredients = json.loads(json.dumps([scene]))
        ingredients[0]["ltx"]["prompt_relay"] = [{"state": "human ingredients relay"}]
        self._write("references.json", references)
        self._write("ingredients.json", ingredients)

        report = self._analyze()

        finding = next(
            item for item in report.importable if item.role == PromptRole.INGREDIENTS_RELAY
        )
        self.assertEqual("ltx.prompt_relay", finding.field_path)


class CanonicalPlanMigrationApplyTests(unittest.TestCase):
    setUp = CanonicalPlanMigrationAnalysisTests.setUp
    _scene = CanonicalPlanMigrationAnalysisTests._scene
    _write = CanonicalPlanMigrationAnalysisTests._write
    _analyze = CanonicalPlanMigrationAnalysisTests._analyze

    def test_apply_backs_up_all_sources_and_only_rewrites_base(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        anchored = self._write("anchored.json", [scene])
        references_value = json.loads(json.dumps([scene]))
        references_value[0]["z_image"]["prompt"] = "human still"
        references = self._write("references.json", references_value)
        original_anchored = anchored.read_bytes()
        original_references = references.read_bytes()
        report = self._analyze()

        result = CanonicalPlanStore(self.project).apply(report, run_id="20260824T120000Z")

        saved = json.loads((self.plans / "base.json").read_text(encoding="utf-8"))
        override = saved[0]["canonical"]["roles"][PromptRole.Z_IMAGE]["override"]
        self.assertEqual("human still", override["value"])
        self.assertEqual("legacy-plan-migration", override["provenance"]["source"])
        self.assertEqual(original_anchored, anchored.read_bytes())
        self.assertEqual(original_references, references.read_bytes())
        self.assertEqual(
            original_references,
            (result.backup_dir / "output/render/plans/references.json").read_bytes(),
        )
        self.assertTrue((result.backup_dir / "output/render/plans/base.json").is_file())
        self.assertTrue((result.backup_dir / "report.json").is_file())

    def test_apply_refuses_unresolved_findings_without_creating_backup(self):
        scene = self._scene(1, "segment-a")
        self._write("base.json", [scene])
        self._write("references.json", [scene, json.loads(json.dumps(scene))])
        report = self._analyze()

        with self.assertRaisesRegex(FeverSlopDataError, "unresolved"):
            CanonicalPlanStore(self.project).apply(report, run_id="blocked")

        self.assertFalse((self.plans / "legacy-migration").exists())

    def test_apply_refuses_source_hash_drift(self):
        scene = self._scene(1, "segment-a")
        scene["h3"]["prompt"] = "human h3"
        base = self._write("base.json", [scene])
        report = self._analyze()
        base.write_text(base.read_text(encoding="utf-8") + "\n", encoding="utf-8")

        with self.assertRaisesRegex(FeverSlopDataError, "changed since analysis"):
            CanonicalPlanStore(self.project).apply(report, run_id="drift")

        self.assertFalse((self.plans / "legacy-migration").exists())

    def test_failure_after_backup_leaves_base_unchanged_and_rerun_is_safe(self):
        scene = self._scene(1, "segment-a")
        scene["h3"]["prompt"] = "human h3"
        base = self._write("base.json", [scene])
        original_base = base.read_bytes()
        report = self._analyze()

        with patch("feverslop.adapters.canonical_plan_store.atomic_write_json") as writer:
            writer.side_effect = OSError("injected write failure")
            with self.assertRaisesRegex(OSError, "injected"):
                CanonicalPlanStore(self.project).apply(report, run_id="interrupted")

        self.assertEqual(original_base, base.read_bytes())
        backup = self.plans / "legacy-migration" / "interrupted"
        self.assertEqual(original_base, (backup / "output/render/plans/base.json").read_bytes())
        self.assertTrue((backup / "report.json").is_file())

        result = CanonicalPlanStore(self.project).apply(report, run_id="recovered")
        self.assertTrue(result.applied)

    def test_second_analysis_and_apply_are_write_free_no_op(self):
        scene = self._scene(1, "segment-a")
        scene["z_image"]["prompt"] = "human still"
        self._write("base.json", [scene])
        store = CanonicalPlanStore(self.project)
        first = store.apply(self._analyze(), run_id="first")
        base_after_first = (self.plans / "base.json").read_bytes()

        second = store.apply(self._analyze(), run_id="second")

        self.assertTrue(first.applied)
        self.assertFalse(second.applied)
        self.assertIsNone(second.backup_dir)
        self.assertEqual(base_after_first, (self.plans / "base.json").read_bytes())
        self.assertFalse((self.plans / "legacy-migration" / "second").exists())


if __name__ == "__main__":
    unittest.main()
