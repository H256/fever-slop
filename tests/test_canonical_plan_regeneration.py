from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.canonical_plan_regeneration import (
    CanonicalPlanRegenerationService,
)
from feverslop.composition.canonical_plan_regenerator import CanonicalPlanRegenerator
from feverslop.domain.canonical_render_plan import (
    PromptRole,
    build_canonical_scene,
    resolve_effective_role,
)
from feverslop.errors import FeverSlopDataError


def _scene(
    number: int,
    segment_id: str,
    generated: str,
    *,
    override: str | None = None,
) -> dict:
    canonical = build_canonical_scene(
        segment_id=segment_id,
        generated_roles={PromptRole.H3_VIDEO: generated},
    )
    if override is not None:
        canonical["roles"][PromptRole.H3_VIDEO]["override"] = {
            "value": override,
            "provenance": {"source": "human", "note": "approved"},
        }
    return {
        "scene": number,
        "metadata": {"segment_id": segment_id},
        "h3": {"prompt": generated},
        "canonical": canonical,
    }


class CanonicalPlanRegenerationMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = CanonicalPlanRegenerationService()

    def test_generate_override_regenerate_keeps_override_and_replaces_generated(self):
        existing = _scene(1, "segment-a", "old generated", override="human approved")
        regenerated = _scene(1, "segment-a", "new generated")

        result = self.service.merge(
            existing_scenes=[existing],
            generated_scenes=[regenerated],
        )

        role = result.scenes[0]["canonical"]["roles"][PromptRole.H3_VIDEO]
        self.assertEqual("new generated", role["generated"]["value"])
        self.assertEqual(
            {
                "value": "human approved",
                "provenance": {"source": "human", "note": "approved"},
            },
            role["override"],
        )
        self.assertEqual(
            "human approved",
            resolve_effective_role(result.scenes[0], PromptRole.H3_VIDEO),
        )
        self.assertNotIn("effective", role)

    def test_scene_reordering_does_not_move_overrides(self):
        first = _scene(1, "segment-a", "old a", override="override a")
        second = _scene(2, "segment-b", "old b", override="override b")

        result = self.service.merge(
            existing_scenes=[first, second],
            generated_scenes=[
                _scene(2, "segment-b", "new b"),
                _scene(1, "segment-a", "new a"),
            ],
        )

        self.assertEqual("override b", resolve_effective_role(result.scenes[0], PromptRole.H3_VIDEO))
        self.assertEqual("override a", resolve_effective_role(result.scenes[1], PromptRole.H3_VIDEO))

    def test_disappearing_generated_role_retains_human_owned_role(self):
        existing = _scene(1, "segment-a", "old", override="human")
        regenerated = _scene(1, "segment-a", "new")
        regenerated["canonical"]["roles"].pop(PromptRole.H3_VIDEO)

        result = self.service.merge([existing], [regenerated])

        role = result.scenes[0]["canonical"]["roles"][PromptRole.H3_VIDEO]
        self.assertNotIn("generated", role)
        self.assertEqual("human", role["override"]["value"])

    def test_deleted_identity_emits_orphan_diagnostic_without_reattaching_override(self):
        existing = _scene(1, "segment-old", "old", override="must not move")
        regenerated = _scene(1, "segment-new", "new")

        result = self.service.merge([existing], [regenerated])

        self.assertNotIn(
            "override",
            result.scenes[0]["canonical"]["roles"][PromptRole.H3_VIDEO],
        )
        self.assertEqual(["orphaned_override_scene"], [item.code for item in result.diagnostics])

    def test_duplicate_or_missing_canonical_identity_is_rejected(self):
        duplicate = _scene(1, "segment-a", "one")
        malformed = {"scene": 2}

        with self.assertRaisesRegex(FeverSlopDataError, "duplicate.*scene_id"):
            self.service.merge([], [duplicate, json.loads(json.dumps(duplicate))])
        with self.assertRaisesRegex(FeverSlopDataError, "canonical identity"):
            self.service.merge([], [malformed])

    def test_same_scene_id_with_changed_segment_id_is_an_explicit_conflict(self):
        existing = _scene(1, "segment-a", "old", override="human")
        regenerated = _scene(1, "segment-a", "new")
        regenerated["canonical"]["segment_id"] = "segment-renamed"

        with self.assertRaisesRegex(FeverSlopDataError, "canonical identity conflict"):
            self.service.merge([existing], [regenerated])

    def test_selected_regeneration_leaves_unselected_scene_object_unchanged(self):
        first = _scene(1, "segment-a", "old a", override="human a")
        second = _scene(2, "segment-b", "old b", override="human b")
        second["references"] = {"operator_note": "keep exact object"}

        result = self.service.merge(
            [first, second],
            [_scene(1, "segment-a", "new a"), _scene(2, "segment-b", "new b")],
            selected_scene_numbers={1},
        )

        self.assertEqual("new a", result.scenes[0]["h3"]["prompt"])
        self.assertEqual(second, result.scenes[1])

    def test_selected_identity_change_retains_old_scene_without_appending_replacement(self):
        existing = _scene(1, "segment-old", "old", override="human")

        result = self.service.merge(
            [existing],
            [_scene(1, "segment-new", "new")],
            selected_scene_numbers={1},
        )

        self.assertEqual((existing,), result.scenes)
        self.assertEqual(["selected_identity_missing"], [item.code for item in result.diagnostics])

    def test_reference_bindings_follow_scene_id_after_reorder(self):
        first = _scene(1, "segment-a", "old a")
        second = _scene(2, "segment-b", "old b")
        enriched_first = json.loads(json.dumps(first))
        enriched_first["references"] = {
            "actor_msr_paths": ["actors/a.png"],
            "operator_note": "must not be copied",
        }
        enriched_second = json.loads(json.dumps(second))
        enriched_second["references"] = {"actor_msr_paths": ["actors/b.png"]}

        result = self.service.merge(
            [first, second],
            [_scene(2, "segment-b", "new b"), _scene(1, "segment-a", "new a")],
            reference_scenes=[enriched_first, enriched_second],
        )

        self.assertEqual(["actors/b.png"], result.scenes[0]["references"]["actor_msr_paths"])
        self.assertEqual(["actors/a.png"], result.scenes[1]["references"]["actor_msr_paths"])
        self.assertNotIn("operator_note", result.scenes[1]["references"])

    def test_reference_binding_with_reused_scene_number_is_not_transferred(self):
        old = _scene(1, "segment-old", "old")
        old["references"] = {"location_msr_path": "wrong.png"}

        result = self.service.merge(
            [],
            [_scene(1, "segment-new", "new")],
            reference_scenes=[old],
        )

        self.assertNotIn("references", result.scenes[0])
        self.assertEqual("orphaned_reference_scene", result.diagnostics[0].code)

    def test_reference_binding_with_conflicting_segment_identity_is_rejected(self):
        generated = _scene(1, "segment-a", "new")
        reference = json.loads(json.dumps(generated))
        reference["canonical"]["segment_id"] = "segment-conflict"
        reference["references"] = {"actor_msr_paths": ["wrong.png"]}

        with self.assertRaisesRegex(FeverSlopDataError, "reference identity conflict"):
            self.service.merge([], [generated], reference_scenes=[reference])


class CanonicalPlanRegenerationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name)
        self.base = self.project / "output/render/plans/base.json"
        self.base.parent.mkdir(parents=True)
        self.store = CanonicalPlanStore(self.project)

    def test_capture_and_commit_replace_base_after_hash_revalidation(self):
        existing = [_scene(1, "segment-a", "old")]
        updated = [_scene(1, "segment-a", "new")]
        self.base.write_text(json.dumps(existing), encoding="utf-8")

        snapshot = self.store.capture_regeneration()
        result = self.store.commit_regeneration(snapshot, updated)

        self.assertTrue(snapshot.exists)
        self.assertIsNotNone(snapshot.sha256)
        self.assertEqual(self.base, result)
        self.assertEqual(updated, json.loads(self.base.read_text(encoding="utf-8")))

    def test_commit_rejects_existing_plan_changed_after_capture(self):
        existing = [_scene(1, "segment-a", "old")]
        self.base.write_text(json.dumps(existing), encoding="utf-8")
        snapshot = self.store.capture_regeneration()
        drift = [_scene(1, "segment-a", "concurrent edit")]
        self.base.write_text(json.dumps(drift), encoding="utf-8")

        with self.assertRaisesRegex(FeverSlopDataError, "changed during regeneration"):
            self.store.commit_regeneration(snapshot, [_scene(1, "segment-a", "new")])

        self.assertEqual(drift, json.loads(self.base.read_text(encoding="utf-8")))

    def test_commit_rejects_plan_created_after_absent_capture(self):
        snapshot = self.store.capture_regeneration()
        concurrent = [_scene(1, "segment-a", "concurrent")]
        self.base.write_text(json.dumps(concurrent), encoding="utf-8")

        with self.assertRaisesRegex(FeverSlopDataError, "appeared during regeneration"):
            self.store.commit_regeneration(snapshot, [_scene(1, "segment-a", "new")])

        self.assertEqual(concurrent, json.loads(self.base.read_text(encoding="utf-8")))


class CanonicalPlanRegeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name)
        self.base = self.project / "output/render/plans/base.json"
        self.base.parent.mkdir(parents=True)

    def test_writer_preserves_override_and_reference_bindings_in_one_commit(self):
        existing = _scene(1, "segment-a", "old", override="human")
        self.base.write_text(json.dumps([existing]), encoding="utf-8")
        enriched = json.loads(json.dumps(existing))
        enriched["references"] = {"actor_msr_paths": ["actors/a.png"]}
        references = self.base.with_name("references.json")
        references.write_text(json.dumps([enriched]), encoding="utf-8")
        regenerator = CanonicalPlanRegenerator(
            self.project,
            reference_plan_path=references,
        )

        result_path = regenerator.write(self.base, [_scene(1, "segment-a", "new")])

        saved = json.loads(result_path.read_text(encoding="utf-8"))[0]
        self.assertEqual("new", saved["canonical"]["roles"][PromptRole.H3_VIDEO]["generated"]["value"])
        self.assertEqual("human", saved["canonical"]["roles"][PromptRole.H3_VIDEO]["override"]["value"])
        self.assertEqual(["actors/a.png"], saved["references"]["actor_msr_paths"])

    def test_selected_writer_keeps_unselected_existing_scene(self):
        first = _scene(1, "segment-a", "old a", override="human a")
        second = _scene(2, "segment-b", "old b", override="human b")
        self.base.write_text(json.dumps([first, second]), encoding="utf-8")
        regenerator = CanonicalPlanRegenerator(
            self.project,
            selected_scene_numbers={1},
        )

        regenerator.write(
            self.base,
            [_scene(1, "segment-a", "new a"), _scene(2, "segment-b", "new b")],
        )

        saved = json.loads(self.base.read_text(encoding="utf-8"))
        self.assertEqual("new a", saved[0]["h3"]["prompt"])
        self.assertEqual(second, saved[1])

    def test_selected_regeneration_without_existing_base_is_rejected(self):
        with self.assertRaisesRegex(FeverSlopDataError, "requires an existing canonical base plan"):
            CanonicalPlanRegenerator(self.project, selected_scene_numbers={1})


if __name__ == "__main__":
    unittest.main()
