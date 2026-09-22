"""Tests for the manifest-governed artifact lifecycle domain (issue #1388).

Pure, path-anchored classification and eligibility decisions; no filesystem
I/O is required because ``classify_artifact`` and ``evaluate_candidate``
compare paths only and all manifest state is injected via
``ScenePruneContext`` / ``PlanPruneContext``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from feverslop.domain.artifact_lifecycle import (
    CANDIDATE_KINDS,
    PROTECTED_KINDS,
    ArtifactKind,
    ArtifactLifecycleClass,
    IneligibilityReason,
    PlanPruneContext,
    PruneCandidate,
    ScenePruneContext,
    classify_artifact,
    evaluate_candidate,
    is_candidate,
    is_protected,
    lifecycle_class_for,
    protected_reason_for,
)


def _path(project: Path, *parts: str) -> Path:
    return project.joinpath(*parts)


class ArtifactLifecycleClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Classification is path-anchored and performs no I/O, so a
        # non-existent project root is sufficient.
        self.project = Path("project")

    def test_classifies_every_protected_class(self):
        cases = (
            ("config.json", ArtifactKind.config, ArtifactLifecycleClass.authoritative),
            ("output/render/plans/base.json", ArtifactKind.canonical_plan, ArtifactLifecycleClass.authoritative),
            ("output/prompts/story_plan_song-a.json", ArtifactKind.canonical_plan, ArtifactLifecycleClass.authoritative),
            ("output/prompts/story_plan_song-a.manifest.json", ArtifactKind.canonical_plan, ArtifactLifecycleClass.authoritative),
            ("output/references/actors/a/views/sheet.png", ArtifactKind.reference_asset, ArtifactLifecycleClass.authoritative),
            ("output/references/manifest.json", ArtifactKind.reference_manifest, ArtifactLifecycleClass.authoritative),
            ("output/render/scenes/scene_0001/manifest.json", ArtifactKind.reference_manifest, ArtifactLifecycleClass.authoritative),
            ("output/render/scenes/scene_0001/final.mp4", ArtifactKind.final_scene_clip, ArtifactLifecycleClass.authoritative),
            ("output/render/scenes/scene_0001/final_facefix.mp4", ArtifactKind.final_scene_clip, ArtifactLifecycleClass.authoritative),
            ("output/render/scenes/scene_0001/upscale_final.mp4", ArtifactKind.final_scene_clip, ArtifactLifecycleClass.authoritative),
            ("output/render/final/video_only.mp4", ArtifactKind.assembled_final_video, ArtifactLifecycleClass.authoritative),
            ("output/review.md", ArtifactKind.review_export, ArtifactLifecycleClass.review_export),
            ("output/render/storyboard/page.html", ArtifactKind.review_export, ArtifactLifecycleClass.review_export),
        )
        for relative, expected_kind, expected_class in cases:
            with self.subTest(path=relative):
                kind = classify_artifact(_path(self.project, *relative.split("/")), self.project)
                self.assertIs(expected_kind, kind)
                self.assertTrue(is_protected(kind))
                self.assertFalse(is_candidate(kind))
                self.assertIs(expected_class, lifecycle_class_for(kind))
                self.assertNotEqual("", protected_reason_for(kind))

    def test_user_override_is_protected_through_the_canonical_plan(self):
        # Override entries live inside base.json; the kind is protected with
        # its own reason even though it has no distinct path.
        self.assertTrue(is_protected(ArtifactKind.user_override))
        self.assertIs(ArtifactLifecycleClass.authoritative, lifecycle_class_for(ArtifactKind.user_override))
        self.assertNotEqual("", protected_reason_for(ArtifactKind.user_override))

    def test_classifies_candidate_kinds(self):
        cases = (
            "output/render/scenes/scene_0001/workflow.json",
            "output/render/scenes/scene_0001/raw.mp4",
            "output/render/plans/compact.json",
            "output/render/plans/anchored.json",
            "output/render/plans/references.json",
            "output/render/plans/ingredients.json",
        )
        for relative in cases:
            with self.subTest(path=relative):
                kind = classify_artifact(_path(self.project, *relative.split("/")), self.project)
                self.assertIn(kind, CANDIDATE_KINDS)
                self.assertTrue(is_candidate(kind))
                self.assertFalse(is_protected(kind))
                self.assertIs(ArtifactLifecycleClass.resume_cache, lifecycle_class_for(kind))
                self.assertEqual("", protected_reason_for(kind))

    def test_other_paths_are_not_candidates(self):
        cases = (
            "output/render/scenes/scene_0001/h3_prompt.json",
            "output/render/scenes/scene_0001/workflow_facefix.json",
            "output/render/scenes/scene_0001/facefix/a/crop.mp4",
            "output/prompts/other.json",
            "output/render/plans/unknown.json",
            "elsewhere/file.json",
        )
        for relative in cases:
            with self.subTest(path=relative):
                kind = classify_artifact(_path(self.project, *relative.split("/")), self.project)
                self.assertIs(ArtifactKind.other, kind)
                self.assertIsNone(lifecycle_class_for(kind))
                self.assertFalse(is_protected(kind))
                self.assertFalse(is_candidate(kind))
                self.assertEqual((False, ("not_a_candidate",)), evaluate_candidate(
                    _path(self.project, *relative.split("/")), self.project
                ))

    def test_protected_kinds_are_never_eligible(self):
        for relative in (
            "config.json",
            "output/render/plans/base.json",
            "output/prompts/story_plan_song-a.json",
            "output/references/actors/a/views/sheet.png",
            "output/references/manifest.json",
            "output/render/scenes/scene_0001/manifest.json",
            "output/render/scenes/scene_0001/final.mp4",
            "output/render/final/video_only.mp4",
            "output/review.md",
        ):
            with self.subTest(path=relative):
                self.assertEqual(
                    (False, ()),
                    evaluate_candidate(_path(self.project, *relative.split("/")), self.project),
                )
        self.assertEqual(PROTECTED_KINDS, frozenset(k for k in ArtifactKind if is_protected(k)))
        self.assertEqual(CANDIDATE_KINDS, frozenset(k for k in ArtifactKind if is_candidate(k)))
        self.assertEqual(12, len(ArtifactKind))


class ArtifactLifecycleEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = Path("project")
        self.workflow = _path(self.project, "output", "render", "scenes", "scene_0001", "workflow.json")
        self.derived_plan = _path(self.project, "output", "render", "plans", "references.json")

    def _clean_scene_context(self) -> ScenePruneContext:
        return ScenePruneContext(
            scene_number=1,
            manifest_present=True,
            manifest_valid=True,
            final_successor_present=True,
        )

    def test_clean_scene_is_eligible(self):
        self.assertEqual(
            (True, ()),
            evaluate_candidate(self.workflow, self.project, scene_context=self._clean_scene_context()),
        )

    def test_blocked_scene_collects_all_reasons(self):
        # ``manifest_missing`` and ``manifest_invalid`` are mutually exclusive
        # (a missing manifest is not also invalid), so the maximal scene
        # blocker set has six reasons; ``manifest_missing`` is covered by the
        # individual-reason test.
        context = ScenePruneContext(
            scene_number=1,
            manifest_present=True,
            manifest_valid=False,
            verify_mismatches=("workflow: sha256 mismatch",),
            dependency_mismatches=("workflow fingerprint changed",),
            final_successor_present=False,
            facefix_pending=True,
            upscale_pending=True,
            referenced_by_unfinished=True,
        )
        eligible, reasons = evaluate_candidate(self.workflow, self.project, scene_context=context)
        self.assertFalse(eligible)
        self.assertEqual(
            {
                IneligibilityReason.manifest_invalid.value,
                IneligibilityReason.fingerprint_mismatch.value,
                IneligibilityReason.no_final_successor.value,
                IneligibilityReason.facefix_pending.value,
                IneligibilityReason.upscale_pending.value,
                IneligibilityReason.referenced_by_unfinished.value,
            },
            set(reasons),
        )
        self.assertEqual(6, len(reasons))

    def test_each_scene_ineligibility_reason_individually(self):
        cases = (
            ("manifest_missing", {"manifest_present": False}),
            ("manifest_invalid", {"manifest_valid": False}),
            ("manifest_invalid_via_verify", {"verify_mismatches": ("template: missing",)}),
            ("fingerprint_mismatch", {"dependency_mismatches": ("reference fingerprint changed",)}),
            ("no_final_successor", {"final_successor_present": False}),
            ("facefix_pending", {"facefix_pending": True}),
            ("upscale_pending", {"upscale_pending": True}),
            ("referenced_by_unfinished", {"referenced_by_unfinished": True}),
        )
        for name, overrides in cases:
            with self.subTest(reason=name):
                context = ScenePruneContext(
                    **{
                        "scene_number": 1,
                        "manifest_present": True,
                        "manifest_valid": True,
                        "final_successor_present": True,
                        **overrides,
                    }
                )
                eligible, reasons = evaluate_candidate(self.workflow, self.project, scene_context=context)
                # "manifest_invalid_via_verify" exercises the verify-mismatch branch.
                expected = IneligibilityReason(name.split("_via")[0]).value
                self.assertFalse(eligible)
                self.assertEqual((expected,), reasons)

    def test_derived_plan_provenance_match_is_eligible(self):
        context = PlanPruneContext(base_plan_present=True, provenance_present=True)
        self.assertEqual(
            (True, ()),
            evaluate_candidate(self.derived_plan, self.project, plan_context=context),
        )

    def test_derived_plan_provenance_mismatch_and_missing_base(self):
        cases = (
            (
                "manifest_missing",
                PlanPruneContext(base_plan_present=True),
                (IneligibilityReason.manifest_missing.value,),
            ),
            (
                "fingerprint_mismatch",
                PlanPruneContext(
                    base_plan_present=True,
                    provenance_present=True,
                    provenance_mismatches=("scene 1: workflow fingerprint changed",),
                ),
                (IneligibilityReason.fingerprint_mismatch.value,),
            ),
            (
                "dependencies_missing",
                PlanPruneContext(provenance_present=True),
                (IneligibilityReason.dependencies_missing.value,),
            ),
            (
                "all_reasons",
                PlanPruneContext(
                    provenance_mismatches=("scene 1: reference fingerprint changed",),
                ),
                (
                    IneligibilityReason.manifest_missing.value,
                    IneligibilityReason.fingerprint_mismatch.value,
                    IneligibilityReason.dependencies_missing.value,
                ),
            ),
        )
        for name, context, expected_reasons in cases:
            with self.subTest(reason=name):
                eligible, reasons = evaluate_candidate(self.derived_plan, self.project, plan_context=context)
                self.assertFalse(eligible)
                self.assertEqual(expected_reasons, reasons)

    def test_evaluate_requires_the_matching_context(self):
        with self.assertRaises(ValueError):
            evaluate_candidate(self.workflow, self.project)
        with self.assertRaises(ValueError):
            evaluate_candidate(self.derived_plan, self.project)
        # Both contexts supplied is fine; the kind selects the relevant one.
        self.assertEqual(
            (True, ()),
            evaluate_candidate(
                self.workflow,
                self.project,
                scene_context=self._clean_scene_context(),
                plan_context=PlanPruneContext(base_plan_present=True, provenance_present=True),
            ),
        )

    def test_prune_candidate_is_frozen(self):
        candidate = PruneCandidate(
            path=self.workflow,
            relative_path="output/render/scenes/scene_0001/workflow.json",
            kind=ArtifactKind.scene_workflow,
            lifecycle_class=ArtifactLifecycleClass.resume_cache,
            size_bytes=2,
            eligible=True,
        )
        with self.assertRaises(AttributeError):
            candidate.eligible = False


if __name__ == "__main__":
    unittest.main()
