import unittest
from feverslop.domain.rebuild_policy import (
    ArtifactFingerprint,
    ArtifactKind,
    ChangeKind,
    ChangeSet,
    Freshness,
    compute_freshness,
    preview_rebuild,
)


class ChangeSetTests(unittest.TestCase):
    def test_prompt_change(self):
        change = ChangeSet.prompt(scene_numbers={4})
        self.assertIn(ChangeKind.PROMPT, change.change_kinds)
        self.assertEqual({4}, change.scene_numbers)

    def test_timeline_change(self):
        change = ChangeSet.timeline(scene_numbers={1, 2})
        self.assertIn(ChangeKind.TIMELINE, change.change_kinds)

    def test_reference_assignment_change(self):
        change = ChangeSet.references(scene_numbers={3})
        self.assertIn(ChangeKind.REFERENCE_ASSIGNMENT, change.change_kinds)

    def test_workflow_profile_change(self):
        change = ChangeSet.workflow_profile()
        self.assertIn(ChangeKind.WORKFLOW_PROFILE, change.change_kinds)
        self.assertIsNone(change.scene_numbers)

    def test_dimensions_change(self):
        change = ChangeSet.dimensions()
        self.assertIn(ChangeKind.DIMENSIONS, change.change_kinds)
        self.assertIsNone(change.scene_numbers)

    def test_review_ordering_change(self):
        change = ChangeSet.review_ordering()
        self.assertIn(ChangeKind.REVIEW_ORDERING, change.change_kinds)
        self.assertIsNone(change.scene_numbers)

    def test_combined_changes(self):
        change = ChangeSet.combine(
            ChangeSet.prompt(scene_numbers={1, 2}),
            ChangeSet.reference_assignment(scene_numbers={2, 3}),
        )
        self.assertIn(ChangeKind.PROMPT, change.change_kinds)
        self.assertIn(ChangeKind.REFERENCE_ASSIGNMENT, change.change_kinds)
        self.assertEqual({1, 2, 3}, change.scene_numbers)


class PromptChangeRebuildTests(unittest.TestCase):
    def test_prompt_change_rebuilds_scene_render_but_reuses_audio_analysis(self):
        plan = preview_rebuild(ChangeSet.prompt(scene_numbers={4}))
        self.assertIn(ArtifactKind.SCENE_RENDER, plan.rebuild)
        self.assertIn(ArtifactKind.AUDIO_TIMELINE, plan.reuse)

    def test_prompt_change_scope_is_scene_local(self):
        plan = preview_rebuild(ChangeSet.prompt(scene_numbers={3}))
        self.assertIn(3, plan.affected_scenes)
        self.assertNotIn(1, plan.affected_scenes)

    def test_prompt_change_does_not_invalidate_all_scenes(self):
        plan = preview_rebuild(ChangeSet.prompt(scene_numbers={5}))
        self.assertNotIn(ArtifactKind.FINAL_VIDEO, plan.rebuild)


class TimelineChangeRebuildTests(unittest.TestCase):
    def test_timeline_change_invalidates_downstream_prompt_and_render(self):
        plan = preview_rebuild(ChangeSet.timeline(scene_numbers={2}))
        self.assertIn(ArtifactKind.SCENE_RENDER, plan.rebuild)
        self.assertIn(ArtifactKind.AUDIO_TIMELINE, plan.rebuild)

    def test_timeline_change_reuses_references(self):
        plan = preview_rebuild(ChangeSet.timeline(scene_numbers={2}))
        self.assertIn(ArtifactKind.REFERENCE_SHEETS, plan.reuse)


class WorkflowProfileChangeTests(unittest.TestCase):
    def test_workflow_profile_change_invalidates_all_scenes(self):
        plan = preview_rebuild(ChangeSet.workflow_profile())
        self.assertIn(ArtifactKind.SCENE_RENDER, plan.rebuild)
        self.assertIn(ArtifactKind.FINAL_VIDEO, plan.rebuild)

    def test_workflow_profile_change_reuses_audio_analysis(self):
        plan = preview_rebuild(ChangeSet.workflow_profile())
        self.assertIn(ArtifactKind.AUDIO_TIMELINE, plan.reuse)

    def test_workflow_profile_change_reuses_reference_sources(self):
        plan = preview_rebuild(ChangeSet.workflow_profile())
        self.assertIn(ArtifactKind.REFERENCE_SOURCES, plan.reuse)


class ReferenceAssignmentChangeTests(unittest.TestCase):
    def test_reference_change_invalidates_affected_scene_render(self):
        plan = preview_rebuild(ChangeSet.references(scene_numbers={1}))
        self.assertIn(ArtifactKind.SCENE_RENDER, plan.rebuild)
        self.assertIn(1, plan.affected_scenes)

    def test_reference_change_reuses_audio(self):
        plan = preview_rebuild(ChangeSet.references(scene_numbers={1}))
        self.assertIn(ArtifactKind.AUDIO_TIMELINE, plan.reuse)


class DimensionsChangeTests(unittest.TestCase):
    def test_dimensions_change_invalidates_all_renders(self):
        plan = preview_rebuild(ChangeSet.dimensions())
        self.assertIn(ArtifactKind.SCENE_RENDER, plan.rebuild)
        self.assertIn(ArtifactKind.FINAL_VIDEO, plan.rebuild)

    def test_dimensions_change_reuses_audio(self):
        plan = preview_rebuild(ChangeSet.dimensions())
        self.assertIn(ArtifactKind.AUDIO_TIMELINE, plan.reuse)


class ReviewOrderingChangeTests(unittest.TestCase):
    def test_review_order_change_invalidates_only_final_video(self):
        plan = preview_rebuild(ChangeSet.review_ordering())
        self.assertIn(ArtifactKind.FINAL_VIDEO, plan.rebuild)

    def test_review_order_change_reuses_scene_renders(self):
        plan = preview_rebuild(ChangeSet.review_ordering())
        self.assertIn(ArtifactKind.SCENE_RENDER, plan.reuse)


class ComputeFreshnessTests(unittest.TestCase):
    def test_current_when_fingerprints_match(self):
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="abc",
            workflow_hash="def",
        )
        self.assertEqual(
            Freshness.CURRENT,
            compute_freshness(fp, prompt_hash="abc", workflow_hash="def"),
        )

    def test_stale_when_prompt_hash_differs(self):
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="abc",
            workflow_hash="def",
        )
        self.assertEqual(
            Freshness.STALE,
            compute_freshness(fp, prompt_hash="changed", workflow_hash="def"),
        )

    def test_stale_when_workflow_hash_differs(self):
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="abc",
            workflow_hash="def",
        )
        self.assertEqual(
            Freshness.STALE,
            compute_freshness(fp, prompt_hash="abc", workflow_hash="new_workflow"),
        )

    def test_unknown_when_provenance_missing(self):
        self.assertEqual(
            Freshness.UNKNOWN,
            compute_freshness(None, prompt_hash="abc", workflow_hash="def"),
        )


class RebuildPlanTests(unittest.TestCase):
    def test_plan_is_frozen(self):
        plan = preview_rebuild(ChangeSet.prompt(scene_numbers={1}))
        with self.assertRaises(Exception):
            plan.rebuild = set()  # type: ignore

    def test_plan_stages_are_ordered(self):
        plan = preview_rebuild(ChangeSet.workflow_profile())
        if plan.stages:
            prev = None
            for stage in plan.stages:
                if prev is not None:
                    self.assertLessEqual(
                        prev.order,
                        stage.order,
                        f"Stage {stage.name} should come before {prev.name}",
                    )
                prev = stage

    def test_plan_unknown_legacy_artifacts(self):
        plan = preview_rebuild(ChangeSet.prompt(scene_numbers={1}))
        self.assertIsInstance(plan.unknown, frozenset)

    def test_empty_change_set_is_noop(self):
        change = ChangeSet.empty()
        plan = preview_rebuild(change)
        self.assertEqual(set(), plan.rebuild)


class ArtifactFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_frozen(self):
        fp = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="a",
            workflow_hash="b",
        )
        with self.assertRaises(Exception):
            fp.prompt_hash = "x"  # type: ignore

    def test_fingerprint_equality(self):
        fp1 = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="a",
            workflow_hash="b",
        )
        fp2 = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="a",
            workflow_hash="b",
        )
        self.assertEqual(fp1, fp2)

    def test_fingerprint_nequality(self):
        fp1 = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="a",
            workflow_hash="b",
        )
        fp2 = ArtifactFingerprint(
            artifact_kind=ArtifactKind.SCENE_RENDER,
            scene_number=1,
            prompt_hash="different",
            workflow_hash="b",
        )
        self.assertNotEqual(fp1, fp2)


class DependencyGraphCycleDetectionTests(unittest.TestCase):
    """Tests for _validate_no_cycles — the standalone DFS cycle checker."""

    def _import_validate(self):
        from feverslop.domain.rebuild_policy import _validate_no_cycles
        return _validate_no_cycles

    def test_acyclic_graph_passes(self):
        validate = self._import_validate()
        from feverslop.domain.rebuild_policy import _ARTIFACT_DEPENDENCIES
        validate(_ARTIFACT_DEPENDENCIES)

    def test_injected_twonode_cycle_detected(self):
        validate = self._import_validate()
        A = ArtifactKind.AUDIO_ANALYSIS
        B = ArtifactKind.BEAT_MARKERS
        graph = {
            A: frozenset({B}),
            B: frozenset({A}),
        }
        with self.assertRaises(ValueError) as ctx:
            validate(graph)
        msg = str(ctx.exception)
        self.assertIn("Circular dependency", msg)
        self.assertIn("audio_analysis", msg)
        self.assertIn("beat_markers", msg)
        cycle_part = msg.split(": ", 1)[-1]
        parts = cycle_part.split(" -> ")
        # Cycle should start and end with same node, no duplicates in between
        self.assertEqual(parts[0], parts[-1])
        # No duplicate consecutive nodes (catches "A -> B -> A -> A" bug)
        for i in range(len(parts) - 1):
            self.assertNotEqual(parts[i], parts[i + 1])

    def test_injected_threenode_cycle_detected(self):
        validate = self._import_validate()
        A = ArtifactKind.AUDIO_ANALYSIS
        B = ArtifactKind.BEAT_MARKERS
        C = ArtifactKind.AUDIO_TIMELINE
        graph = {
            A: frozenset({C}),
            B: frozenset({A}),
            C: frozenset({B}),
        }
        with self.assertRaises(ValueError) as ctx:
            validate(graph)
        msg = str(ctx.exception)
        self.assertIn("Circular dependency", msg)
        cycle_part = msg.split(": ", 1)[-1]
        parts = cycle_part.split(" -> ")
        self.assertEqual(parts[0], parts[-1])
        for i in range(len(parts) - 1):
            self.assertNotEqual(parts[i], parts[i + 1])

    def test_empty_graph_passes(self):
        validate = self._import_validate()
        validate({})

    def test_selfloop_detected(self):
        validate = self._import_validate()
        A = ArtifactKind.AUDIO_ANALYSIS
        graph = {A: frozenset({A})}
        with self.assertRaises(ValueError):
            validate(graph)


class DependencyGraphAcyclicTests(unittest.TestCase):
    def test_dependency_graph_has_no_cycles(self):
        from feverslop.domain.rebuild_policy import _ARTIFACT_DEPENDENCIES
        visited = set()
        for kind in ArtifactKind:
            self._detect_cycle(kind, _ARTIFACT_DEPENDENCIES, set(), visited)

    @staticmethod
    def _detect_cycle(kind, dependencies, stack, visited):
        if kind in stack:
            raise ValueError(f"Cycle detected at {kind}")
        if kind in visited:
            return
        stack.add(kind)
        visited.add(kind)
        for dep in dependencies.get(kind, set()):
            DependencyGraphAcyclicTests._detect_cycle(dep, dependencies, stack, visited)
        stack.remove(kind)


if __name__ == "__main__":
    unittest.main()
