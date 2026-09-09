import unittest

from feverslop.application.rebuild_preview import (
    PreviewRebuildUseCase,
    RebuildPreviewResult,
    RebuildStage,
)
from feverslop.domain.rebuild_policy import (
    ArtifactKind,
    ChangeKind,
    ChangeSet,
)
class PreviewRebuildUseCaseTests(unittest.TestCase):
    def test_prompt_change_affects_prompt_generation(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet(
            change_kinds=frozenset({ChangeKind.PROMPT}),
            scene_numbers=frozenset({2, 3}),
        )

        result = use_case.execute(change=change)

        self.assertIsInstance(result, RebuildPreviewResult)
        self.assertIn(RebuildStage.PLANNING, result.stages)
        self.assertIn(RebuildStage.RENDER, result.stages)
        self.assertEqual(result.affected_scenes, frozenset({2, 3}))
        self.assertTrue(any(a.kind == ArtifactKind.PROMPT_GENERATION for a in result.stale_artifacts))

    def test_global_prompt_change_has_no_scenes(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet(
            change_kinds=frozenset({ChangeKind.PROMPT}),
            scene_numbers=None,
        )

        result = use_case.execute(change=change)

        self.assertIn(RebuildStage.PLANNING, result.stages)
        self.assertIn(RebuildStage.RENDER, result.stages)
        self.assertEqual(result.affected_scenes, frozenset())

    def test_timeline_change_affects_assembly(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet(
            change_kinds=frozenset({ChangeKind.TIMELINE}),
            scene_numbers=frozenset({5}),
        )

        result = use_case.execute(change=change)

        self.assertIn(RebuildStage.RENDER, result.stages)
        self.assertTrue(any(a.kind == ArtifactKind.AUDIO_TIMELINE for a in result.stale_artifacts))
        self.assertTrue(any(a.kind == ArtifactKind.FINAL_VIDEO for a in result.reusable_artifacts))
        self.assertEqual(result.affected_scenes, frozenset({5}))

    def test_reference_change_affects_references_and_downstream(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet(
            change_kinds=frozenset({ChangeKind.REFERENCE_ASSIGNMENT}),
            scene_numbers=frozenset({1}),
        )

        result = use_case.execute(change=change)

        self.assertIn(RebuildStage.REFERENCES, result.stages)
        self.assertIn(RebuildStage.RENDER, result.stages)
        self.assertTrue(any(a.kind == ArtifactKind.REFERENCE_SHEETS for a in result.stale_artifacts))

    def test_workflow_profile_affects_all(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet(change_kinds=frozenset({ChangeKind.WORKFLOW_PROFILE}))

        result = use_case.execute(change=change)

        self.assertIn(RebuildStage.PLANNING, result.stages)
        self.assertIn(RebuildStage.REFERENCES, result.stages)
        self.assertIn(RebuildStage.RENDER, result.stages)
        self.assertIn(RebuildStage.FINAL, result.stages)
        self.assertTrue(any(a.kind == ArtifactKind.AUDIO_TIMELINE for a in result.reusable_artifacts))

    def test_dimensions_change_keeps_audio_and_references(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet(change_kinds=frozenset({ChangeKind.DIMENSIONS}))

        result = use_case.execute(change=change)

        self.assertIn(RebuildStage.RENDER, result.stages)
        self.assertTrue(any(a.kind == ArtifactKind.AUDIO_TIMELINE for a in result.reusable_artifacts))
        self.assertTrue(any(a.kind == ArtifactKind.REFERENCE_SOURCES for a in result.reusable_artifacts))


if __name__ == "__main__":
    unittest.main()
