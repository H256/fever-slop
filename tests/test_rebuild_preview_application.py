import unittest

from feverslop.application.rebuild_preview import (
    PreviewRebuildUseCase,
    RebuildPreviewResult,
    RebuildStage,
    RequestRebuildUseCase,
)
from feverslop.domain.rebuild_policy import (
    ArtifactKind,
    ChangeKind,
    ChangeSet,
)
from feverslop.ports.rebuild_execution import RebuildExecutionPort


class _MemoryRebuildExecutor(RebuildExecutionPort):
    def __init__(self) -> None:
        self.requests = []

    def request_rebuild(self, project_id: str, plan) -> str:
        self.requests.append((project_id, plan))
        return f"job-{len(self.requests)}"


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


class RequestRebuildUseCaseTests(unittest.TestCase):
    def test_request_dispatches_plan(self):
        executor = _MemoryRebuildExecutor()
        use_case = RequestRebuildUseCase(executor=executor)
        change = ChangeSet(
            change_kinds=frozenset({ChangeKind.PROMPT}),
            scene_numbers=frozenset({1}),
        )

        job_id = use_case.execute(project_id="test", change=change)

        self.assertEqual(job_id, "job-1")
        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(executor.requests[0][0], "test")

    def test_constructor_rejects_executor_without_request_rebuild(self):
        with self.assertRaisesRegex(TypeError, "request_rebuild"):
            RequestRebuildUseCase(executor=object())
        with self.assertRaisesRegex(TypeError, "request_rebuild"):
            RequestRebuildUseCase(executor=None)

    def test_empty_change_no_rebuild(self):
        use_case = PreviewRebuildUseCase()
        change = ChangeSet.empty()

        result = use_case.execute(change=change)

        self.assertEqual(len(result.stale_artifacts), 0)

    def test_no_rebuild_needed_returns_empty(self):
        executor = _MemoryRebuildExecutor()
        use_case = RequestRebuildUseCase(executor=executor)
        change = ChangeSet.empty()

        job_id = use_case.execute(project_id="test", change=change)

        self.assertEqual(job_id, "")
        self.assertEqual(len(executor.requests), 0)


if __name__ == "__main__":
    unittest.main()
