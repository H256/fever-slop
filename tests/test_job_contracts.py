import unittest
from dataclasses import FrozenInstanceError

from feverslop.application.job_contracts import (
    JobLogEvent,
    JobHandler,
    JobRuntime,
    JobSnapshot,
    JobStatus,
    JobSubmission,
)


class _MemoryJobRuntime:
    def __init__(self):
        self.jobs = {}
        self.next_id = 1
        self.shutdown_calls = []

    def submit(self, submission: JobSubmission, handler: JobHandler) -> str:
        job_id = f"job-{self.next_id}"
        self.next_id += 1
        logs = []
        snapshot = JobSnapshot(
            job_id=job_id,
            project_id=submission.project_id,
            action=submission.action,
            status=JobStatus.RUNNING,
        )
        self.jobs[job_id] = snapshot
        try:
            result = handler(logs.append)
        except Exception as exc:
            snapshot = JobSnapshot(
                **{**snapshot.__dict__, "status": JobStatus.FAILED, "error": str(exc)}
            )
        else:
            events = tuple(
                JobLogEvent(job_id=job_id, message=message, sequence=index)
                for index, message in enumerate(logs, start=1)
            )
            snapshot = JobSnapshot(
                **{**snapshot.__dict__, "status": JobStatus.SUCCEEDED, "result": result, "logs": events}
            )
        self.jobs[job_id] = snapshot
        return job_id

    def get(self, job_id: str) -> JobSnapshot:
        return self.jobs[job_id]

    def list(self, project_id: str | None = None) -> tuple[JobSnapshot, ...]:
        snapshots = tuple(self.jobs.values())
        if project_id is None:
            return snapshots
        return tuple(snapshot for snapshot in snapshots if snapshot.project_id == project_id)

    def cancel(self, job_id: str) -> bool:
        snapshot = self.jobs[job_id]
        if snapshot.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return False
        self.jobs[job_id] = JobSnapshot(
            **{**snapshot.__dict__, "status": JobStatus.CANCELLED}
        )
        return True

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_pending))


class JobContractsTests(unittest.TestCase):
    def test_submission_and_snapshot_are_transport_neutral_value_objects(self):
        submission = JobSubmission(
            project_id="demo",
            action="render",
            project_type="standard_music_video",
            pipeline_mode="classic",
        )
        event = JobLogEvent(job_id="job-1", message="started", sequence=1)
        snapshot = JobSnapshot(
            job_id="job-1",
            project_id="demo",
            action="render",
            status=JobStatus.RUNNING,
            logs=(event,),
        )

        self.assertEqual("demo", submission.project_id)
        self.assertEqual(JobStatus.RUNNING, snapshot.status)
        self.assertEqual((event,), snapshot.logs)
        with self.assertRaises(FrozenInstanceError):
            snapshot.status = JobStatus.SUCCEEDED

    def test_runtime_protocol_exposes_lifecycle_operations(self):
        runtime = _MemoryJobRuntime()
        self.assertIsInstance(runtime, JobRuntime)
        self.assertEqual(
            {"submit", "get", "list", "cancel", "shutdown"},
            set(JobRuntime.__protocol_attrs__),
        )

    def test_in_memory_runtime_covers_success_logs_failure_and_shutdown(self):
        runtime = _MemoryJobRuntime()
        submission = JobSubmission(project_id="demo", action="render")

        job_id = runtime.submit(
            submission,
            lambda log: (log("started"), "artifact.json")[1],
        )
        snapshot = runtime.get(job_id)
        self.assertEqual(JobStatus.SUCCEEDED, snapshot.status)
        self.assertEqual("artifact.json", snapshot.result)
        self.assertEqual(("started",), tuple(event.message for event in snapshot.logs))

        failed_id = runtime.submit(submission, lambda log: 1 / 0)
        self.assertEqual(JobStatus.FAILED, runtime.get(failed_id).status)
        self.assertIn("division by zero", runtime.get(failed_id).error)

        runtime.jobs["pending"] = JobSnapshot(
            job_id="pending",
            project_id="demo",
            action="render",
            status=JobStatus.QUEUED,
        )
        self.assertTrue(runtime.cancel("pending"))
        self.assertEqual(JobStatus.CANCELLED, runtime.get("pending").status)
        runtime.shutdown(wait=False, cancel_pending=True)
        self.assertEqual([(False, True)], runtime.shutdown_calls)


if __name__ == "__main__":
    unittest.main()
