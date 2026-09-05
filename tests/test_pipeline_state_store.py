from __future__ import annotations

import gc
import json
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters import artifact_locking
from feverslop.adapters.pipeline_state_store import PipelineStateStore


class PipelineStateStoreTests(unittest.TestCase):
    def test_path_locks_are_released_when_no_run_uses_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock_key = str((root / ".studio" / "pipeline_state.json").resolve())
            lock_references = []

            def read_json(path: Path):
                lock_references.append(weakref.ref(artifact_locking._LOCKS[lock_key]))
                return {}

            store = PipelineStateStore(lambda _project_id: root, read_json)
            store.record_pipeline_run("demo", action="first", stages=[], status="succeeded")
            gc.collect()

            self.assertEqual(1, len(lock_references))
            self.assertIsNone(lock_references[0]())
            self.assertNotIn(lock_key, artifact_locking._LOCKS)

    def test_successful_main_pipeline_invalidates_downstream_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_path = root / ".studio" / "pipeline_state.json"
            state_path.parent.mkdir()
            state_path.write_text(
                json.dumps(
                    {
                        "completed_stages": [
                            "main-pipeline",
                            "main_pipeline",
                            "relay_compact",
                            "anchor_fix",
                            "storyboard_frames",
                            "storyboard_page",
                            "msr-references",
                            "msr_references",
                            "msr_reference_sheets",
                            "msr_prompt_enrich",
                            "ingredients_sheets",
                            "ltx_prepare_workflows",
                            "ltx_render_scenes",
                            "concat_video_only",
                            "mux_original_audio",
                            "diagnostic_scene_audio_concat",
                            "facefix",
                            "facefix_concat",
                        ],
                    },
                ),
                encoding="utf-8",
            )
            store = PipelineStateStore(
                lambda _project_id: root,
                lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

            state = store.record_pipeline_run(
                "project",
                action="main-pipeline",
                stages=["main-pipeline"],
                status="succeeded",
            )

        self.assertEqual(["main-pipeline"], state["completed_stages"])

    def test_concurrent_successful_runs_preserve_both_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            start = threading.Barrier(3)

            def read_json(path: Path):
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                time.sleep(0.05)
                return value

            store = PipelineStateStore(lambda _project_id: root, read_json)

            def record(action: str, stage: str) -> None:
                start.wait()
                store.record_pipeline_run(
                    "demo",
                    action=action,
                    stages=[stage],
                    status="succeeded",
                )

            threads = [
                threading.Thread(target=record, args=("first", "stage-one")),
                threading.Thread(target=record, args=("second", "stage-two")),
            ]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(2.0)

            state = json.loads(
                (root / ".studio" / "pipeline_state.json").read_text(encoding="utf-8"),
            )

        self.assertEqual({"stage-one", "stage-two"}, set(state["completed_stages"]))
        self.assertEqual({"first", "second"}, {run["action"] for run in state["runs"]})

    def test_separate_store_instances_share_lock_for_same_state_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            start = threading.Barrier(3)

            def read_json(path: Path):
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                time.sleep(0.05)
                return value

            (root / "alias").mkdir()
            stores = [
                PipelineStateStore(lambda _project_id: root, read_json),
                PipelineStateStore(lambda _project_id: root / "alias" / "..", read_json),
            ]

            def record(store: PipelineStateStore, action: str, stage: str) -> None:
                start.wait()
                store.record_pipeline_run(
                    "demo",
                    action=action,
                    stages=[stage],
                    status="succeeded",
                )

            threads = [
                threading.Thread(target=record, args=(stores[0], "first", "stage-one")),
                threading.Thread(target=record, args=(stores[1], "second", "stage-two")),
            ]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(2.0)

            state = json.loads(
                (root / ".studio" / "pipeline_state.json").read_text(encoding="utf-8"),
            )

        self.assertEqual({"stage-one", "stage-two"}, set(state["completed_stages"]))
        self.assertEqual({"first", "second"}, {run["action"] for run in state["runs"]})

    def test_failed_write_preserves_existing_state_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_path = root / ".studio" / "pipeline_state.json"
            state_path.parent.mkdir(parents=True)
            original = '{"completed_stages": ["existing"]}\n'
            state_path.write_text(original, encoding="utf-8")
            store = PipelineStateStore(
                lambda _project_id: root,
                lambda path: json.loads(path.read_text(encoding="utf-8")),
            )
            real_write_text = Path.write_text

            def partial_write_then_fail(path: Path, *_args, **_kwargs):
                real_write_text(path, "partial", encoding="utf-8")
                raise OSError("disk full")

            with patch.object(Path, "write_text", partial_write_then_fail):
                with self.assertRaisesRegex(OSError, "disk full"):
                    store.record_pipeline_run(
                        "demo",
                        action="new",
                        stages=["new-stage"],
                        status="succeeded",
                    )

            self.assertEqual(original, state_path.read_text(encoding="utf-8"))
            self.assertEqual([state_path], list(state_path.parent.iterdir()))


if __name__ == "__main__":
    unittest.main()
