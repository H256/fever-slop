from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class PipelineStateStoreTests(unittest.TestCase):
    def test_successful_main_pipeline_invalidates_downstream_completion(self):
        from feverslop.studio.pipeline_state_store import PipelineStateStore

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
                        ]
                    }
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


if __name__ == "__main__":
    unittest.main()
