import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.sync_project_render_settings import (
    sync_project_render_settings,
)
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.project_render_settings import ProjectRenderSettings
from feverslop.composition.arg_parser import PipelineStage
from feverslop.composition.stage_runners import STAGE_RUNNERS
from feverslop.scene_artifacts import SceneArtifactLayout


class SyncProjectRenderSettingsTests(unittest.TestCase):
    def test_registered_stage_materializes_settings_from_runner_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            scene = {
                "scene": 1,
                "segment_id": "segment-1",
                "width": 1280,
                "height": 704,
                "canonical": build_canonical_scene(
                    segment_id="segment-1",
                    generated_roles={PromptRole.LTX_BASE: "generated"},
                    provenance_source="test",
                ),
            }
            layout.base_plan.write_text(json.dumps([scene]), encoding="utf-8")
            state = SimpleNamespace(
                args=Namespace(
                    project_render_settings=ProjectRenderSettings(width=1024, height=576),
                ),
                context=SimpleNamespace(
                    project_config_dir=project,
                    artifact_layout=layout,
                ),
                plan_for_next_step=layout.base_plan,
            )

            STAGE_RUNNERS[PipelineStage.SYNC_PROJECT_SETTINGS](state)

            updated = json.loads(layout.base_plan.read_text(encoding="utf-8"))
            self.assertEqual((1024, 576), (updated[0]["width"], updated[0]["height"]))

    def test_updates_only_canonical_plan_and_preserves_human_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            layout = SceneArtifactLayout(project)
            layout.plans_dir.mkdir(parents=True)
            scene = {
                "scene": 1,
                "segment_id": "segment-1",
                "width": 1280,
                "height": 704,
                "canonical": build_canonical_scene(
                    segment_id="segment-1",
                    generated_roles={PromptRole.LTX_BASE: "generated"},
                    provenance_source="test",
                ),
            }
            scene["canonical"]["roles"][str(PromptRole.LTX_BASE)]["override"] = {
                "value": "human",
                "provenance": {"source": "human"},
            }
            layout.base_plan.write_text(json.dumps([scene]), encoding="utf-8")
            layout.references_plan.write_text(json.dumps([{"derived": True}]), encoding="utf-8")
            derived_before = layout.references_plan.read_bytes()

            changed = sync_project_render_settings(
                CanonicalPlanStore(project),
                ProjectRenderSettings(width=1024, height=576),
            )

            updated = json.loads(layout.base_plan.read_text(encoding="utf-8"))
            self.assertTrue(changed)
            self.assertEqual((1024, 576), (updated[0]["width"], updated[0]["height"]))
            self.assertEqual(
                "human",
                updated[0]["canonical"]["roles"][str(PromptRole.LTX_BASE)]["override"]["value"],
            )
            self.assertEqual(derived_before, layout.references_plan.read_bytes())

            unchanged = sync_project_render_settings(
                CanonicalPlanStore(project),
                ProjectRenderSettings(width=1024, height=576),
            )

            self.assertFalse(unchanged)


if __name__ == "__main__":
    unittest.main()
