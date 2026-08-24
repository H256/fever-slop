from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.effective_render_plan import project_effective_plan
from feverslop.composition.resume_plan import build_resume_plan
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.execution_plan import ExecutionPlan, ExecutionPlanItem, PlanAction
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.domain.project_render_settings import (
    ProjectRenderSettings,
    WorkflowSelection,
)
from feverslop.scene_artifacts import SceneArtifactLayout


class ExecutionPlanTests(unittest.TestCase):
    def test_plan_exposes_stable_runnable_stages_and_scene_union(self):
        plan = ExecutionPlan(
            project=Path("project"),
            mode="resume",
            items=(
                ExecutionPlanItem("projection", PlanAction.RUN, "prompt changed", 2, "msr_prompt_enrich"),
                ExecutionPlanItem("projection", PlanAction.REUSE, "fingerprint matches", 1, "msr_prompt_enrich"),
                ExecutionPlanItem("prepare", PlanAction.RUN, "workflow stale", 2, "ltx_prepare_workflows"),
                ExecutionPlanItem("render", PlanAction.RUN, "clip missing", 2, "ltx_render_scenes"),
                ExecutionPlanItem("assembly", PlanAction.RUN, "scene changed", None, "concat_video_only"),
            ),
        )

        self.assertFalse(plan.blocked)
        self.assertEqual(
            ("msr_prompt_enrich", "ltx_prepare_workflows", "ltx_render_scenes", "concat_video_only"),
            plan.runnable_stages,
        )
        self.assertEqual((2,), plan.runnable_scenes)

    def test_blocked_plan_has_no_runnable_work(self):
        plan = ExecutionPlan(
            project=Path("project"),
            mode="resume",
            items=(ExecutionPlanItem("canonical", PlanAction.BLOCKED, "run plan-migrate"),),
        )

        self.assertTrue(plan.blocked)
        self.assertEqual((), plan.runnable_stages)
        self.assertEqual((), plan.runnable_scenes)

    def test_resume_stages_are_dependency_ordered_across_scenes(self):
        plan = ExecutionPlan(
            project=Path("project"),
            mode="resume",
            items=(
                ExecutionPlanItem("projection", PlanAction.RUN, "prompt", 1, "msr_prompt_enrich"),
                ExecutionPlanItem("references", PlanAction.RUN, "binding", 2, "msr_references"),
                ExecutionPlanItem("sheets", PlanAction.RUN, "binding", 2, "msr_reference_sheets"),
            ),
        )

        self.assertEqual(
            ("msr_references", "msr_reference_sheets", "msr_prompt_enrich"),
            plan.runnable_stages,
        )
        self.assertEqual((2,), plan.runnable_scenes_for_stage("msr_references"))
        self.assertEqual((1,), plan.runnable_scenes_for_stage("msr_prompt_enrich"))


class ResumePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name)
        self.layout = SceneArtifactLayout(self.project)
        self.layout.plans_dir.mkdir(parents=True)
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3",
            "video_pipeline": "ltx_msr",
        }), encoding="utf-8")
        (self.project / "song.mp3").write_bytes(b"audio")

    def tearDown(self):
        self.temp.cleanup()

    def _scene(self, number: int) -> dict:
        scene = {
            "scene": number,
            "segment_id": f"segment-{number}",
            "fps": 24,
            "frame_count": 25,
            "width": 1280,
            "height": 704,
            "ltx": {"base_prompt": f"prompt {number}", "msr_prompt_relay": f"relay {number}"},
            "references": {"actor_ids": ["hero"], "location_id": "room"},
        }
        scene["canonical"] = build_canonical_scene(
            segment_id=scene["segment_id"],
            generated_roles={
                PromptRole.LTX_BASE: scene["ltx"]["base_prompt"],
                PromptRole.LTX_MSR_RELAY: scene["ltx"]["msr_prompt_relay"],
            },
            provenance_source="test",
        )
        return scene

    def _write_base_and_derived(self, count: int = 2) -> list[dict]:
        base = [self._scene(number) for number in range(1, count + 1)]
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")
        self.layout.anchored_plan.write_text(json.dumps(base), encoding="utf-8")
        derived = project_effective_plan(base, base)
        self.layout.references_plan.write_text(json.dumps(derived), encoding="utf-8")
        return base

    def _prepare(self, scene: dict, *, pipeline: str = "ltx_msr") -> None:
        projected = project_effective_plan([scene], [scene])[0]
        dependencies = projected["canonical_projection"]["dependencies"]
        workflow = self.layout.scene_workflow(int(scene["scene"]))
        template = self.project / "template.json"
        workflow.parent.mkdir(parents=True, exist_ok=True)
        workflow.write_text("{}", encoding="utf-8")
        template.write_text("{}", encoding="utf-8")
        actor = self.project / "actor.png"
        location = self.project / "location.png"
        actor.write_bytes(b"actor")
        location.write_bytes(b"location")
        assets = [] if pipeline != "ltx_msr" else [
            ("actor_sheet", actor, "actor.png", "hero"),
            ("location_sheet", location, "location.png", "room"),
        ]
        SceneWorkflowManifest.create(
            project_dir=self.project,
            scene=int(scene["scene"]),
            pipeline=pipeline,
            workflow_path=workflow,
            template_path=template,
            render_plan_path=(self.layout.references_plan if pipeline == "ltx_msr" else self.layout.base_plan),
            assets=assets,
            seed=1,
            fps=24,
            frame_count=25,
            width=1280,
            height=704,
            canonical_dependencies=__import__(
                "feverslop.domain.effective_render_plan", fromlist=["CanonicalSceneDependencies"],
            ).CanonicalSceneDependencies.from_dict(dependencies),
        ).write(self.layout.scene_manifest(int(scene["scene"])))

    def _tree_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.project.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.project).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def test_missing_canonical_plan_runs_main_pipeline(self):
        plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

        self.assertEqual(PlanAction.RUN, plan.items[0].action)
        self.assertEqual("main_pipeline", plan.items[0].stage)

    def test_unknown_legacy_edit_blocks_with_repair_command(self):
        base = self._write_base_and_derived(1)
        edited = project_effective_plan(base, base)
        edited[0]["ltx"]["base_prompt"] = "unowned legacy edit"
        self.layout.references_plan.write_text(json.dumps(edited), encoding="utf-8")

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

        self.assertTrue(plan.blocked)
        self.assertIn("plan-migrate", " ".join(item.reason for item in plan.items))

    def test_malformed_override_fails_closed_as_blocked_plan(self):
        base = self._write_base_and_derived(1)
        base[0]["canonical"]["roles"][str(PromptRole.LTX_BASE)]["override"] = {}
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

        self.assertTrue(plan.blocked)
        self.assertIn("plan validate", plan.items[0].reason)

    def test_prompt_change_only_runs_affected_scene_downstream(self):
        base = self._write_base_and_derived()
        for scene in base:
            self._prepare(scene)
            self.layout.scene_final_video(int(scene["scene"])).write_bytes(b"clip")
        base[1]["canonical"]["roles"][str(PromptRole.LTX_BASE)]["override"] = {
            "value": "human prompt",
            "provenance": {"source": "human"},
        }
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")
        before = self._tree_hash()

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

        self.assertEqual((2,), plan.runnable_scenes)
        self.assertIn("msr_prompt_enrich", plan.runnable_stages)
        self.assertIn("ltx_prepare_workflows", plan.runnable_stages)
        self.assertIn("ltx_render_scenes", plan.runnable_stages)
        self.assertIn("concat_video_only", plan.runnable_stages)
        scene_one = [item for item in plan.items if item.scene == 1]
        self.assertTrue(all(item.action is PlanAction.REUSE for item in scene_one))
        self.assertEqual(before, self._tree_hash())

    def test_stale_workflow_requires_prepare_before_render(self):
        base = self._write_base_and_derived(1)
        self._prepare(base[0])
        self.layout.scene_workflow(1).write_text('{"changed": true}', encoding="utf-8")
        self.layout.scene_final_video(1).write_bytes(b"clip")

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

        prepare = next(item for item in plan.items if item.phase == "prepare")
        render = next(item for item in plan.items if item.phase == "render")
        self.assertEqual(PlanAction.RUN, prepare.action)
        self.assertEqual(PlanAction.RUN, render.action)
        self.assertLess(plan.items.index(prepare), plan.items.index(render))

    def test_reference_binding_change_runs_reference_assets_before_projection(self):
        base = self._write_base_and_derived(1)
        self._prepare(base[0])
        self.layout.scene_final_video(1).write_bytes(b"clip")
        base[0]["references"]["actor_ids"] = ["replacement"]
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

        self.assertEqual(
            (
                "msr_references",
                "msr_reference_sheets",
                "msr_prompt_enrich",
                "ltx_prepare_workflows",
                "ltx_render_scenes",
                "concat_video_only",
                "mux_original_audio",
                "export_timeline",
            ),
            plan.runnable_stages,
        )

    def test_ingredients_reference_change_runs_msr_enrichment_before_sheet_projection(self):
        base = self._write_base_and_derived(1)
        derived = project_effective_plan(base, base)
        self.layout.ingredients_plan.write_text(json.dumps(derived), encoding="utf-8")
        base[0]["references"]["location_id"] = "replacement-room"
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")

        plan = build_resume_plan(self.project, video_pipeline="ltx_ingredients")

        stages = plan.runnable_stages
        self.assertLess(stages.index("msr_reference_sheets"), stages.index("msr_prompt_enrich"))
        self.assertLess(stages.index("msr_prompt_enrich"), stages.index("ingredients_sheets"))

    def test_timing_and_resolution_changes_invalidate_only_changed_scene(self):
        for field, value in (("frame_count", 49), ("width", 1024), ("height", 576)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as isolated:
                self.project = Path(isolated)
                self.layout = SceneArtifactLayout(self.project)
                self.layout.plans_dir.mkdir(parents=True)
                (self.project / "config.json").write_text(json.dumps({
                    "input_audio": "song.mp3", "video_pipeline": "ltx_msr",
                }), encoding="utf-8")
                (self.project / "song.mp3").write_bytes(b"audio")
                base = self._write_base_and_derived(2)
                for scene in base:
                    self._prepare(scene)
                    self.layout.scene_final_video(int(scene["scene"])).write_bytes(b"clip")
                base[1][field] = value
                self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")

                plan = build_resume_plan(self.project, video_pipeline="ltx_msr")

                self.assertEqual((2,), plan.runnable_scenes)
                self.assertNotIn("msr_references", plan.runnable_stages)
                self.assertNotIn("msr_reference_sheets", plan.runnable_stages)
                self.assertIn("ltx_prepare_workflows", plan.runnable_stages)

    def test_project_resolution_change_syncs_canonical_plan_and_rerenders_all_scenes(self):
        base = self._write_base_and_derived(2)
        for scene in base:
            self._prepare(scene)
            self.layout.scene_final_video(int(scene["scene"])).write_bytes(b"clip")

        plan = build_resume_plan(
            self.project,
            video_pipeline="ltx_msr",
            render_settings=ProjectRenderSettings(width=1024, height=576),
        )

        self.assertEqual("sync_project_settings", plan.runnable_stages[0])
        self.assertEqual((1, 2), plan.runnable_scenes_for_stage("ltx_prepare_workflows"))
        self.assertEqual((1, 2), plan.runnable_scenes_for_stage("ltx_render_scenes"))
        self.assertNotIn("main_pipeline", plan.runnable_stages)
        self.assertNotIn("h3_prompts", plan.runnable_stages)

    def test_project_video_workflow_change_invalidates_workflows_not_references(self):
        base = self._write_base_and_derived(1)
        self._prepare(base[0])
        self.layout.scene_final_video(1).write_bytes(b"clip")
        workflow = self.project / "custom-video.json"
        workflow.write_text('{"steps": 8}', encoding="utf-8")

        plan = build_resume_plan(
            self.project,
            video_pipeline="ltx_msr",
            render_settings=ProjectRenderSettings(
                width=1280,
                height=704,
                video_workflow=WorkflowSelection.from_path(workflow, root=self.project),
            ),
        )

        self.assertIn("sync_project_settings", plan.runnable_stages)
        self.assertIn("ltx_prepare_workflows", plan.runnable_stages)
        self.assertIn("ltx_render_scenes", plan.runnable_stages)
        self.assertNotIn("msr_references", plan.runnable_stages)
        self.assertNotIn("msr_reference_sheets", plan.runnable_stages)

    def test_project_reference_workflow_change_refreshes_h3_reference_dependents(self):
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3", "video_pipeline": "minimax-h3-r2v",
        }), encoding="utf-8")
        scene = self._scene(1)
        scene["canonical"]["roles"][str(PromptRole.H3_VIDEO)] = {
            "generated": {"value": "h3", "provenance": {"input_fingerprint": "fp-1"}},
        }
        base = project_effective_plan([scene], [scene])
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")
        self.layout.anchored_plan.write_text(json.dumps(base), encoding="utf-8")
        checkpoint = self.layout.scene_h3_prompt(1)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"status": "good", "input_fingerprint": "fp-1"}), encoding="utf-8")
        self.layout.scene_final_video(1).write_bytes(b"clip")
        self._prepare(base[0], pipeline="minimax-h3-r2v")
        hero = self.project / "hero.json"
        edit = self.project / "edit.json"
        hero.write_text("hero", encoding="utf-8")
        edit.write_text("edit", encoding="utf-8")

        plan = build_resume_plan(
            self.project,
            video_pipeline="minimax-h3-r2v",
            render_settings=ProjectRenderSettings(
                width=1280,
                height=704,
                reference_hero_workflow=WorkflowSelection.from_path(hero, root=self.project),
                reference_edit_workflow=WorkflowSelection.from_path(edit, root=self.project),
            ),
        )

        stages = plan.runnable_stages
        self.assertEqual("sync_project_settings", stages[0])
        self.assertIn("msr_references", stages)
        self.assertIn("msr_reference_sheets", stages)
        self.assertIn("h3_prompts", stages)
        self.assertIn("render_plan", stages)
        self.assertIn("ltx_render_scenes", stages)


    def test_scene_selection_marks_other_scenes_not_selected(self):
        self._write_base_and_derived(2)

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr", selected_scenes={2})

        excluded = next(item for item in plan.items if item.scene == 1)
        self.assertEqual(PlanAction.NOT_SELECTED, excluded.action)
        self.assertEqual((2,), plan.runnable_scenes)

    def test_unknown_scene_selection_blocks_before_assembly(self):
        self._write_base_and_derived(2)

        plan = build_resume_plan(self.project, video_pipeline="ltx_msr", selected_scenes={99})

        self.assertTrue(plan.blocked)
        self.assertEqual((), plan.runnable_stages)
        self.assertIn("99", plan.items[0].reason)

    def test_classic_i2v_clip_without_dependency_provenance_is_not_reused(self):
        base = [self._scene(1)]
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")
        self.layout.scene_final_video(1).parent.mkdir(parents=True, exist_ok=True)
        self.layout.scene_final_video(1).write_bytes(b"unproven clip")

        plan = build_resume_plan(self.project, video_pipeline="ltx_i2v")

        render = next(item for item in plan.items if item.phase == "render")
        self.assertEqual(PlanAction.RUN, render.action)
        self.assertIn("provenance", render.reason)

    def test_valid_h3_checkpoint_and_clip_are_reused_individually(self):
        (self.project / "config.json").write_text(json.dumps({
            "input_audio": "song.mp3", "video_pipeline": "minimax-h3-r2v",
        }), encoding="utf-8")
        base = [self._scene(1), self._scene(2)]
        for scene in base:
            scene["canonical"]["roles"][str(PromptRole.H3_VIDEO)] = {
                "generated": {"value": f"h3 {scene['scene']}", "provenance": {"input_fingerprint": f"fp-{scene['scene']}"}},
            }
        base = project_effective_plan(base, base)
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")
        self.layout.anchored_plan.write_text(json.dumps(base), encoding="utf-8")
        for number in (1, 2):
            checkpoint = self.layout.scene_h3_prompt(number)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"status": "good", "input_fingerprint": f"fp-{number}"}), encoding="utf-8")
            self.layout.scene_final_video(number).write_bytes(b"clip")
        self._prepare(base[0], pipeline="minimax-h3-r2v")
        self.layout.scene_h3_prompt(2).unlink()

        plan = build_resume_plan(self.project, video_pipeline="minimax-h3-r2v")

        h3 = {item.scene: item.action for item in plan.items if item.phase == "h3 prompts"}
        self.assertEqual(PlanAction.REUSE, h3[1])
        self.assertEqual(PlanAction.RUN, h3[2])
        render = {item.scene: item.action for item in plan.items if item.phase == "render"}
        self.assertEqual(PlanAction.REUSE, render[1])
        self.assertEqual(PlanAction.RUN, render[2])

    def test_h3_r2v_reference_change_refreshes_reference_pipeline(self):
        base = project_effective_plan([self._scene(1)], [self._scene(1)])
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")
        self.layout.anchored_plan.write_text(json.dumps(base), encoding="utf-8")
        base[0]["references"]["actor_ids"] = ["replacement"]
        self.layout.base_plan.write_text(json.dumps(base), encoding="utf-8")

        plan = build_resume_plan(self.project, video_pipeline="minimax-h3-r2v")

        self.assertIn("msr_references", plan.runnable_stages)
        self.assertIn("msr_reference_sheets", plan.runnable_stages)

    def test_human_h3_override_does_not_require_generated_checkpoint(self):
        scene = self._scene(1)
        scene["canonical"]["roles"][str(PromptRole.H3_VIDEO)] = {
            "generated": {"value": "generated", "provenance": {"input_fingerprint": "old"}},
            "override": {"value": "human", "provenance": {"source": "human"}},
        }
        scene = project_effective_plan([scene], [scene])[0]
        self.layout.base_plan.write_text(json.dumps([scene]), encoding="utf-8")
        self.layout.anchored_plan.write_text(json.dumps([scene]), encoding="utf-8")

        plan = build_resume_plan(self.project, video_pipeline="minimax-h3-t2v")

        h3 = next(item for item in plan.items if item.phase == "h3 prompts")
        self.assertEqual(PlanAction.REUSE, h3.action)
        self.assertIn("override", h3.reason)


if __name__ == "__main__":
    unittest.main()
