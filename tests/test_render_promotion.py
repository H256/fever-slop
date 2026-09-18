"""Focused tests for explicit replay policy and idempotent render promotion.

These tests exercise the promotion unit in isolation (no live ComfyUI server)
and verify the replay policy is declared, not assumed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.comfyui_render_queue import (
    COMFYUI_REPLAY_POLICY,
    ComfyUIRenderQueue,
    ReplayPolicy,
)
from feverslop.adapters.render_promotion import (
    NOOP,
    PROMOTE,
    RenderPromotion,
    StaleRunError,
    decide_promotion,
    scene_fingerprint,
)
from feverslop.domain.prepared_workflow import SceneWorkflowManifest


def _payload(**overrides):
    base = {
        "schema": "feverslop.scene-workflow/v3",
        "scene": 1,
        "pipeline": "ltx_i2v",
        "workflow": {"path": "workflow.json", "sha256": "a" * 64},
        "template": {"path": "template.json", "sha256": "b" * 64, "external": True},
        "render_plan": {"path": "plan.json", "sha256": "c" * 64},
        "assets": [],
        "seed": 1,
        "fps": 24,
        "frame_count": 25,
        "width": 1280,
        "height": 704,
    }
    base.update(overrides)
    return base


class ReplayPolicyTests(unittest.TestCase):
    def test_comfyui_replay_policy_is_declared_replayable(self):
        self.assertTrue(COMFYUI_REPLAY_POLICY.replayable)
        self.assertIsInstance(COMFYUI_REPLAY_POLICY, ReplayPolicy)
        self.assertTrue(COMFYUI_REPLAY_POLICY.reason)

    def test_render_queue_exposes_declared_replay_policy(self):
        queue = ComfyUIRenderQueue(client=None)
        self.assertIs(queue.replay_policy, COMFYUI_REPLAY_POLICY)


class RenderPromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.manifest_path = self.project / "manifest.json"
        SceneWorkflowManifest.from_dict(_payload()).write(self.manifest_path)

    def test_first_promotion_writes_terminal_record(self):
        manifest = SceneWorkflowManifest.read(self.manifest_path)
        fp = scene_fingerprint(manifest)
        self.assertEqual(PROMOTE, RenderPromotion(self.project).apply(self.manifest_path, current_fingerprint=fp))
        stored = SceneWorkflowManifest.read(self.manifest_path).render_promotion
        self.assertIsNotNone(stored)
        self.assertEqual(fp, stored["fingerprint"])
        self.assertEqual("terminal", stored["status"])
        self.assertEqual(1, stored["scene"])

    def test_reapplying_same_fingerprint_is_noop(self):
        manifest = SceneWorkflowManifest.read(self.manifest_path)
        fp = scene_fingerprint(manifest)
        promotion = RenderPromotion(self.project)
        self.assertEqual(PROMOTE, promotion.apply(self.manifest_path, current_fingerprint=fp))
        self.assertEqual(NOOP, promotion.apply(self.manifest_path, current_fingerprint=fp))

    def test_stale_run_is_refused_with_clear_message(self):
        manifest = SceneWorkflowManifest.read(self.manifest_path)
        fp = scene_fingerprint(manifest)
        RenderPromotion(self.project).apply(self.manifest_path, current_fingerprint=fp)
        with self.assertRaises(StaleRunError) as ctx:
            RenderPromotion(self.project).apply(self.manifest_path, current_fingerprint="f" * 64)
        self.assertIn("Refusing to apply stale render result", str(ctx.exception))

    def test_fingerprint_changes_with_render_inputs(self):
        base = SceneWorkflowManifest.from_dict(_payload())
        self.assertNotEqual(
            scene_fingerprint(base),
            scene_fingerprint(SceneWorkflowManifest.from_dict(_payload(seed=2))),
        )
        self.assertNotEqual(
            scene_fingerprint(base),
            scene_fingerprint(SceneWorkflowManifest.from_dict(_payload(frame_count=30))),
        )
        self.assertEqual(
            scene_fingerprint(base),
            scene_fingerprint(SceneWorkflowManifest.from_dict(_payload())),
        )

    def test_decide_promotion_is_pure(self):
        self.assertEqual(PROMOTE, decide_promotion(None, "x"))
        self.assertEqual(NOOP, decide_promotion({"fingerprint": "x"}, "x"))
        self.assertEqual("stale", decide_promotion({"fingerprint": "y"}, "x"))

    def test_promotion_record_survives_round_trip(self):
        from dataclasses import replace

        manifest = SceneWorkflowManifest.from_dict(_payload())
        updated = replace(manifest, render_promotion={"fingerprint": "z", "status": "terminal"})
        self.assertEqual({"fingerprint": "z", "status": "terminal"}, updated.to_dict()["render_promotion"])
        self.assertEqual(
            {"fingerprint": "z", "status": "terminal"},
            SceneWorkflowManifest.from_dict(updated.to_dict()).render_promotion,
        )


if __name__ == "__main__":
    unittest.main()
