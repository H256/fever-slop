"""Focused tests for the ComfyUI render queue's declared replay policy.

The replay policy is a live, declared contract (not an assumption): the queue
exposes it explicitly so callers can inspect whether a terminal result is
replayable. These tests exercise that declaration in isolation (no live
ComfyUI server).
"""

from __future__ import annotations

import unittest

from feverslop.adapters.comfyui_render_queue import (
    COMFYUI_REPLAY_POLICY,
    ComfyUIRenderQueue,
    ReplayPolicy,
)


class ReplayPolicyTests(unittest.TestCase):
    def test_comfyui_replay_policy_is_declared_replayable(self):
        self.assertTrue(COMFYUI_REPLAY_POLICY.replayable)
        self.assertIsInstance(COMFYUI_REPLAY_POLICY, ReplayPolicy)
        self.assertTrue(COMFYUI_REPLAY_POLICY.reason)

    def test_render_queue_exposes_declared_replay_policy(self):
        queue = ComfyUIRenderQueue(client=None)
        self.assertIs(queue.replay_policy, COMFYUI_REPLAY_POLICY)


if __name__ == "__main__":
    unittest.main()
