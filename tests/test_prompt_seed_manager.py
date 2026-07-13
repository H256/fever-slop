from __future__ import annotations

import unittest

from feverslop.domain.prompt_seed_manager import PromptSeedManager


class PromptSeedManagerTest(unittest.TestCase):
    def test_global_seed_returns_seed(self):
        manager = PromptSeedManager(global_seed=42)
        self.assertEqual(manager.get_global_seed(), 42)

    def test_per_prompt_seeds_differ(self):
        manager = PromptSeedManager(global_seed=42)
        seeds = [manager.get_seed_for_prompt(i) for i in range(10)]
        self.assertEqual(len(set(seeds)), 10)

    def test_changing_global_seed_changes_all_seeds(self):
        manager1 = PromptSeedManager(global_seed=0)
        manager2 = PromptSeedManager(global_seed=1)
        for i in range(10):
            self.assertNotEqual(
                manager1.get_seed_for_prompt(i),
                manager2.get_seed_for_prompt(i),
                f"Seeds should differ at index {i}",
            )

    def test_seeds_are_reproducible(self):
        manager = PromptSeedManager(global_seed=42)
        seed_0_a = manager.get_seed_for_prompt(0)
        seed_0_b = manager.get_seed_for_prompt(0)
        self.assertEqual(seed_0_a, seed_0_b)
        self.assertEqual(manager.get_seed_for_prompt(5), manager.get_seed_for_prompt(5))

    def test_seeds_within_range(self):
        manager = PromptSeedManager(global_seed=42)
        for i in range(1000):
            seed = manager.get_seed_for_prompt(i)
            self.assertLessEqual(seed, PromptSeedManager.MAX_SEED)
            self.assertGreaterEqual(seed, 0)

    def test_global_seed_wraps_at_max(self):
        manager = PromptSeedManager(global_seed=PromptSeedManager.MAX_SEED + 100)
        self.assertLessEqual(manager.get_global_seed(), PromptSeedManager.MAX_SEED)

    def test_default_global_seed_is_zero(self):
        manager = PromptSeedManager()
        self.assertEqual(manager.get_global_seed(), 0)

    def test_seeds_are_deterministic_across_instances(self):
        manager1 = PromptSeedManager(global_seed=123)
        manager2 = PromptSeedManager(global_seed=123)
        for i in range(50):
            self.assertEqual(
                manager1.get_seed_for_prompt(i),
                manager2.get_seed_for_prompt(i),
            )

    def test_negative_global_seed_wraps(self):
        manager = PromptSeedManager(global_seed=-1)
        self.assertGreaterEqual(manager.get_global_seed(), 0)


if __name__ == "__main__":
    unittest.main()
