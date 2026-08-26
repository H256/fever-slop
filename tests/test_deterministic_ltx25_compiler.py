import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.deterministic_ltx25_compiler import DeterministicLTX25Compiler
from feverslop.prompting.dspy_h3_models import CreativeShotPayload


class DeterministicLTX25CompilerTests(unittest.TestCase):
    def test_compilation_is_stable_and_mode_aware(self):
        facts = LockedSceneFacts.create(scene_id="s1", facts=[])
        shot = CreativeShotPayload(
            shot_id="b", visible_action="walks", performance="calm",
            camera_behavior="tracking", environmental_motion="rain", transition_intent="match",
        )
        compiler = DeterministicLTX25Compiler()
        first = compiler.compile(facts=facts, shots=[shot], shot_windows={"b": (1, 3)}, mode="r2v")
        second = compiler.compile(facts=facts, shots=[shot], shot_windows={"b": (1, 3)}, mode="r2v")
        self.assertEqual(first, second)
        self.assertIn("LTX 2.5 R2V PROMPT", first)


if __name__ == "__main__":
    unittest.main()
