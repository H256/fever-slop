import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.deterministic_h3_compiler import DeterministicH3Compiler


class DeterministicH3CompilerTests(unittest.TestCase):
    def setUp(self):
        self.facts = LockedSceneFacts.create(
            scene_id="scene-01",
            facts=[
                {"category": "wardrobe", "key": "hero", "value": "silver cloak", "source_id": "cast:hero"},
                {"category": "location", "key": "primary", "value": "ruined gate", "source_id": "location:gate"},
            ],
        )
        self.shots = [
            CreativeShotPayload(
                shot_id="shot-02", visible_action="The lantern rises.", performance="defiant", camera_behavior="slow push",
            ),
            CreativeShotPayload(
                shot_id="shot-01", visible_action="The singer waits.", performance="restrained grief", transition_intent="hold for continuation",
            ),
        ]

    def test_compiles_stable_base_and_reference_sections(self):
        compiler = DeterministicH3Compiler()
        base = compiler.compile(
            mode="base", facts=self.facts, shots=self.shots,
            shot_windows={"shot-01": (0.0, 4.5), "shot-02": (4.5, 9.0)},
            references={"shot-01": ["<Picture 1>"], "shot-02": ["<Picture 1>", "<Audio 1>"]},
        )
        reference = compiler.compile(
            mode="reference", facts=self.facts, shots=list(reversed(self.shots)),
            shot_windows={"shot-02": (4.5, 9.0), "shot-01": (0.0, 4.5)},
            references={"shot-02": ["<Audio 1>", "<Picture 1>"], "shot-01": ["<Picture 1>"]},
        )

        self.assertIn("BASE PROMPT", base)
        self.assertIn("FULL REFERENCE PROMPT", reference)
        self.assertIn("[Shot 1 | 00:00.000-00:04.500]", base)
        self.assertIn("<Picture 1>", reference)
        self.assertEqual(base, compiler.compile(
            mode="base", facts=self.facts, shots=list(reversed(self.shots)),
            shot_windows={"shot-02": (4.5, 9.0), "shot-01": (0.0, 4.5)},
            references={"shot-02": ["<Audio 1>", "<Picture 1>"], "shot-01": ["<Picture 1>"]},
        ))

    def test_enforces_word_budget(self):
        with self.assertRaisesRegex(ValueError, "word budget"):
            DeterministicH3Compiler(max_words=3).compile(
                mode="base", facts=self.facts, shots=self.shots,
                shot_windows={"shot-01": (0.0, 4.5), "shot-02": (4.5, 9.0)},
            )


if __name__ == "__main__":
    unittest.main()
