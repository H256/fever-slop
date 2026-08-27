import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.prompt_contract_validation import validate_h3_prompt_shape, validate_prompt_contract


class PromptContractValidationTests(unittest.TestCase):
    def setUp(self):
        self.facts = LockedSceneFacts.create(
            scene_id="scene-01",
            facts=[{"category": "wardrobe", "key": "hero", "value": "silver coat", "source_id": "cast:hero"}],
        )
        self.shots = [
            CreativeShotPayload(shot_id="shot-0001", visible_action="waits", performance="quiet"),
            CreativeShotPayload(shot_id="shot-0002", visible_action="turns", performance="alert"),
        ]
        self.windows = {"shot-0001": (0, 2), "shot-0002": (2, 4)}
        self.prompt = (
            "Scene: scene-01\n"
            "Locked facts:\n"
            "- wardrobe/hero: silver coat\n"
            "[Shot 1 | 00:00.000-00:02.000]\nAction: waits\nPerformance: quiet\n"
            "References: <Picture 1>\n"
            "[Shot 2 | 00:02.000-00:04.000]\nAction: turns\nPerformance: alert\n"
            "References: <Picture 2>"
        )

    def test_accepts_complete_structured_prompt(self):
        issues = validate_prompt_contract(
            self.prompt,
            facts=self.facts,
            shots=self.shots,
            shot_windows=self.windows,
            references={"shot-0001": ["<Picture 1>"], "shot-0002": ["<Picture 2>"]},
            prepared_reference_labels={"<Picture 1>", "<Picture 2>"},
            duration_seconds=4,
        )
        self.assertEqual([], issues)

    def test_returns_stable_source_addressable_issues_without_fact_values(self):
        issues = validate_prompt_contract(
            self.prompt.replace("silver coat", "missing"),
            facts=self.facts,
            shots=self.shots,
            shot_windows={"shot-0001": (0, 3), "shot-0002": (2, 5)},
            references={"shot-0001": ["<Picture 9>"], "shot-0002": ["<Picture 2>"]},
            prepared_reference_labels={"<Picture 1>", "<Picture 2>"},
            duration_seconds=4,
        )

        self.assertEqual(
            ["fact.missing", "reference.unknown", "timing.overlap", "timing.duration_exceeded"],
            [issue.code for issue in issues],
        )
        self.assertEqual("facts[0].value", issues[0].path)
        self.assertEqual("cast:hero", issues[0].source_id)
        self.assertNotIn("silver coat", issues[0].message)

    def test_h3_shape_validator_rejects_internal_compiler_format(self):
        issues = validate_h3_prompt_shape("FULL REFERENCE PROMPT\nScene: scene-01", mode="r2v")
        self.assertEqual(["h3.sections.missing"], [issue.code for issue in issues])

    def test_h3_shape_validator_accepts_compiled_base_format(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] action\n\n"
            "overall_soundscape: N/A\n\n"
            "non_diegetic_music: N/A"
        )
        self.assertEqual([], validate_h3_prompt_shape(prompt, mode="t2v"))


if __name__ == "__main__":
    unittest.main()
