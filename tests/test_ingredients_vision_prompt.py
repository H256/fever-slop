from __future__ import annotations

import json
import unittest
from pathlib import Path

from feverslop.application.ingredients_vision_prompt import build_ingredients_vision_prompt
from feverslop.domain.vision_references import ReferenceImage


class FakeVisionLLM:
    def __init__(self, response: object = None, *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def complete_prompt_with_images(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[Path],
        timeout: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "prompt": prompt,
                "image_paths": image_paths,
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response)


class IngredientsVisionPromptTests(unittest.TestCase):
    def setUp(self):
        self.actor = Path("actor.png")
        self.location = Path("location.png")
        self.references = [
            ReferenceImage("mara", "actor", self.actor),
            ReferenceImage("archive", "location", self.location),
        ]
        self.reference_metadata = [
            {
                "id": "mara",
                "type": "actor",
                "t2i_description": "Mara has cropped black hair and a charcoal coat.",
                "manifest_description": "A guarded archivist with a silver ear cuff.",
            },
            {
                "id": "archive",
                "type": "location",
                "t2i_description": "A narrow stone archive lined with shelves.",
                "manifest_description": "Dusty subterranean records room.",
            },
        ]
        self.target_context = {
            "action": "Mara searches the shelves while the lights fail.",
            "camera": "Slow push from the doorway to a close medium shot.",
            "acting": "Controlled fear becomes resolve.",
            "dialogue": "I know you are here.",
            "continuity": "Her coat is wet from the previous shot.",
            "duration": "8 seconds",
        }
        self.fallback_reference = "fallback refs"
        self.fallback_invariants = "fallback shot invariants"

    @staticmethod
    def shot_invariants() -> str:
        return " ".join(f"invariant{i}" for i in range(80))

    def build(self, llm: FakeVisionLLM | None):
        return build_ingredients_vision_prompt(
            llm=llm,
            references=self.references,
            reference_metadata=self.reference_metadata,
            target_context=self.target_context,
            fallback_reference_description=self.fallback_reference,
            fallback_shot_invariants=self.fallback_invariants,
        )

    def test_builds_grounded_prompt_and_attaches_references_in_source_order(self):
        llm = FakeVisionLLM(
            {
                "references": [
                    {"id": "mara", "type": "actor", "description": "Cropped black hair, silver ear cuff, charcoal coat."},
                    {"id": "archive", "type": "location", "description": "Narrow stone archive with dusty shelves."},
                ],
                "shot_invariants": self.shot_invariants(),
            }
        )

        result = self.build(llm)

        call = llm.calls[0]
        self.assertEqual([self.actor, self.location], call["image_paths"])
        payload = json.loads(call["prompt"])
        self.assertEqual(self.reference_metadata, payload["references"])
        for key, value in self.target_context.items():
            self.assertEqual(value, payload["target_context"][key])
        system_prompt = str(call["system_prompt"])
        self.assertIn("60-160 words", system_prompt)
        self.assertIn("non-temporal", system_prompt)
        self.assertIn("Do not schedule", system_prompt)
        self.assertIn("singing", system_prompt)
        self.assertIn("lip-sync", system_prompt)
        self.assertIn("single continuous full-frame shot", system_prompt)
        self.assertIn("do not reproduce", system_prompt.lower())
        self.assertIn("framing", system_prompt.lower())
        self.assertIn("layout", system_prompt.lower())
        self.assertEqual(
            ["### Reference Sheet Description", "### Target Description"],
            [line for line in result.positive_prompt.splitlines() if line.startswith("###")],
        )
        for reference_id in ("mara", "archive"):
            self.assertIn(reference_id, result.reference_description)
        self.assertEqual(self.shot_invariants(), result.shot_invariants)

    def test_invalid_responses_use_deterministic_fallback_with_headings(self):
        invalid_responses = {
            "invalid json": "not JSON",
            "missing reference": {
                "references": [{"id": "mara", "type": "actor", "description": "Mara"}],
                "shot_invariants": self.shot_invariants(),
            },
            "wrong type": {
                "references": [
                    {"id": "mara", "type": "location", "description": "Mara"},
                    {"id": "archive", "type": "location", "description": "Archive"},
                ],
                "shot_invariants": self.shot_invariants(),
            },
            "empty target": {
                "references": [
                    {"id": "mara", "type": "actor", "description": "Mara"},
                    {"id": "archive", "type": "location", "description": "Archive"},
                ],
                "shot_invariants": "",
            },
            "short target": {
                "references": [
                    {"id": "mara", "type": "actor", "description": "Mara"},
                    {"id": "archive", "type": "location", "description": "Archive"},
                ],
                "shot_invariants": "too short",
            },
        }
        for label, response in invalid_responses.items():
            with self.subTest(label):
                result = self.build(FakeVisionLLM(response))
                self.assertEqual(self.fallback_reference, result.reference_description)
                self.assertEqual(self.fallback_invariants, result.shot_invariants)
                self.assertEqual("invalid response", result.fallback_reason)
                self.assertEqual(
                    "### Reference Sheet Description\nfallback refs\n\n### Target Description\nfallback shot invariants",
                    result.positive_prompt,
                )

    def test_model_exception_preserves_vision_unavailable_reason(self):
        for llm in (FakeVisionLLM(error=RuntimeError("offline")), None):
            with self.subTest(llm=llm):
                result = self.build(llm)
                self.assertEqual(self.fallback_reference, result.reference_description)
                self.assertEqual(self.fallback_invariants, result.shot_invariants)
                self.assertEqual("vision unavailable", result.fallback_reason)


if __name__ == "__main__":
    unittest.main()
