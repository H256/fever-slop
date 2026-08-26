import unittest

from feverslop.prompting.dspy_h3_models import CreativeShotPayload


class CreativeShotPayloadTests(unittest.TestCase):
    def test_roundtrips_bounded_creative_fields_in_shot_order(self):
        payload = CreativeShotPayload(
            shot_id="shot-0002",
            visible_action="The singer raises the lantern toward the ruined gate.",
            performance="restrained grief building into defiance",
            camera_behavior="slow lateral dolly from medium shot to close-up",
            environmental_motion="fog streams through the gate and embers drift upward",
            transition_intent="end with the lantern held in the same position for continuation",
        )

        restored = CreativeShotPayload.model_validate(payload.model_dump())

        self.assertEqual(payload, restored)
        self.assertEqual("shot-0002", payload.shot_id)

    def test_rejects_backend_facts_reference_syntax_and_free_prompt(self):
        invalid = (
            {"shot_id": "shot-1", "visible_action": "<Picture 1>", "performance": "quiet"},
            {"shot_id": "shot-1", "visible_action": "at 00:01.200 she turns", "performance": "quiet"},
            {"shot_id": "shot-1", "visible_action": "turns", "performance": "quiet", "prompt": "arbitrary prose"},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CreativeShotPayload.model_validate(values)

    def test_optional_fields_are_omitted_from_renderable_payload(self):
        payload = CreativeShotPayload(
            shot_id="shot-1",
            visible_action="The gate opens.",
            performance="watchful",
        )

        self.assertNotIn("camera_behavior", payload.model_dump(exclude_none=True))
        self.assertNotIn("transition_intent", payload.model_dump(exclude_none=True))


if __name__ == "__main__":
    unittest.main()
