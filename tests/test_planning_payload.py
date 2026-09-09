import unittest

from feverslop.prompting.planning_payload import compact_creative_context


class PlanningPayloadTests(unittest.TestCase):
    def test_creative_context_drops_global_artifact_payloads(self):
        result = compact_creative_context({
            "story_idea": "A gothic performance.",
            "style": "Dark and cinematic.",
            "actors": [{"id": "lead", "description": "x" * 10000}],
            "structured_locations": [{"id": "ruins", "description": "y" * 10000}],
            "global_asset_snapshots": [{"path": "huge.json", "payload": "z" * 10000}],
            "props": [{"id": "tank"}],
            "prompt_guidance": {"max_words": 500},
        })

        self.assertEqual({"story_idea", "style", "prompt_guidance"}, set(result))
        self.assertNotIn("global_asset_snapshots", result)
        self.assertNotIn("actors", result)
