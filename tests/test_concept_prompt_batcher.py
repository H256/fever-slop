import unittest
import json
from feverslop.prompting.concept_prompt_batcher import ConceptPromptBatcher


class FakeConceptModules:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def concepts(self, payload, *, batch=False, silent_mode=False, timeout=None):
        self.calls.append(("concepts", payload, timeout))
        return next(self.responses)

    def repair_concepts(self, payload, *, timeout=None):
        self.calls.append(("repair_concepts", payload, timeout))
        return next(self.responses)

    def summary(self, payload, *, timeout=None):
        self.calls.append(("summary", payload, timeout))
        return next(self.responses)


class ConceptPromptBatcherTests(unittest.TestCase):
    def test_reports_batch_progress(self):
        modules = FakeConceptModules([
            json.dumps({"seg_1": "concept 1"}),
            "summary",
            json.dumps({"seg_2": "concept 2"}),
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            progress_callback=progress.append,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}, {"segment_id": "seg_2"}],
            story_idea="idea",
            global_context={},
        )

        self.assertEqual(
            [
                "Concept batch 1/2: generating scenes 1-1",
                "Concept batch 1/2: response received, validating keys",
                "Concept batch 1/2: complete (1 scenes total)",
                "Concept batch 2/2: generating scenes 2-2",
                "Concept batch 2/2: response received, validating keys",
                "Concept batch 2/2: complete (2 scenes total)",
            ],
            progress,
        )

    def test_passes_timeout_to_llm(self):
        modules = FakeConceptModules([json.dumps({"seg_1": "concept 1"}), "summary"])

        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            request_timeout_seconds=42.0
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}],
            story_idea="idea",
            global_context={"silent_mode": False},
            notes="notes"
        )

        # Check _generate_batch call
        found_batch_call = False
        found_summary_call = False

        for name, payload, timeout in modules.calls:
            if name == "concepts" and "CURRENT_BATCH_SEGMENTS" in payload:
                found_batch_call = True
                self.assertEqual(timeout, 42.0)
            if name == "summary" and "RECENT_CONCEPTS" in payload:
                found_summary_call = True
                self.assertEqual(timeout, 42.0)

        self.assertTrue(found_batch_call, "_generate_batch was not called")
        self.assertTrue(found_summary_call, "_summarize_progress was not called")

    def test_passes_timeout_to_repair_call(self):
        modules = FakeConceptModules([
            json.dumps({}),  # batch result (missing seg_1)
            json.dumps({"seg_1": "repaired concept"}),  # repair result
            "summary",  # summarize progress result
        ])

        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=1,
            request_timeout_seconds=99.0
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}],
            story_idea="idea",
            global_context={},
        )

        found_repair_call = False
        for name, payload, timeout in modules.calls:
            if name == "repair_concepts" and "MISSING_SEGMENTS" in payload:
                found_repair_call = True
                self.assertEqual(timeout, 99.0)

        self.assertTrue(found_repair_call, "_repair_missing_or_extra_keys was not called")

    def test_reports_ids_of_missing_scene_keys_before_repair(self):
        modules = FakeConceptModules([
            json.dumps({"seg_1": "concept 1"}),
            json.dumps({"seg_2": "repaired concept 2", "seg_3": "repaired concept 3"}),
            "summary",
        ])
        progress = []
        batcher = ConceptPromptBatcher(
            llm=object(),
            prompt_modules=modules,
            batch_size=3,
            progress_callback=progress.append,
        )

        batcher.create_concept_prompts_batched(
            stage1_segments=[
                {"segment_id": "seg_1"},
                {"segment_id": "seg_2"},
                {"segment_id": "seg_3"},
            ],
            story_idea="idea",
            global_context={},
        )

        self.assertIn(
            "Concept batch: repairing 2 missing or invalid scene keys: seg_2, seg_3",
            progress,
        )

    def test_repairs_concept_when_selected_actor_is_not_named(self):
        modules = FakeConceptModules([
            json.dumps({
                "seg_1": {
                    "concept": "The singer performs on stage.",
                    "references": {"actor_ids": ["singer", "bass"], "location_id": "stage"},
                }
            }),
            json.dumps({
                "seg_1": {
                    "concept": "The singer and Bass Player perform together on stage; the Bass Player holds the bass.",
                    "references": {"actor_ids": ["singer", "bass"], "location_id": "stage"},
                }
            }),
            "summary",
        ])
        batcher = ConceptPromptBatcher(llm=object(), prompt_modules=modules, batch_size=1)

        result = batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1", "type": "vocals"}],
            story_idea="idea",
            global_context={
                "actors": [
                    {"id": "singer", "name": "Goth Singer"},
                    {"id": "bass", "name": "Bass Player"},
                ]
            },
        )

        repair_calls = [call for call in modules.calls if call[0] == "repair_concepts"]
        self.assertEqual(1, len(repair_calls))
        self.assertEqual("seg_1", repair_calls[0][1]["INVALID_SEGMENTS"][0]["segment_id"])
        self.assertIn("bass player", repair_calls[0][1]["INVALID_SEGMENTS"][0]["reason"])
        self.assertIn("Bass Player", result["seg_1"]["concept"])

if __name__ == "__main__":
    unittest.main()
