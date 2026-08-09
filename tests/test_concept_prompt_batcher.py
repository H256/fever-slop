import unittest
import json
from unittest.mock import MagicMock
from feverslop.prompting.concept_prompt_batcher import ConceptPromptBatcher


class ConceptPromptBatcherTests(unittest.TestCase):
    def test_reports_batch_progress(self):
        mock_llm = MagicMock()
        mock_llm.complete_prompt.side_effect = [
            json.dumps({"seg_1": "concept 1"}),
            "summary",
            json.dumps({"seg_2": "concept 2"}),
            "summary",
        ]
        progress = []
        batcher = ConceptPromptBatcher(
            llm=mock_llm,
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
        mock_llm = MagicMock()
        mock_llm.complete_prompt.return_value = json.dumps({"seg_1": "concept 1"})
        
        batcher = ConceptPromptBatcher(
            llm=mock_llm,
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
        self.assertTrue(mock_llm.complete_prompt.called)
        # Find the call in _generate_batch
        # It's called multiple times: 
        # 1. _generate_batch
        # 2. _summarize_progress
        
        found_batch_call = False
        found_summary_call = False
        
        for call in mock_llm.complete_prompt.call_args_list:
            if "CURRENT_BATCH_SEGMENTS" in call.kwargs.get("prompt", ""):
                found_batch_call = True
                self.assertEqual(call.kwargs.get("timeout"), 42.0)
            if "RECENT_CONCEPTS" in call.kwargs.get("prompt", ""):
                found_summary_call = True
                self.assertEqual(call.kwargs.get("timeout"), 42.0)
        
        self.assertTrue(found_batch_call, "_generate_batch was not called")
        self.assertTrue(found_summary_call, "_summarize_progress was not called")

    def test_passes_timeout_to_repair_call(self):
        mock_llm = MagicMock()
        # First call (generate batch) returns empty dict to trigger repair
        # Second call (repair) returns the missing key
        # Third call (summarize)
        mock_llm.complete_prompt.side_effect = [
            json.dumps({}), # batch result (missing seg_1)
            json.dumps({"seg_1": "repaired concept"}), # repair result
            "summary" # summarize progress result
        ]
        
        batcher = ConceptPromptBatcher(
            llm=mock_llm,
            batch_size=1,
            request_timeout_seconds=99.0
        )
        
        batcher.create_concept_prompts_batched(
            stage1_segments=[{"segment_id": "seg_1"}],
            story_idea="idea",
            global_context={},
        )
        
        found_repair_call = False
        for call in mock_llm.complete_prompt.call_args_list:
            if "MISSING_SEGMENTS" in call.kwargs.get("prompt", ""):
                found_repair_call = True
                self.assertEqual(call.kwargs.get("timeout"), 99.0)
                
        self.assertTrue(found_repair_call, "_repair_missing_or_extra_keys was not called")

if __name__ == "__main__":
    unittest.main()
