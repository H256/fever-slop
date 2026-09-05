import unittest

from feverslop.adapters.workflow_patcher import (
    WorkflowPatcher,
    _CLIP_STRENGTH_INPUTS,
    _MODEL_STRENGTH_INPUTS,
)

TITLE = "#LORA_1"


def _lora_workflow(inputs: dict) -> dict:
    return {"1": {"class_type": "LoraLoaderModelOnly", "inputs": dict(inputs), "_meta": {"title": TITLE}}}


class TestSetFirstMatchingInput(unittest.TestCase):
    def test_set_first_matching_input_returns_first_existing_candidate(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "sentinel_lora",
            "strength_model": "sentinel_model",
            "model_strength": "sentinel_model_strength",
            "strength": "sentinel_strength",
            "strength_clip": "sentinel_clip",
            "clip_strength": "sentinel_clip_strength",
        }))

        matched = patcher._set_first_matching_input(TITLE, _MODEL_STRENGTH_INPUTS, 0.75)

        self.assertEqual("strength_model", matched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual(0.75, inputs["strength_model"])
        self.assertEqual("sentinel_model_strength", inputs["model_strength"])
        self.assertEqual("sentinel_strength", inputs["strength"])
        self.assertEqual("sentinel_clip", inputs["strength_clip"])
        self.assertEqual("sentinel_clip_strength", inputs["clip_strength"])

    def test_set_first_matching_input_falls_back_to_later_candidate(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "sentinel_lora",
            "model_strength": "sentinel_model_strength",
        }))

        matched = patcher._set_first_matching_input(TITLE, _MODEL_STRENGTH_INPUTS, 0.5)

        self.assertEqual("model_strength", matched)
        self.assertEqual(0.5, patcher.get()["1"]["inputs"]["model_strength"])

    def test_set_first_matching_input_returns_none_when_no_candidate_exists(self):
        patcher = WorkflowPatcher(_lora_workflow({"lora_name": "sentinel_lora"}))

        matched = patcher._set_first_matching_input(TITLE, _CLIP_STRENGTH_INPUTS, 0.5)

        self.assertIsNone(matched)
        self.assertEqual({"lora_name": "sentinel_lora"}, patcher.get()["1"]["inputs"])


class TestPatchLoraByTitle(unittest.TestCase):
    def test_patch_lora_by_title_prefers_first_model_and_clip_candidates(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "strength_model": 1.0,
            "model_strength": 1.0,
            "strength": 1.0,
            "strength_clip": 1.0,
            "clip_strength": 1.0,
        }))

        patched = patcher.patch_lora_by_title(TITLE, "new_lora", 0.8, 0.6)

        self.assertEqual(["lora_name", "strength_model", "strength_clip"], patched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual("new_lora", inputs["lora_name"])
        self.assertEqual(0.8, inputs["strength_model"])
        self.assertEqual(0.6, inputs["strength_clip"])
        self.assertEqual(1.0, inputs["model_strength"])
        self.assertEqual(1.0, inputs["strength"])
        self.assertEqual(1.0, inputs["clip_strength"])

    def test_patch_lora_by_title_falls_back_to_later_candidates(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "model_strength": 1.0,
            "clip_strength": 1.0,
        }))

        patched = patcher.patch_lora_by_title(TITLE, "new_lora", 0.8, 0.6)

        self.assertEqual(["lora_name", "model_strength", "clip_strength"], patched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual("new_lora", inputs["lora_name"])
        self.assertEqual(0.8, inputs["model_strength"])
        self.assertEqual(0.6, inputs["clip_strength"])

    def test_patch_lora_by_title_without_model_strength_raises_exact_message(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "strength_clip": 1.0,
        }))

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_by_title(TITLE, "new_lora", 0.8, 0.6)

        self.assertEqual(
            "No known LoRA model strength input found on node '#LORA_1'. "
            "Tried: strength_model, model_strength, strength",
            ctx.exception.args[0],
        )
        self.assertEqual(1.0, patcher.get()["1"]["inputs"]["strength_clip"])

    def test_patch_lora_by_title_without_clip_strength_is_optional(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "strength_model": 1.0,
        }))

        patched = patcher.patch_lora_by_title(TITLE, "new_lora", 0.8, 0.6)

        self.assertEqual(["lora_name", "strength_model"], patched)
        self.assertEqual(0.8, patcher.get()["1"]["inputs"]["strength_model"])

    def test_patch_lora_by_title_missing_node_raises_title_error(self):
        patcher = WorkflowPatcher({})

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_by_title(TITLE, "new_lora", 0.8, 0.6)

        self.assertEqual("Node with _meta.title not found: #LORA_1", ctx.exception.args[0])


class TestPatchLoraFieldsByTitle(unittest.TestCase):
    def test_patch_lora_fields_by_title_patches_only_requested_fields(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "strength_model": 1.0,
            "strength_clip": 1.0,
        }))

        patched = patcher.patch_lora_fields_by_title(TITLE, strength_clip=0.7)

        self.assertEqual(["strength_clip"], patched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual(0.7, inputs["strength_clip"])
        self.assertEqual("old_lora", inputs["lora_name"])
        self.assertEqual(1.0, inputs["strength_model"])

    def test_patch_lora_fields_by_title_all_none_raises_no_fields_message(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "strength_model": 1.0,
            "strength_clip": 1.0,
        }))

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_fields_by_title(TITLE)

        self.assertEqual("No LoRA fields were patched on node '#LORA_1'", ctx.exception.args[0])

    def test_patch_lora_fields_by_title_unmatchable_model_strength_raises_exact_message(self):
        patcher = WorkflowPatcher(_lora_workflow({"lora_name": "old_lora"}))

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_fields_by_title(TITLE, strength_model=1.0)

        self.assertEqual(
            "No known LoRA model strength input found on node '#LORA_1'. "
            "Tried: strength_model, model_strength, strength",
            ctx.exception.args[0],
        )

    def test_patch_lora_fields_by_title_unmatchable_clip_is_optional_but_empty_patch_raises_combined(self):
        patcher = WorkflowPatcher(_lora_workflow({"lora_name": "old_lora"}))

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_fields_by_title(TITLE, strength_clip=0.5)

        self.assertEqual("No LoRA fields were patched on node '#LORA_1'", ctx.exception.args[0])

    def test_patch_lora_fields_by_title_combined_order(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "lora_name": "old_lora",
            "model_strength": 1.0,
            "clip_strength": 1.0,
        }))

        patched = patcher.patch_lora_fields_by_title(TITLE, lora_name="new_lora", strength_model=0.8, strength_clip=0.6)

        self.assertEqual(["lora_name", "model_strength", "clip_strength"], patched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual("new_lora", inputs["lora_name"])
        self.assertEqual(0.8, inputs["model_strength"])
        self.assertEqual(0.6, inputs["clip_strength"])


class TestPatchLoraStrengthsByTitle(unittest.TestCase):
    def test_patch_lora_strengths_by_title_patches_first_match_per_group(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "strength_model": 1.0,
            "model_strength": 1.0,
            "strength": 1.0,
            "strength_clip": 1.0,
            "clip_strength": 1.0,
        }))

        patched = patcher.patch_lora_strengths_by_title(TITLE, 0.8, 0.6)

        self.assertEqual(["strength_model", "strength_clip"], patched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual(0.8, inputs["strength_model"])
        self.assertEqual(0.6, inputs["strength_clip"])
        self.assertEqual(1.0, inputs["model_strength"])
        self.assertEqual(1.0, inputs["strength"])
        self.assertEqual(1.0, inputs["clip_strength"])

    def test_patch_lora_strengths_by_title_only_model_present(self):
        patcher = WorkflowPatcher(_lora_workflow({"model_strength": 1.0}))

        patched = patcher.patch_lora_strengths_by_title(TITLE, 0.8, 0.6)

        self.assertEqual(["model_strength"], patched)
        self.assertEqual(0.8, patcher.get()["1"]["inputs"]["model_strength"])

    def test_patch_lora_strengths_by_title_only_clip_present(self):
        patcher = WorkflowPatcher(_lora_workflow({"clip_strength": 1.0}))

        patched = patcher.patch_lora_strengths_by_title(TITLE, 0.8, 0.6)

        self.assertEqual(["clip_strength"], patched)
        self.assertEqual(0.6, patcher.get()["1"]["inputs"]["clip_strength"])

    def test_patch_lora_strengths_by_title_no_match_raises_combined_message(self):
        patcher = WorkflowPatcher(_lora_workflow({"lora_name": "old_lora"}))

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_strengths_by_title(TITLE, 0.8, 0.6)

        self.assertEqual(
            "No known LoRA strength input found on node '#LORA_1'. "
            "Tried model and clip strength inputs.",
            ctx.exception.args[0],
        )

    def test_patch_lora_strengths_by_title_missing_node_raises_combined_message(self):
        patcher = WorkflowPatcher({})

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_strengths_by_title(TITLE, 0.8, 0.6)

        self.assertEqual(
            "No known LoRA strength input found on node '#LORA_1'. "
            "Tried model and clip strength inputs.",
            ctx.exception.args[0],
        )


class TestPatchLoraStrengthByTitle(unittest.TestCase):
    def test_patch_lora_strength_by_title_patches_all_existing_matches_in_tuple_order(self):
        patcher = WorkflowPatcher(_lora_workflow({
            "strength_model": 1.0,
            "strength": 1.0,
            "model_strength": 1.0,
            "clip_strength": 1.0,
        }))

        patched = patcher.patch_lora_strength_by_title(TITLE, 0.42)

        self.assertEqual(["strength_model", "strength", "model_strength", "clip_strength"], patched)
        inputs = patcher.get()["1"]["inputs"]
        self.assertEqual(0.42, inputs["strength_model"])
        self.assertEqual(0.42, inputs["strength"])
        self.assertEqual(0.42, inputs["model_strength"])
        self.assertEqual(0.42, inputs["clip_strength"])

    def test_patch_lora_strength_by_title_no_match_raises_with_joined_names(self):
        patcher = WorkflowPatcher(_lora_workflow({"lora_name": "old_lora"}))

        with self.assertRaises(KeyError) as ctx:
            patcher.patch_lora_strength_by_title(TITLE, 0.42)

        self.assertEqual(
            "No known LoRA strength input found on node '#LORA_1'. "
            "Tried: strength_model, strength_clip, strength, model_strength, clip_strength",
            ctx.exception.args[0],
        )

    def test_patch_lora_strength_by_title_respects_custom_input_names(self):
        patcher = WorkflowPatcher(_lora_workflow({"s1": 1.0}))

        patched = patcher.patch_lora_strength_by_title(TITLE, 0.42, input_names=("s1",))

        self.assertEqual(["s1"], patched)
        self.assertEqual(0.42, patcher.get()["1"]["inputs"]["s1"])


if __name__ == "__main__":
    unittest.main()
