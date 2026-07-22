from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkflowCase:
    name: str
    stage2_model_id: str
    stage2_class_type: str
    stage2_model_input: str
    stage2_model_name: str
    stage1_source_id: str
    has_audio: bool


CASES = (
    WorkflowCase(
        name="video_ltxv_ingredients_audio_2stage_v5.json",
        stage2_model_id="4922",
        stage2_class_type="LoraLoaderModelOnly",
        stage2_model_input="lora_name",
        stage2_model_name="ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        stage1_source_id="4922",
        has_audio=True,
    ),
    WorkflowCase(
        name="video_ltxv_ingredients_audio_2stage_gguf_v5.json",
        stage2_model_id="5307",
        stage2_class_type="UnetLoaderGGUF",
        stage2_model_input="unet_name",
        stage2_model_name="LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
        stage1_source_id="5307",
        has_audio=True,
    ),
    WorkflowCase(
        name="video_ltxv_ingredients_2stage_v5.json",
        stage2_model_id="4922",
        stage2_class_type="LoraLoaderModelOnly",
        stage2_model_input="lora_name",
        stage2_model_name="ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        stage1_source_id="4922",
        has_audio=False,
    ),
    WorkflowCase(
        name="video_ltxv_ingredients_2stage_gguf_v5.json",
        stage2_model_id="5307",
        stage2_class_type="UnetLoaderGGUF",
        stage2_model_input="unet_name",
        stage2_model_name="LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
        stage1_source_id="5306",
        has_audio=False,
    ),
)
V4_NAMES = (
    "video_ltxv_ingredients_audio_2stage_v4.json",
    "video_ltxv_ingredients_audio_2stage_gguf_v4.json",
    "video_ltxv_ingredients_2stage_v4.json",
    "video_ltxv_ingredients_2stage_gguf_v4.json",
)
REQUIRED_ANCHORS = {
    "#INGREDIENTS",
    "#PROMPT_RELAY",
    "#SEED",
    "#WIDTH",
    "#HEIGHT",
    "#FRAMES",
    "#FRAMERATE",
    "#SAVE_VIDEO",
}
AUDIO_ANCHORS = {"#LOAD_AUDIO", "#TRIM_AUDIO"}


class IngredientsWorkflowV5Tests(unittest.TestCase):
    def test_stage2_bypasses_ingredients_ic_lora(self):
        for case in CASES:
            with self.subTest(workflow=case.name):
                workflow = self._load(ROOT / "workflows" / case.name)
                self.assertNotEqual("5011", case.stage2_model_id)
                self.assertIn(case.stage2_model_id, workflow)
                self.assertEqual(
                    [case.stage2_model_id, 0],
                    workflow["5203"]["inputs"]["model"],
                )
                stage2_model = workflow[case.stage2_model_id]
                self.assertEqual(case.stage2_class_type, stage2_model["class_type"])
                self.assertEqual(
                    case.stage2_model_name,
                    stage2_model["inputs"][case.stage2_model_input],
                )
                self.assertEqual(
                    "0.909375, 0.725, 0.421875, 0.0",
                    workflow["5204"]["inputs"]["sigmas"],
                )

    def test_stage1_retains_ingredients_ic_lora(self):
        for case in CASES:
            with self.subTest(workflow=case.name):
                workflow = self._load(ROOT / "workflows" / case.name)
                self.assertEqual(
                    ["5011", 0],
                    workflow["2483"]["inputs"]["model"],
                )
                self.assertEqual(
                    [case.stage1_source_id, 0],
                    workflow["5011"]["inputs"]["model"],
                )
                self.assertEqual(
                    "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
                    workflow["5011"]["inputs"]["lora_name"],
                )
                if case.stage2_class_type == "UnetLoaderGGUF":
                    stage1_source = workflow[case.stage1_source_id]
                    self.assertEqual("UnetLoaderGGUF", stage1_source["class_type"])
                    self.assertEqual(
                        case.stage2_model_name,
                        stage1_source["inputs"]["unet_name"],
                    )
                if case.stage1_source_id == "5306":
                    self.assertNotEqual(
                        workflow["5011"]["inputs"]["model"],
                        workflow["5203"]["inputs"]["model"],
                    )

    def test_v5_preserves_required_semantic_anchors(self):
        for case in CASES:
            with self.subTest(workflow=case.name):
                workflow = self._load(ROOT / "workflows" / case.name)
                required_anchors = REQUIRED_ANCHORS | (
                    AUDIO_ANCHORS if case.has_audio else set()
                )
                titles = Counter(
                    node.get("_meta", {}).get("title")
                    for node in workflow.values()
                    if str(node.get("_meta", {}).get("title", "")).startswith("#")
                )
                self.assertTrue(required_anchors.issubset(titles))
                self.assertTrue(all(titles[title] == 1 for title in required_anchors))

    def test_v4_workflows_are_archived_without_root_aliases(self):
        for name in V4_NAMES:
            with self.subTest(workflow=name):
                self.assertFalse((ROOT / "workflows" / name).exists())
                archived = ROOT / "workflows" / "old" / name
                self.assertTrue(archived.is_file())
                workflow = self._load(archived)
                self.assertEqual(["5011", 0], workflow["5203"]["inputs"]["model"])

    def _load(self, path: Path) -> dict:
        self.assertTrue(path.is_file(), f"Missing workflow: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
