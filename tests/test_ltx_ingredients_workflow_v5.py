from collections import Counter
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "video_ltxv_ingredients_audio_2stage_v5.json",
        ["4922", 0],
        True,
    ),
    (
        "video_ltxv_ingredients_audio_2stage_gguf_v5.json",
        ["5307", 0],
        True,
    ),
    (
        "video_ltxv_ingredients_2stage_v5.json",
        ["4922", 0],
        False,
    ),
    (
        "video_ltxv_ingredients_2stage_gguf_v5.json",
        ["5307", 0],
        False,
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
        for name, expected_stage2_model, _ in CASES:
            with self.subTest(workflow=name):
                workflow = self._load(ROOT / "workflows" / name)
                self.assertEqual(
                    expected_stage2_model,
                    workflow["5203"]["inputs"]["model"],
                )
                self.assertEqual(
                    "0.909375, 0.725, 0.421875, 0.0",
                    workflow["5204"]["inputs"]["sigmas"],
                )

    def test_stage1_retains_ingredients_ic_lora(self):
        for name, _, _ in CASES:
            with self.subTest(workflow=name):
                workflow = self._load(ROOT / "workflows" / name)
                self.assertEqual(
                    ["5011", 0],
                    workflow["2483"]["inputs"]["model"],
                )
                self.assertEqual(
                    "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
                    workflow["5011"]["inputs"]["lora_name"],
                )

    def test_v5_preserves_required_semantic_anchors(self):
        for name, _, has_audio in CASES:
            with self.subTest(workflow=name):
                workflow = self._load(ROOT / "workflows" / name)
                required_anchors = REQUIRED_ANCHORS | (AUDIO_ANCHORS if has_audio else set())
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
