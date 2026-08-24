import json
import tempfile
import unittest
from pathlib import Path

from feverslop.utils.render_plan_selection import load_render_plan_subset, parse_scene_list


class RenderPlanSelectionTests(unittest.TestCase):
    def test_parse_scene_list_supports_numbers_ranges_and_deduplicates(self):
        self.assertEqual({1, 2, 3, 5}, parse_scene_list("1,2-3,3,5"))

    def test_parse_scene_list_empty_value_means_all_scenes(self):
        self.assertIsNone(parse_scene_list(""))
        self.assertIsNone(parse_scene_list(None))

    def test_parse_scene_list_rejects_reversed_ranges(self):
        with self.assertRaisesRegex(ValueError, "reversed"):
            parse_scene_list("5-2")

    def test_load_render_plan_subset_filters_and_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "render_plan.json"
            plan_path.write_text(json.dumps([{"scene": 1}, {"scene": 2}, {"scene": 3}]), encoding="utf-8")

            result = load_render_plan_subset(plan_path, {2, 3}, limit=1)

        self.assertEqual([{"scene": 2}], result)


if __name__ == "__main__":
    unittest.main()
