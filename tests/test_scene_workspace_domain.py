from __future__ import annotations

import unittest

from feverslop.domain.scene_workspace import SceneMedia, SceneWorkspace, SceneWorkspaceItem


class SceneWorkspaceDomainTests(unittest.TestCase):
    def test_workspace_preserves_scene_order_and_raw_payload(self):
        later_scene = {
            "scene": 3,
            "abs_start_seconds": 8.0,
            "abs_end_seconds": 12.0,
            "workflow_extension": {"model": "custom", "weights": [1, 2]},
        }
        workspace = SceneWorkspace.from_scenes(
            [later_scene, {"scene": 1, "abs_start_seconds": 0.0, "abs_end_seconds": 4.0}]
        )

        later_scene["workflow_extension"]["weights"].append(3)

        self.assertEqual((3, 1), tuple(item.scene_number for item in workspace.items))
        self.assertEqual(
            {"model": "custom", "weights": [1, 2]},
            workspace.items[0].raw_scene["workflow_extension"],
        )
        self.assertEqual([3, 1], [scene["scene"] for scene in workspace.to_scenes()])

    def test_internal_raw_payload_cannot_be_mutated(self):
        workspace = SceneWorkspace.from_scenes(
            [
                {
                    "scene": 1,
                    "workflow_extension": {"weights": [1, 2]},
                }
            ]
        )
        item = workspace.items[0]

        with self.assertRaises(TypeError):
            item._raw_scene["workflow_extension"] = {"weights": [3]}
        with self.assertRaises((AttributeError, TypeError)):
            item._raw_scene["workflow_extension"]["weights"].append(3)

        raw_scene = item.raw_scene
        raw_scene["workflow_extension"]["weights"].append(3)
        self.assertIsInstance(raw_scene, dict)
        self.assertIsInstance(raw_scene["workflow_extension"]["weights"], list)
        self.assertEqual(
            {"scene": 1, "workflow_extension": {"weights": [1, 2]}},
            item.to_scene(),
        )

    def test_raw_payload_round_trips_json_scalars(self):
        scene = {
            "scene": 1,
            "extension": {
                "text": "value",
                "integer": 4,
                "floating": 2.5,
                "enabled": True,
                "missing": None,
                "items": ["x", 1, 1.5, False, None],
            },
        }

        item = SceneWorkspace.from_scenes([scene]).items[0]

        self.assertEqual(scene, item.to_scene())

    def test_raw_payload_rejects_non_json_values(self):
        with self.assertRaisesRegex(TypeError, "non-JSON value: tuple"):
            SceneWorkspace.from_scenes([{"scene": 1, "extension": (1, 2)}])

    def test_raw_payload_rejects_non_string_keys(self):
        with self.assertRaisesRegex(TypeError, "JSON object keys must be strings"):
            SceneWorkspace.from_scenes([{"scene": 1, "extension": {2: "invalid"}}])

    def test_direct_item_construction_copies_reference_ids(self):
        reference_ids = ["mara", "archive"]

        item = SceneWorkspaceItem(scene_number=1, reference_ids=reference_ids)
        reference_ids.append("later")

        self.assertEqual(("mara", "archive"), item.reference_ids)

    def test_direct_workspace_construction_copies_items(self):
        items = [SceneWorkspaceItem(scene_number=1)]

        workspace = SceneWorkspace(items=items)
        items.append(SceneWorkspaceItem(scene_number=2))

        self.assertEqual((1,), tuple(item.scene_number for item in workspace.items))

    def test_reference_ids_support_standard_scene_references(self):
        workspace = SceneWorkspace.from_scenes(
            [
                {
                    "scene": 1,
                    "references": {
                        "actor_ids": ["mara", "ivo"],
                        "location_id": "archive",
                        "metadata": {"confidence": 0.8},
                    },
                }
            ]
        )

        self.assertEqual(("mara", "ivo", "archive"), workspace.items[0].reference_ids)

    def test_reference_ids_support_movie_schema_without_stringifying_metadata(self):
        workspace = SceneWorkspace.from_scenes(
            [
                {
                    "scene": 1,
                    "reference_ids": {
                        "actors": ["mara"],
                        "location": "archive",
                        "metadata": {"confidence": 0.8},
                    },
                }
            ]
        )

        self.assertEqual(("mara", "archive"), workspace.items[0].reference_ids)

    def test_missing_optional_fields_use_empty_display_values(self):
        workspace = SceneWorkspace.from_scenes([{"scene": 7}])

        item = workspace.items[0]

        self.assertEqual(0.0, item.start_seconds)
        self.assertEqual(0.0, item.end_seconds)
        self.assertEqual("", item.performance_state)
        self.assertEqual("", item.shot_description)
        self.assertEqual("", item.image_prompt)
        self.assertEqual("", item.video_prompt)
        self.assertEqual((), item.reference_ids)
        self.assertEqual(SceneMedia(), item.media)

    def test_status_is_derived_only_from_supplied_media_facts(self):
        scenes = [{"scene": number} for number in range(1, 5)]
        workspace = SceneWorkspace.from_scenes(
            scenes,
            media_by_scene={
                2: SceneMedia(workflow_path="scenes/0002/workflow.json"),
                3: SceneMedia(video_path="output/final/scene_0003.mp4"),
                4: SceneMedia(
                    workflow_path="scenes/0004/workflow.json",
                    video_path="output/final/scene_0004.mp4",
                    failure_message="ComfyUI execution failed",
                ),
            },
        )

        self.assertEqual(
            ("missing", "planned", "rendered", "failed"),
            tuple(item.status for item in workspace.items),
        )

    def test_workspace_rejects_duplicate_scene_numbers(self):
        with self.assertRaisesRegex(ValueError, "Duplicate scene number: 2"):
            SceneWorkspace.from_scenes(
                [
                    {"scene": 2, "start": 0.0, "end": 4.0},
                    {"scene": 2, "start": 4.0, "end": 8.0},
                ]
            )


if __name__ == "__main__":
    unittest.main()
