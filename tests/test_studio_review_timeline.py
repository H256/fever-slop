import unittest


class ReviewTimelineStateTests(unittest.TestCase):
    def test_builds_ordered_items_and_finds_raw_and_final_clips(self):
        from feverslop.studio.desktop.review_timeline import ReviewTimelineState

        state = ReviewTimelineState.from_document([
            {"scene": 1, "duration_seconds": 2.0, "fps": 24},
            {"scene": 3, "duration_seconds": 3.0, "fps": 24},
        ])

        items = state.items([
            "output/raw/scene_0001_raw.mp4",
            "output/final/scene_0001.mp4",
            "output/raw/scene_0003_raw.mp4",
        ])

        self.assertEqual([(item["scene"], item["start"], item["duration"]) for item in items], [
            (1, 0.0, 2.0),
            (3, 2.0, 3.0),
        ])
        self.assertEqual(items[0]["status"], "final")
        self.assertEqual(items[1]["status"], "raw")

    def test_move_scene_supports_undo_and_redo(self):
        from feverslop.studio.desktop.review_timeline import ReviewTimelineState

        state = ReviewTimelineState.from_document({"shots": [{"scene": 1}, {"scene": 2}, {"scene": 3}]})

        self.assertTrue(state.move(2, 0))
        self.assertEqual([scene["scene"] for scene in state.scenes], [3, 1, 2])
        self.assertTrue(state.undo())
        self.assertEqual([scene["scene"] for scene in state.scenes], [1, 2, 3])
        self.assertTrue(state.redo())
        self.assertEqual([scene["scene"] for scene in state.scenes], [3, 1, 2])

    def test_trim_marks_scene_stale_and_preserves_wrapper_on_save(self):
        from feverslop.studio.desktop.review_timeline import ReviewTimelineState

        state = ReviewTimelineState.from_document({"scenes": [{"scene": 5, "fps": 24, "duration_seconds": 4.0}]})

        self.assertTrue(state.trim(5, 0.5, 3.5))

        scene = state.scenes[0]
        self.assertEqual(scene["edit"]["raw_in_frame"], 12)
        self.assertEqual(scene["edit"]["raw_out_frame"], 84)
        self.assertTrue(scene["edit"]["studio_stale"])
        self.assertEqual(state.document(), {"scenes": [scene]})

    def test_trim_rejects_empty_range(self):
        from feverslop.studio.desktop.review_timeline import ReviewTimelineState

        state = ReviewTimelineState.from_document([{"scene": 1, "fps": 24, "duration_seconds": 4.0}])

        self.assertFalse(state.trim(1, 2.0, 2.0))
        self.assertFalse(state.dirty)


if __name__ == "__main__":
    unittest.main()
