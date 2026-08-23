import unittest

from feverslop.errors import FeverSlopValidationError
from feverslop.prompting.minimax_h3_prompt_style import (
    MAX_R2V_AUDIO_REFS,
    MAX_R2V_IMAGE_REFS,
    MAX_R2V_VIDEO_REFS,
    MAX_T2V_FRAME_REFS,
    _build_r2v_prompt,
    _build_t2v_prompt,
    _collect_r2v_image_refs,
    _collect_t2v_frame_refs,
    build_r2v_prompt,
    build_t2v_prompt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R2V = [
    "MAX_R2V_IMAGE_REFS is 9",
    "MAX_R2V_VIDEO_REFS is 3",
    "MAX_R2V_AUDIO_REFS is 3",
]

T2V = ["MAX_T2V_FRAME_REFS is 2"]


class ConstantsTests(unittest.TestCase):
    def test_r2v_image_refs_limit(self):
        self.assertEqual(MAX_R2V_IMAGE_REFS, 9)

    def test_r2v_video_refs_limit(self):
        self.assertEqual(MAX_R2V_VIDEO_REFS, 3)

    def test_r2v_audio_refs_limit(self):
        self.assertEqual(MAX_R2V_AUDIO_REFS, 3)

    def test_t2v_frame_refs_limit(self):
        self.assertEqual(MAX_T2V_FRAME_REFS, 2)


# ---------------------------------------------------------------------------
# _collect_r2v_image_refs
# ---------------------------------------------------------------------------

class CollectR2VImageRefsTests(unittest.TestCase):
    def test_actors_before_location_before_style(self):
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png", "a2.png"],
                "location_sheet_path": "loc.png",
                "style_reference_paths": [{"name": "Moody", "path": "s1.png"}],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        labels = [lbl for lbl, _ in refs]
        self.assertEqual(["Actor 1", "Actor 2", "Location", "Moody"], labels)

    def test_location_missing_when_empty(self):
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png"],
                "location_sheet_path": "",
                "style_reference_paths": [],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0][0], "Actor 1")

    def test_location_missing_when_absent(self):
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png"],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0][0], "Actor 1")

    def test_style_refs_last(self):
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png"],
                "style_reference_paths": [
                    {"name": "S1", "path": "s1.png"},
                    {"name": "S2", "path": "s2.png"},
                ],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        labels = [lbl for lbl, _ in refs]
        self.assertEqual(["Actor 1", "S1", "S2"], labels)

    def test_clamped_to_nine(self):
        paths = [f"img{i}.png" for i in range(12)]
        scene = {"references": {"actor_sheet_paths": paths}}
        refs = _collect_r2v_image_refs(scene)
        self.assertEqual(len(refs), MAX_R2V_IMAGE_REFS)

    def test_actor_names_from_descriptions(self):
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png", "a2.png"],
                "actor_reference_descriptions": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                ],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        labels = [lbl for lbl, _ in refs]
        self.assertEqual(["Alice", "Bob"], labels)

    def test_fallback_positional_labels(self):
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png", "a2.png", "a3.png"],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        labels = [lbl for lbl, _ in refs]
        self.assertEqual(["Actor 1", "Actor 2", "Actor 3"], labels)

    def test_empty_references_returns_empty(self):
        scene = {"references": {}}
        self.assertEqual(_collect_r2v_image_refs(scene), [])

    def test_no_references_key_returns_empty(self):
        scene = {}
        self.assertEqual(_collect_r2v_image_refs(scene), [])

    def test_location_name_from_description(self):
        scene = {
            "references": {
                "location_sheet_path": "loc.png",
                "location_reference_description": {"name": "Castle"},
            },
        }
        refs = _collect_r2v_image_refs(scene)
        self.assertEqual(refs[0][0], "Castle")

    def test_style_ref_path_as_string(self):
        scene = {
            "references": {
                "style_reference_paths": ["plain.png"],
            },
        }
        refs = _collect_r2v_image_refs(scene)
        self.assertEqual(refs[0], ("Style ref", "plain.png"))


# ---------------------------------------------------------------------------
# _collect_t2v_frame_refs
# ---------------------------------------------------------------------------

class CollectT2VFrameRefsTests(unittest.TestCase):
    def test_both_frames_present(self):
        scene = {
            "keyframes": {
                "startframe_path": "start.png",
                "lastframe_path": "end.png",
            },
        }
        refs = _collect_t2v_frame_refs(scene)
        self.assertEqual(
            refs,
            [("first_frame", "start.png"), ("last_frame", "end.png")],
        )

    def test_start_frame_only(self):
        scene = {
            "keyframes": {
                "startframe_path": "start.png",
            },
        }
        refs = _collect_t2v_frame_refs(scene)
        self.assertEqual(refs, [("first_frame", "start.png")])

    def test_missing_startframe_returns_empty(self):
        scene = {"keyframes": {"lastframe_path": "end.png"}}
        self.assertEqual(_collect_t2v_frame_refs(scene), [])

    def test_no_keyframes_returns_empty(self):
        self.assertEqual(_collect_t2v_frame_refs({}), [])

    def test_empty_keyframes_returns_empty(self):
        scene = {"keyframes": {}}
        self.assertEqual(_collect_t2v_frame_refs(scene), [])


# ---------------------------------------------------------------------------
# _build_r2v_prompt
# ---------------------------------------------------------------------------

class BuildR2VPromptTests(unittest.TestCase):
    def test_two_actors_produces_correct_tags(self):
        scene = {
            "description": "a cinematic shot",
            "references": {
                "actor_sheet_paths": ["a1.png", "a2.png"],
            },
        }
        prompt = _build_r2v_prompt(scene)
        self.assertIn("<Picture 1> Actor 1 ", prompt)
        self.assertIn("<Picture 2> Actor 2 ", prompt)
        self.assertTrue(prompt.endswith("a cinematic shot"))

    def test_full_scene_with_location_and_style(self):
        scene = {
            "description": "final description",
            "references": {
                "actor_sheet_paths": ["a1.png"],
                "location_sheet_path": "loc.png",
                "style_reference_paths": [{"name": "Noir", "path": "s1.png"}],
            },
        }
        prompt = _build_r2v_prompt(scene)
        self.assertIn("<Picture 1> Actor 1 ", prompt)
        self.assertIn("<Picture 2> Location ", prompt)
        self.assertIn("<Picture 3> Noir ", prompt)
        self.assertTrue(prompt.endswith("final description"))

    def test_video_and_audio_tags(self):
        scene = {
            "description": "desc",
            "references": {
                "reference_video_paths": ["v1.mp4", "v2.mp4"],
                "reference_audio_paths": ["a1.mp3"],
            },
        }
        prompt = _build_r2v_prompt(scene)
        self.assertIn("<Video> ", prompt)
        self.assertIn("<Audio> ", prompt)

    def test_no_references_returns_description_only(self):
        prompt = _build_r2v_prompt({"description": "plain text"})
        self.assertEqual(prompt, "plain text")

    def test_description_preserved_at_end(self):
        scene = {
            "description": "the end of time",
            "references": {"reference_video_paths": ["v.mp4"]},
        }
        prompt = _build_r2v_prompt(scene)
        self.assertTrue(prompt.endswith("the end of time"))

    def test_1_based_tag_indexing(self):
        scene = {
            "description": "ok",
            "references": {"actor_sheet_paths": ["a.png", "b.png", "c.png"]},
        }
        prompt = _build_r2v_prompt(scene)
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 2>", prompt)
        self.assertIn("<Picture 3>", prompt)
        self.assertNotIn("<Picture 0>", prompt)

    def test_deterministic_output(self):
        scene = {
            "description": "det",
            "references": {"actor_sheet_paths": ["a.png"]},
        }
        self.assertEqual(_build_r2v_prompt(scene), _build_r2v_prompt(scene))

    def test_raises_on_missing_description(self):
        with self.assertRaises(FeverSlopValidationError):
            _build_r2v_prompt({"other": "data"})

    def test_raises_on_bad_input_type(self):
        with self.assertRaises(FeverSlopValidationError):
            _build_r2v_prompt("not a dict")  # type: ignore[arg-type]
        with self.assertRaises(FeverSlopValidationError):
            _build_r2v_prompt(None)  # type: ignore[arg-type]

    def test_video_refs_clamped_to_three(self):
        scene = {
            "description": "x",
            "references": {
                "reference_video_paths": list(range(10)),
            },
        }
        prompt = _build_r2v_prompt(scene)
        self.assertEqual(prompt.count("<Video> "), MAX_R2V_VIDEO_REFS)

    def test_audio_refs_clamped_to_three(self):
        scene = {
            "description": "x",
            "references": {
                "reference_audio_paths": list(range(10)),
            },
        }
        prompt = _build_r2v_prompt(scene)
        self.assertEqual(prompt.count("<Audio> "), MAX_R2V_AUDIO_REFS)


# ---------------------------------------------------------------------------
# _build_t2v_prompt
# ---------------------------------------------------------------------------

class BuildT2VPromptTests(unittest.TestCase):
    def test_no_keyframes_returns_description_only(self):
        prompt = _build_t2v_prompt({"description": "pure text-to-video"})
        self.assertEqual(prompt, "pure text-to-video")

    def test_start_frame_only(self):
        scene = {
            "description": "end",
            "keyframes": {"startframe_path": "s.png"},
        }
        prompt = _build_t2v_prompt(scene)
        self.assertIn("<Picture 1> first_frame ", prompt)
        self.assertTrue(prompt.endswith("end"))

    def test_both_frames(self):
        scene = {
            "description": "end",
            "keyframes": {
                "startframe_path": "s.png",
                "lastframe_path": "e.png",
            },
        }
        prompt = _build_t2v_prompt(scene)
        self.assertIn("<Picture 1> first_frame ", prompt)
        self.assertIn("<Picture 2> last_frame ", prompt)
        self.assertTrue(prompt.endswith("end"))

    def test_description_preserved(self):
        scene = {
            "description": "hello world",
            "keyframes": {"startframe_path": "s.png"},
        }
        self.assertTrue(_build_t2v_prompt(scene).endswith("hello world"))


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

class PublicWrappersTests(unittest.TestCase):
    def test_build_r2v_prompt_delegates(self):
        scene = {"description": "x", "references": {"actor_sheet_paths": ["a.png"]}}
        self.assertEqual(
            build_r2v_prompt(scene),
            "<Picture 1> Actor 1 x",
        )

    def test_build_t2v_prompt_delegates(self):
        scene = {"description": "x", "keyframes": {"startframe_path": "s.png"}}
        self.assertEqual(
            build_t2v_prompt(scene),
            "<Picture 1> first_frame x",
        )

    def test_build_r2v_prompt_raises_validations(self):
        with self.assertRaises(FeverSlopValidationError):
            build_r2v_prompt(42)  # type: ignore[arg-type]

    def test_build_t2v_prompt_raises_validations(self):
        with self.assertRaises(FeverSlopValidationError):
            build_t2v_prompt([])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
