"""Tests for MovieInput Pydantic validation."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from feverslop.application.movie_common import MovieInput
from feverslop.domain.slug_utils import slugify_project_name


_VALID_STORY = "Once upon a time in a place far away there was adventure."


class TestMovieInputValidation(unittest.TestCase):
    """Verify that MovieInput rejects invalid input."""

    def test_rejects_empty_name(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="", source_type="short_story", story_text=_VALID_STORY, desired_length=12)

    def test_rejects_whitespace_name(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="   ", source_type="short_story", story_text=_VALID_STORY, desired_length=12)

    def test_rejects_slug_empty_name(self):
        """Name consisting only of punctuation should produce empty slug."""
        with self.assertRaises(ValidationError):
            MovieInput(name="!!!", source_type="short_story", story_text=_VALID_STORY, desired_length=12)

    def test_rejects_short_story_text(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="short_story", story_text="too short", desired_length=12)

    def test_rejects_invalid_source_type(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="novel", story_text=_VALID_STORY, desired_length=12)

    def test_rejects_invalid_mode(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="short_story", story_text=_VALID_STORY, desired_length=12, mode="turbo")

    def test_rejects_zero_desired_length(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="short_story", story_text=_VALID_STORY, desired_length=0)

    def test_rejects_negative_desired_length(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="short_story", story_text=_VALID_STORY, desired_length=-5)

    def test_rejects_zero_width(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="short_story", story_text=_VALID_STORY, desired_length=12, width=0)

    def test_rejects_zero_height(self):
        with self.assertRaises(ValidationError):
            MovieInput(name="Test", source_type="short_story", story_text=_VALID_STORY, desired_length=12, height=0)

    def test_accepts_valid_input(self):
        input_ = MovieInput(
            name="My Movie",
            source_type="short_story",
            story_text=_VALID_STORY,
            desired_length=12,
        )
        self.assertEqual(input_.name, "My Movie")
        self.assertEqual(input_.mode, "scaffold")
        self.assertEqual(input_.width, 1280)
        self.assertEqual(input_.height, 704)
        # confirm slug generation works (tied to name validator slug check)
        self.assertEqual(slugify_project_name(input_.name), "my-movie")

    def test_accepts_screenplay_source(self):
        input_ = MovieInput(
            name="My Screenplay",
            source_type="screenplay",
            story_text="INT. CAVE - DAY\n\nA hero walks in.",
            desired_length=20,
        )
        self.assertEqual(input_.source_type, "screenplay")

    def test_rejects_screenplay_without_scene_heading(self):
        with self.assertRaises(ValidationError):
            MovieInput(
                name="Bad Screenplay",
                source_type="screenplay",
                story_text="This is just normal prose without scene headings anywhere.",
                desired_length=20,
            )

    def test_rejects_extra_fields(self):
        """Model has extra='forbid', so unknown fields should raise."""
        with self.assertRaises(ValidationError) as ctx:
            MovieInput.model_validate({
                "name": "Test",
                "source_type": "short_story",
                "story_text": _VALID_STORY,
                "desired_length": 12,
                "unknown_field": "oops",
            })
        self.assertIn("extra_forbidden", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
