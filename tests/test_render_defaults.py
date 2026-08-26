import unittest

from feverslop.composition.project_repository import movie_default_config, movie_project_config
from feverslop.ports.project_requests import ProjectCreateRequest


class RenderDefaultsTests(unittest.TestCase):
    def test_new_project_defaults_to_draft_two_pass_without_postprocess(self):
        request = ProjectCreateRequest(project_type="standard_music_video", name="Demo")

        config = movie_default_config(request)

        self.assertEqual(
            {
                "quality": "draft",
                "pass_strategy": "two_pass",
                "postprocess": "none",
            },
            config["render_profile"],
        )

    def test_explicit_render_choices_are_persisted(self):
        request = ProjectCreateRequest(
            project_type="standard_music_video",
            name="Demo",
            render_quality="final",
            render_pass_strategy="single_pass",
            render_postprocess="seedvr",
        )

        config = movie_default_config(request)
        movie = movie_project_config(request)

        self.assertEqual(
            {"quality": "final", "pass_strategy": "single_pass", "postprocess": "seedvr"},
            config["render_profile"],
        )
        self.assertEqual(config["render_profile"], movie["render_profile"])


if __name__ == "__main__":
    unittest.main()
