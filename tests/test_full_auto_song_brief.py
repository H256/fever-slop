import unittest


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_prompt(self, system_prompt, prompt):
        self.calls.append((system_prompt, prompt))
        return self.response


class SongBriefGeneratorTests(unittest.TestCase):
    def test_llm_song_brief_generator_returns_song_spec_with_request_defaults(self):
        from feverslop.adapters.llm_song_brief_generator import LLMSongBriefGenerator
        from feverslop.application.full_auto import FullAutoRequest

        generator = LLMSongBriefGenerator(
            FakeLLM(
                """
                {
                  "title": "Joy Demo",
                  "tags": "bright pop song",
                  "lyrics": "[Verse]\\nhello",
                  "bpm": 123,
                  "language": "en",
                  "keyscale": "D major",
                  "visual_story_idea": "friends",
                  "visual_style": "warm"
                }
                """
            )
        )

        spec = generator.generate(
            FullAutoRequest(
                idea="friendship",
                style="bright pop",
                duration_seconds=90.5,
                language="en",
            )
        )

        self.assertEqual("Joy Demo", spec.title)
        self.assertEqual("bright pop song", spec.tags)
        self.assertEqual("[Verse]\nhello", spec.lyrics)
        self.assertEqual(123, spec.bpm)
        self.assertEqual(90.5, spec.duration_seconds)
        self.assertEqual("en", spec.language)
        self.assertEqual("D major", spec.keyscale)
        self.assertEqual("friends", spec.visual_story_idea)
        self.assertEqual("warm", spec.visual_style)

    def test_system_prompt_adapts_ace_step_prompt_designer_to_json_schema(self):
        from feverslop.adapters.llm_song_brief_generator import LLMSongBriefGenerator

        prompt = LLMSongBriefGenerator._system_prompt()

        self.assertIn("Return ONLY valid JSON", prompt)
        self.assertIn('"tags"', prompt)
        self.assertIn('"lyrics"', prompt)
        self.assertIn('"visual_story_idea"', prompt)
        self.assertIn("caption", prompt)
        self.assertIn("timeline", prompt)
        self.assertIn("casting call", prompt)
        self.assertIn("ingredient has a behavior", prompt)
        self.assertIn("Do not use code fences", prompt)
        self.assertIn("6-10 syllables", prompt)
        self.assertIn("metadata", prompt)


if __name__ == "__main__":
    unittest.main()
