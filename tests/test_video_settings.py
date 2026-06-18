import unittest

from autoprompter.config.video_settings import VideoSettings


class VideoSettingsTests(unittest.TestCase):
    def test_scene_frame_count_matches_frame_snapped_duration(self):
        settings = VideoSettings(fps=24)

        self.assertEqual(197, settings.scene_frame_count(8.2))
        self.assertEqual(48, settings.scene_frame_count(2.0))

    def test_scene_frame_count_between_uses_absolute_frame_boundaries(self):
        settings = VideoSettings(fps=24)

        frame_counts = [
            settings.scene_frame_count_between(0.0, 2.49),
            settings.scene_frame_count_between(2.49, 4.98),
            settings.scene_frame_count_between(4.98, 7.47),
        ]

        self.assertEqual([60, 60, 59], frame_counts)
        self.assertEqual(settings.seconds_to_frame(7.47), sum(frame_counts))


if __name__ == "__main__":
    unittest.main()
