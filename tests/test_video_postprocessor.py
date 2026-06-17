import unittest
from pathlib import Path
from unittest.mock import patch

from video_postprocessor import VideoPostProcessor


class VideoPostProcessorConcatTests(unittest.TestCase):
    def test_concat_clips_can_write_video_only_concat(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("video_postprocessor.subprocess.run") as run:
            output = processor.concat_clips(
                concat_list=Path("concat_list.txt"),
                output_file=Path("final_concat_video_only.mp4"),
                video_only=True,
            )

        cmd = run.call_args.args[0]
        self.assertEqual(Path("final_concat_video_only.mp4"), output)
        self.assertIn("-f", cmd)
        self.assertIn("concat", cmd)
        self.assertIn("-an", cmd)
        self.assertIn("-c:v", cmd)
        self.assertIn("copy", cmd)

    def test_original_audio_mux_maps_video_and_full_audio(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("video_postprocessor.subprocess.run") as run:
            output = processor.mux_original_audio(
                video_file=Path("final_concat_video_only.mp4"),
                audio_file=Path("song.mp3"),
                output_file=Path("final_concat.mp4"),
            )

        cmd = run.call_args.args[0]
        self.assertEqual(Path("final_concat.mp4"), output)
        self.assertIn("-map", cmd)
        self.assertIn("0:v:0", cmd)
        self.assertIn("1:a:0", cmd)
        self.assertIn("-shortest", cmd)


if __name__ == "__main__":
    unittest.main()
