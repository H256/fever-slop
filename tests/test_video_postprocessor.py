import unittest
from pathlib import Path
import subprocess
from unittest.mock import patch

from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.domain.postprocessing import TrimSpec


class VideoPostProcessorConcatTests(unittest.TestCase):
    def test_concat_clips_can_write_video_only_concat(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
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

    def test_concat_clips_can_retain_audio_in_named_output(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
            output = processor.concat_clips(
                concat_list=Path("concat_raw.txt"),
                output_file=Path("video_audio.mp4"),
            )

        cmd = run.call_args.args[0]
        self.assertEqual(Path("video_audio.mp4"), output)
        self.assertNotIn("-an", cmd)
        self.assertIn("-c", cmd)
        self.assertIn("copy", cmd)

    def test_original_audio_mux_maps_video_and_full_audio(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
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

    def test_ffmpeg_output_is_suppressed_by_default(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
            processor.concat_clips(
                concat_list=Path("concat_list.txt"),
                output_file=Path("final_concat.mp4"),
            )

        self.assertEqual(
            {
                "check": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            },
            run.call_args.kwargs,
        )

    def test_concat_clips_can_reencode_for_native_audio_segments(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
            processor.concat_clips(
                concat_list=Path("concat_list.txt"),
                output_file=Path("final_concat.mp4"),
                reencode=True,
            )

        cmd = run.call_args.args[0]
        self.assertIn("-c:v", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("-c:a", cmd)
        self.assertIn("aac", cmd)
        self.assertNotIn("copy", cmd)

    def test_ffmpeg_output_is_visible_in_debug_mode(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg", debug=True)

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
            processor.concat_clips(
                concat_list=Path("concat_list.txt"),
                output_file=Path("final_concat.mp4"),
            )

        self.assertEqual({"check": True}, run.call_args.kwargs)

    def test_trim_clip_pads_short_output_to_requested_frames(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")
        spec = TrimSpec(
            source_file=Path("raw.mp4"),
            output_file=Path("scene_0001.mp4"),
            fps=24,
            trim_front_frames=6,
            keep_frames=10,
            scene=1,
        )

        ffprobe_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="9\n",
            stderr="",
        )
        with (
            patch("feverslop.adapters.video_postprocessor.subprocess.run") as run,
            patch("feverslop.adapters.video_postprocessor.os.replace") as replace,
        ):
            audio_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="10.000000\n", stderr="")
            run.side_effect = [None, ffprobe_result, None, audio_result]
            output = processor.trim_clip(spec)

        self.assertEqual(Path("scene_0001.mp4"), output)
        self.assertEqual(4, run.call_count)
        pad_cmd = run.call_args_list[2].args[0]
        self.assertIn("tpad=stop_mode=clone:stop=1", pad_cmd)
        self.assertIn("-frames:v", pad_cmd)
        self.assertIn("10", pad_cmd)
        replace.assert_called_once_with(Path("scene_0001.padded.mp4"), Path("scene_0001.mp4"))

    def test_trim_clip_pads_short_audio_to_requested_duration(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")
        spec = TrimSpec(
            source_file=Path("raw.mp4"),
            output_file=Path("scene_0001.mp4"),
            fps=24,
            trim_front_frames=0,
            keep_frames=240,
            scene=1,
        )

        frame_count_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="240\n", stderr="")
        short_audio_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="1.948000\n", stderr="")
        with (
            patch("feverslop.adapters.video_postprocessor.subprocess.run") as run,
            patch("feverslop.adapters.video_postprocessor.os.replace") as replace,
        ):
            run.side_effect = [None, frame_count_result, short_audio_result, None]
            output = processor.trim_clip(spec)

        self.assertEqual(Path("scene_0001.mp4"), output)
        audio_pad_cmd = run.call_args_list[3].args[0]
        self.assertIn("-af", audio_pad_cmd)
        self.assertIn("apad", audio_pad_cmd)
        self.assertIn("-t", audio_pad_cmd)
        self.assertIn("10.000000000", audio_pad_cmd)
        replace.assert_called_once_with(Path("scene_0001.audiopad.mp4"), Path("scene_0001.mp4"))

    def test_extract_last_frame_writes_single_png(self):
        processor = VideoPostProcessor(ffmpeg_path="ffmpeg")
        frame_count_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="240\n", stderr="")

        with patch("feverslop.adapters.video_postprocessor.subprocess.run") as run:
            run.side_effect = [frame_count_result, None]
            output = processor.extract_last_frame(
                source_file=Path("scene_0001.mp4"),
                output_file=Path("keyframes/scene_0002_start.png"),
            )

        cmd = run.call_args_list[1].args[0]
        self.assertEqual(Path("keyframes/scene_0002_start.png"), output)
        self.assertIn("-vf", cmd)
        self.assertIn("select=eq(n\\,239)", cmd)
        self.assertIn("-vsync", cmd)
        self.assertIn("0", cmd)
        self.assertIn("-frames:v", cmd)
        self.assertIn("1", cmd)


if __name__ == "__main__":
    unittest.main()
