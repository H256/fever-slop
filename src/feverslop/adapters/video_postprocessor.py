from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from feverslop.domain.postprocessing import TrimSpec
from feverslop.errors import FeverSlopAdaptationError
from feverslop.utils.media_paths import write_concat_list as write_media_concat_list

FFMPEG_TIMEOUT_SECONDS = 120
FFPROBE_TIMEOUT_SECONDS = 30


class VideoPostProcessor:
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        reencode: bool = True,
        video_codec: str = "libx264",
        crf: int = 18,
        preset: str = "slow",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        debug: bool = False,
        ffmpeg_timeout_seconds: float = FFMPEG_TIMEOUT_SECONDS,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.reencode = reencode
        self.video_codec = video_codec
        self.crf = crf
        self.preset = preset
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.debug = debug
        self.ffmpeg_timeout_seconds = ffmpeg_timeout_seconds

    def trim_clip(self, spec: TrimSpec) -> Path:
        spec.output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            f"{spec.start_seconds:.9f}",
            "-i",
            str(spec.source_file),
            "-t",
            f"{spec.duration_seconds:.9f}",
        ]

        if self.reencode:
            cmd.extend([
                "-c:v", self.video_codec,
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-pix_fmt", "yuv420p",
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-movflags", "+faststart",
            ])
        else:
            cmd.extend(["-c", "copy"])

        cmd.append(str(spec.output_file))
        self._run_ffmpeg(cmd)
        self._validate_video_output(spec.output_file, spec.duration_seconds, "trim_clip")
        self._pad_short_clip(spec)
        if self.reencode:
            self._pad_short_audio(spec.output_file, spec.duration_seconds)
        if spec.extract_boundary_frames:
            self.extract_first_and_last_frames(
                spec.output_file,
                spec.output_file.with_name("firstframe.png"),
                spec.output_file.with_name("lastframe.png"),
            )
        return spec.output_file

    @staticmethod
    def _validate_video_output(video_file: Path, expected_duration: float, operation: str) -> None:
        if not video_file.is_file():
            raise FeverSlopAdaptationError(
                f"{operation} did not produce an output file: {video_file}",
            )
        size = video_file.stat().st_size
        if size < 1024:
            raise FeverSlopAdaptationError(
                f"{operation} produced a file that is too small ({size} bytes): {video_file}",
            )
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration:stream=codec_type", "-of", "json", str(video_file),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
            probe = json.loads(result.stdout)
            duration = float(probe["format"]["duration"])
            stream_types = {stream.get("codec_type") for stream in probe.get("streams", [])}
            if "video" not in stream_types:
                duration = None
        except subprocess.TimeoutExpired as exc:
            raise FeverSlopAdaptationError(
                f"{operation} timed out while probing video: {video_file}",
            ) from exc
        except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            duration = None
        if duration is None or duration < expected_duration * 0.5:
            raise FeverSlopAdaptationError(
                f"{operation} produced an invalid video: expected about "
                f"{expected_duration:.3f}s, got {duration!r}: {video_file}",
            )

    def _pad_short_clip(self, spec: TrimSpec) -> None:
        frame_count = self._frame_count(spec.output_file)
        missing_frames = spec.keep_frames - frame_count
        if missing_frames <= 0:
            return

        padded_file = spec.output_file.with_name(f"{spec.output_file.stem}.padded{spec.output_file.suffix}")
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(spec.output_file),
            "-vf", f"tpad=stop_mode=clone:stop={missing_frames}",
            "-frames:v", str(spec.keep_frames),
            "-c:v", self.video_codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(padded_file),
        ]
        self._run_ffmpeg(cmd)
        os.replace(padded_file, spec.output_file)

    @staticmethod
    def _frame_count(video_file: Path) -> int:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(video_file),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return int(result.stdout.strip())

    def _pad_short_audio(self, video_file: Path, target_duration: float) -> None:
        audio_duration = self._audio_duration(video_file)
        if audio_duration is None or audio_duration + 0.05 >= float(target_duration):
            return
        padded_file = video_file.with_name(f"{video_file.stem}.audiopad{video_file.suffix}")
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_file),
            "-c:v", "copy",
            "-af", "apad",
            "-t", f"{float(target_duration):.9f}",
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-movflags", "+faststart",
            str(padded_file),
        ]
        self._run_ffmpeg(cmd)
        os.replace(padded_file, video_file)

    @staticmethod
    def _audio_duration(video_file: Path) -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=duration",
                    "-of", "default=nokey=1:noprint_wrappers=1",
                    str(video_file),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise FeverSlopAdaptationError(
                f"Timed out while probing audio duration: {video_file}",
            ) from exc
        except subprocess.CalledProcessError:
            return None
        value = result.stdout.strip()
        if not value or value.upper() == "N/A":
            return None
        return float(value)

    def concat_clips(
        self,
        concat_list: str | Path,
        output_file: str | Path,
        video_only: bool = False,
        reencode: bool = False,
        fps: float | None = None,
        frame_count: int | None = None,
        timeout_seconds: float | None = None,
    ) -> Path:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
        ]
        if video_only and reencode:
            cmd.extend([
                "-an",
                "-c:v", self.video_codec,
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr",
            ])
            if fps is not None:
                cmd.extend(["-r", str(fps)])
            if frame_count is not None:
                cmd.extend(["-frames:v", str(frame_count)])
            cmd.extend(["-movflags", "+faststart"])
        elif video_only:
            cmd.extend(["-an", "-c:v", "copy"])
        elif reencode:
            cmd.extend([
                "-c:v", self.video_codec,
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-pix_fmt", "yuv420p",
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-movflags", "+faststart",
            ])
        else:
            cmd.extend(["-c", "copy"])
        cmd.append(str(output_file))
        self._run_ffmpeg(cmd, timeout_seconds=timeout_seconds)
        self._validate_video_output(output_file, 0.1, "concat_clips")
        return output_file

    def extract_last_frame(self, source_file: str | Path, output_file: str | Path) -> Path:
        return self._extract_frame(source_file, output_file, "last")

    def extract_first_frame(self, source_file: str | Path, output_file: str | Path) -> Path:
        return self._extract_frame(source_file, output_file, "first")

    def extract_first_and_last_frames(
        self,
        source_file: str | Path,
        first_output_file: str | Path,
        last_output_file: str | Path,
    ) -> tuple[Path, Path]:
        return (
            self.extract_first_frame(source_file, first_output_file),
            self.extract_last_frame(source_file, last_output_file),
        )

    def _extract_frame(
        self,
        source_file: str | Path,
        output_file: str | Path,
        position: str,
    ) -> Path:
        source_file = Path(source_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        if position == "last":
            frame_index = max(0, self._frame_count(source_file) - 1)
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source_file),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(output_file),
        ]
        self._run_ffmpeg(cmd)
        return output_file

    def mux_original_audio(
        self,
        video_file: str | Path,
        audio_file: str | Path,
        output_file: str | Path,
    ) -> Path:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_file),
            "-i", str(audio_file),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-shortest",
            str(output_file),
        ]
        self._run_ffmpeg(cmd)
        return output_file

    def mux_original_audio_for_diagnostics(
        self,
        video_file: str | Path,
        audio_file: str | Path,
        output_file: str | Path,
    ) -> Path:
        return self.mux_original_audio(video_file, audio_file, output_file)

    @staticmethod
    def write_concat_list(video_files: list[Path], output_file: str | Path) -> Path:
        return write_media_concat_list(video_files, output_file)

    @staticmethod
    def write_manifest(entries: list[dict], output_file: str | Path) -> Path:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_file

    def _run_ffmpeg(self, cmd: list[str], *, timeout_seconds: float | None = None) -> None:
        timeout = timeout_seconds or self.ffmpeg_timeout_seconds
        if self.debug:
            subprocess.run(cmd, check=True, timeout=timeout)
            return
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise FeverSlopAdaptationError(
                f"FFmpeg timed out after {timeout}s: {' '.join(cmd)}",
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = str(exc.stderr or "").strip()
            raise FeverSlopAdaptationError(
                f"FFmpeg failed: {exc.returncode} for command: {' '.join(cmd)}"
                + (f"\n{details}" if details else ""),
            ) from exc
