from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import json


@dataclass(frozen=True)
class TrimSpec:
    source_file: Path
    output_file: Path
    fps: int
    trim_front_frames: int
    keep_frames: int
    scene: int

    @property
    def start_seconds(self) -> float:
        return self.trim_front_frames / float(self.fps)

    @property
    def duration_seconds(self) -> float:
        return self.keep_frames / float(self.fps)


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
    ):
        self.ffmpeg_path = ffmpeg_path
        self.reencode = reencode
        self.video_codec = video_codec
        self.crf = crf
        self.preset = preset
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate

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
        subprocess.run(cmd, check=True)
        return spec.output_file

    def concat_clips(self, concat_list: str | Path, output_file: str | Path, video_only: bool = False) -> Path:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
        ]
        if video_only:
            cmd.extend(["-an", "-c:v", "copy"])
        else:
            cmd.extend(["-c", "copy"])
        cmd.append(str(output_file))
        subprocess.run(cmd, check=True)
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
        subprocess.run(cmd, check=True)
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
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for path in video_files:
            absolute = Path(path).resolve()
            escaped = str(absolute).replace("'", r"'\''")
            lines.append(f"file '{escaped}'")

        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_file

    @staticmethod
    def write_manifest(entries: list[dict], output_file: str | Path) -> Path:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_file
