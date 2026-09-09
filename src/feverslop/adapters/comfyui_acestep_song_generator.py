from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.adapters.workflow_debug import write_debug_workflow
from feverslop.domain.full_auto import GeneratedSong, SongSpec
from feverslop.errors import FeverSlopRenderError


class ComfyUIAceStepSongGenerator:
    ace_title = "ACE_STEP"
    sampler_title = "KSampler"
    latent_title = "Empty Ace Step 1.5 Latent Audio"
    save_title = "SAVE"

    required_inputs = {
        ace_title: {
            "tags",
            "lyrics",
            "seed",
            "bpm",
            "duration",
            "timesignature",
            "language",
            "keyscale",
        },
        sampler_title: {"seed"},
        latent_title: {"seconds"},
        save_title: {"filename_prefix", "quality"},
    }

    def __init__(
        self,
        *,
        client: Any,
        workflow_path: str | Path,
        model_resolver: Any | None = None,
        audio_normalizer: Callable[[Path], bool] | None = None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()
        self.audio_normalizer = audio_normalizer or self._ensure_decodable_audio

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def validate_workflow(self) -> None:
        workflow = self.load_workflow()
        patcher = WorkflowPatcher(workflow)
        for title, inputs in self.required_inputs.items():
            try:
                _, node = patcher.find_node_by_meta_title(title)
            except KeyError as exc:
                raise ValueError(f"Missing ACE-STEP workflow anchor: {title}") from exc
            node_inputs = set(node.get("inputs", {}))
            missing = inputs - node_inputs
            if missing:
                raise ValueError(
                    f"ACE-STEP workflow anchor {title} is missing inputs: {sorted(missing)}",
                )

    def generate(
        self,
        spec: SongSpec,
        *,
        project_slug: str,
        output_dir: Path,
        seed: int,
    ) -> GeneratedSong:
        self.validate_workflow()
        patcher = WorkflowPatcher(self.load_workflow())
        seed = int(seed)
        duration = float(spec.duration_seconds)

        patcher.set_existing_input_by_title(self.ace_title, "tags", spec.tags)
        patcher.set_existing_input_by_title(self.ace_title, "lyrics", spec.lyrics)
        patcher.set_existing_input_by_title(self.ace_title, "bpm", int(spec.bpm))
        patcher.set_existing_input_by_title(self.ace_title, "duration", duration)
        patcher.set_existing_input_by_title(self.ace_title, "language", spec.language)
        patcher.set_existing_input_by_title(self.ace_title, "keyscale", spec.keyscale)
        patcher.set_existing_input_by_title(self.ace_title, "timesignature", "4")
        patcher.set_existing_input_by_title(self.ace_title, "seed", seed)
        patcher.set_existing_input_by_title(self.sampler_title, "seed", seed)
        patcher.set_existing_input_by_title(self.latent_title, "seconds", duration)
        patcher.set_existing_input_by_title(self.save_title, "filename_prefix", f"audio/{project_slug}")

        workflow = self.model_resolver.resolve_workflow_models(
            patcher.get(),
            workflow_path=self.workflow_path,
        )
        self._write_debug_workflow(output_dir=Path(output_dir), workflow=workflow)
        prompt_id = self.client.queue_prompt(workflow)
        history = self.client.wait_for_completion(prompt_id)
        output = self._first_audio_output(history)
        output_path = Path(output_dir) / f"{project_slug}.mp3"
        downloaded = self.client.download_view_file(
            filename=output["filename"],
            subfolder=output.get("subfolder", ""),
            file_type=output.get("type", "output"),
            output_path=output_path,
        )
        audio_normalized = self.audio_normalizer(downloaded)
        return GeneratedSong(
            audio_path=downloaded,
            manifest={
                "prompt_id": prompt_id,
                "seed": seed,
                "workflow_path": str(self.workflow_path),
                "audio_normalized": audio_normalized,
            },
        )

    @staticmethod
    def _ensure_decodable_audio(path: Path) -> bool:
        """Normalize malformed ComfyUI audio before handing it to Demucs."""
        if ComfyUIAceStepSongGenerator._audio_is_decodable(path):
            if not ComfyUIAceStepSongGenerator._audio_has_signal(path):
                raise FeverSlopRenderError(
                    "ComfyUI returned a silent ACE-STEP audio file; refusing to continue to Demucs",
                )
            return False

        original = path.with_name(f"{path.stem}.comfyui-original{path.suffix}")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.stem}.", suffix=".mp3", delete=False,
            ) as handle:
                temporary = Path(handle.name)
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "mp3", "-i", str(path),
                 "-map", "0:a:0", "-c:a", "libmp3lame", "-q:a", "2", str(temporary)],
                check=True,
                capture_output=True,
                text=True,
            )
            if not temporary.stat().st_size:
                raise ValueError("FFmpeg produced an empty normalized audio file")
            if not ComfyUIAceStepSongGenerator._audio_has_signal(temporary):
                raise ValueError("FFmpeg produced a silent normalized audio file")
            os.replace(path, original)
            os.replace(temporary, path)
            return True
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise FeverSlopRenderError(
                f"ComfyUI returned an undecodable audio file and normalization failed: {exc}",
            ) from exc

    @staticmethod
    def _audio_is_decodable(path: Path) -> bool:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return probe.returncode == 0 and bool(probe.stdout.strip())

    @staticmethod
    def _audio_has_signal(path: Path) -> bool:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path),
             "-af", "highpass=f=20,volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
        return result.returncode == 0 and match is not None and float(match.group(1)) > -45.0

    def _write_debug_workflow(self, *, output_dir: Path, workflow: dict) -> None:
        project_dir = output_dir.parent
        debug_dir = project_dir / "output" / "debug" / "ace_step"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_debug_workflow(
            debug_dir / f"ace_step_{run_id}_workflow.json",
            workflow,
        )

    def _first_audio_output(self, history: dict) -> dict:
        if hasattr(self.client, "extract_output_files"):
            files = self.client.extract_output_files(history)
        else:
            files = self._extract_output_files(history)
        for item in files:
            filename = str(item.get("filename", ""))
            if item.get("kind") == "audio" or filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                return item
        raise FeverSlopRenderError("No audio output found in ACE-STEP ComfyUI history")

    @staticmethod
    def _extract_output_files(history: dict) -> list[dict]:
        files = []
        for node_output in history.get("outputs", {}).values():
            for key in ("files", "videos", "audio"):
                for item in node_output.get(key, []):
                    files.append(
                        {
                            "kind": key,
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        },
                    )
        return files
