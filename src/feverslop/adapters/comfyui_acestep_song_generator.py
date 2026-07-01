from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.full_auto import GeneratedSong, SongSpec


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
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

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
                    f"ACE-STEP workflow anchor {title} is missing inputs: {sorted(missing)}"
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
        return GeneratedSong(
            audio_path=downloaded,
            manifest={
                "prompt_id": prompt_id,
                "seed": seed,
                "workflow_path": str(self.workflow_path),
            },
        )

    def _write_debug_workflow(self, *, output_dir: Path, workflow: dict) -> None:
        project_dir = output_dir.parent
        debug_dir = project_dir / "output" / "debug" / "ace_step"
        debug_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (debug_dir / f"ace_step_{run_id}_workflow.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
        raise RuntimeError("No audio output found in ACE-STEP ComfyUI history")

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
                        }
                    )
        return files
