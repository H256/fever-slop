from __future__ import annotations

from pathlib import Path
from typing import Any


class RunPipelineAdapter:
    def run(self, *, project_config_path: Path, options: dict[str, Any]) -> Path | None:
        import run_pipeline

        argv = [
            "--project-config",
            str(project_config_path),
        ]
        self._append_value(argv, "--app-config", options.get("app_config"))
        self._append_value(argv, "--concept-batch-size", options.get("concept_batch_size"))
        self._append_value(argv, "--storyboard-workflow", options.get("storyboard_workflow"))
        self._append_value(argv, "--relay-workflow", options.get("relay_workflow"))
        self._append_value(argv, "--single-prompt-workflow", options.get("single_prompt_workflow"))
        self._append_value(argv, "--render-mode", options.get("render_mode"))
        self._append_value(argv, "--single-prompt-title", options.get("single_prompt_title"))
        self._append_value(argv, "--single-prompt-input", options.get("single_prompt_input"))
        self._append_value(argv, "--rolling-frame-profile", options.get("rolling_frame_profile"))
        self._append_value(argv, "--storyboard-lora-strength", options.get("storyboard_lora_strength"))
        self._append_value(argv, "--video-character-lora-strength", options.get("video_character_lora_strength"))
        self._append_value(argv, "--video-lora-1-strength-model", options.get("video_lora_1_strength_model"))
        self._append_value(argv, "--video-lora-1-strength-clip", options.get("video_lora_1_strength_clip"))
        self._append_lora_split(argv, options.get("lora_split_enabled"))
        self._append_value(argv, "--smoke-scene", options.get("smoke_scene"))
        self._append_flag(argv, "--smoke-only", options.get("smoke_only"))
        self._append_flag(argv, "--no-skip-existing", options.get("no_skip_existing"))
        self._append_flag(argv, "--skip-tests", options.get("skip_tests"))
        self._append_flag(argv, "--skip-main-pipeline", options.get("skip_main_pipeline"))
        self._append_flag(argv, "--skip-relay-compact", options.get("skip_relay_compact"))
        self._append_flag(argv, "--skip-anchor-fix", options.get("skip_anchor_fix"))
        self._append_flag(argv, "--skip-storyboard", options.get("skip_storyboard"))
        self._append_flag(argv, "--skip-storyboard-page", options.get("skip_storyboard_page"))
        self._append_flag(argv, "--skip-ltx", options.get("skip_ltx"))
        self._append_flag(argv, "--skip-final-concat", options.get("skip_final_concat"))
        self._append_flag(argv, "--diagnostic-original-audio-mux", options.get("diagnostic_original_audio_mux"))
        self._append_flag(argv, "--no-original-audio-mux", options.get("no_original_audio_mux"))

        result = run_pipeline.run(run_pipeline.build_arg_parser().parse_args(argv))
        return result.final_video_path

    @staticmethod
    def _append_value(argv: list[str], flag: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str) and value == "":
            return
        argv.extend([flag, str(value)])

    @staticmethod
    def _append_flag(argv: list[str], flag: str, enabled: Any) -> None:
        if enabled:
            argv.append(flag)

    @staticmethod
    def _append_lora_split(argv: list[str], value: Any) -> None:
        if value is True:
            argv.append("--lora-split-enabled")
        elif value is False:
            argv.append("--no-lora-split-enabled")
