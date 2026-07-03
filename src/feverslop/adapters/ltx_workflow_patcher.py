from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.adapters.lora_workflow_patcher import LoraPatchSettings, LoraWorkflowPatcher
from feverslop.domain.ltx_rendering import AudioWindowSpec, PromptRelayPayloadBuilder


@dataclass(frozen=True)
class ResolvedLoraConfig:
    index: int
    enabled: bool = False
    name: str = ""
    strength_model: float = 1.0
    strength_clip: float = 1.0
    name_explicit: bool = False
    strength_model_explicit: bool = False
    strength_clip_explicit: bool = False

    @property
    def lora_title(self) -> str:
        return f"#LORA_{self.index}"

    @property
    def split_title(self) -> str:
        return f"#SPLIT_LORA_{self.index}"


@dataclass
class LTXWorkflowSettings:
    ltx_workflow_path: Path
    single_prompt_workflow_path: Path | None
    render_mode: str
    width_node_title: str
    height_node_title: str
    load_audio_node_title: str
    trim_audio_node_title: str
    startframe_node_title: str
    frames_node_title: str
    framerate_node_title: str
    seed_node_title: str
    prompt_relay_node_title: str
    single_prompt_node_title: str
    single_prompt_input_name: str
    save_video_node_title: str
    character_lora_node_title: str | None
    character_lora_strength: float | None
    lora_1_enabled: bool
    lora_1_name: str
    lora_1_strength_model: float
    lora_1_strength_clip: float
    lora_1_strengths_explicit: bool
    lora_1_node_title: str
    randomize_seed: bool
    seed_offset: int
    segment_length_mode: str
    debug_workflows_dir: Path | None
    loras: tuple[ResolvedLoraConfig, ...] = ()
    lora_split_enabled: bool = False


class LTXWorkflowPatcher:
    def __init__(self, settings: LTXWorkflowSettings):
        self.settings = settings
        self.lora_patcher = LoraWorkflowPatcher(
            LoraPatchSettings(
                character_lora_node_title=settings.character_lora_node_title,
                character_lora_strength=settings.character_lora_strength,
                lora_1_enabled=settings.lora_1_enabled,
                lora_1_name=settings.lora_1_name,
                lora_1_strength_model=settings.lora_1_strength_model,
                lora_1_strength_clip=settings.lora_1_strength_clip,
                lora_1_strengths_explicit=settings.lora_1_strengths_explicit,
                lora_1_node_title=settings.lora_1_node_title,
                lora_split_enabled=settings.lora_split_enabled,
            ),
            settings.loras,
        )

    def workflow_path_for_mode(self, mode: str) -> Path:
        if mode == "single_prompt":
            return self.settings.single_prompt_workflow_path or self.settings.ltx_workflow_path
        return self.settings.ltx_workflow_path

    def load_workflow(self, mode: str = "relay") -> dict:
        return json.loads(self.workflow_path_for_mode(mode).read_text(encoding="utf-8-sig"))

    def validate_workflow(self, mode: str = "relay") -> None:
        workflow_path = self.workflow_path_for_mode(mode)
        patcher = WorkflowPatcher(json.loads(workflow_path.read_text(encoding="utf-8-sig")))

        required_titles = [
            self.settings.width_node_title,
            self.settings.height_node_title,
            self.settings.load_audio_node_title,
            self.settings.trim_audio_node_title,
            self.settings.startframe_node_title,
            self.settings.frames_node_title,
            self.settings.framerate_node_title,
            self.settings.seed_node_title,
            self.settings.save_video_node_title,
        ]
        if mode != "single_prompt":
            required_titles.append(self.settings.prompt_relay_node_title)
        for lora in self.active_loras():
            required_titles.append(lora.lora_title)

        for title in dict.fromkeys(required_titles):
            try:
                patcher.find_node_by_meta_title(title)
            except KeyError as exc:
                raise ValueError(f"Missing workflow anchor {title} in workflow file {workflow_path}") from exc

        if mode == "single_prompt":
            prompt_title_candidates = [
                self.settings.single_prompt_node_title,
                "#PROMPT_POSITIVE",
                "#PROMPT",
            ]
            for title in dict.fromkeys(prompt_title_candidates):
                try:
                    patcher.find_node_by_meta_title(title)
                    break
                except KeyError:
                    continue
            else:
                anchors = ", ".join(dict.fromkeys(prompt_title_candidates))
                raise ValueError(f"Missing workflow anchor {anchors} in workflow file {workflow_path}")

    def render_mode_for_scene(self, scene: dict) -> str:
        if self.settings.render_mode != "auto":
            return self.settings.render_mode

        hint = str(scene.get("ltx", {}).get("render_mode_hint", "relay")).strip()
        if hint == "single_prompt":
            return "single_prompt"
        return "relay"

    def seed_for_scene(self, scene_number: int) -> int:
        if self.settings.randomize_seed:
            return random.randint(0, 2**63 - 1)
        return self.settings.seed_offset + scene_number

    def build_workflow(
        self,
        *,
        scene: dict,
        comfy_audio_name: str,
        comfy_startframe_name: str,
        rolling: AudioWindowSpec,
    ) -> dict:
        mode = self.render_mode_for_scene(scene)
        self.validate_workflow(mode=mode)
        patcher = WorkflowPatcher(self.load_workflow(mode=mode))

        scene_number = int(scene["scene"])
        fps = int(scene["fps"])
        width = int(scene["width"])
        height = int(scene["height"])
        render_frame_count = int(rolling["render_frame_count"])

        patcher.set_input_by_title(self.settings.width_node_title, "value", width)
        patcher.set_input_by_title(self.settings.height_node_title, "value", height)
        patcher.set_input_by_title(self.settings.frames_node_title, "value", render_frame_count)
        patcher.set_input_by_title(self.settings.framerate_node_title, "value", fps)
        patcher.set_input_by_title(self.settings.seed_node_title, "noise_seed", self.seed_for_scene(scene_number))

        patcher.set_input_by_title(self.settings.load_audio_node_title, "audio", comfy_audio_name)
        patcher.try_set_existing_input_by_title(
            self.settings.load_audio_node_title,
            "audioUI",
            f"/api/view?filename={comfy_audio_name}&type=input",
        )

        patcher.set_input_by_title(self.settings.trim_audio_node_title, "start_index", float(rolling["audio_start_seconds"]))
        patcher.set_input_by_title(self.settings.trim_audio_node_title, "duration", float(rolling["audio_duration_seconds"]))
        patcher.set_input_by_title(self.settings.startframe_node_title, "image", comfy_startframe_name)

        self.patch_prompt_inputs(
            patcher=patcher,
            scene=scene,
            mode=mode,
            render_frame_count=render_frame_count,
            trim_front_frames=int(rolling["trim_front_frames"]),
            tail_loss_frames=int(rolling["tail_loss_frames"]),
        )

        patcher.set_input_by_title(self.settings.save_video_node_title, "filename_prefix", f"ltx_raw/scene_{scene_number:04}")
        self.patch_lora_inputs(patcher)

        workflow = patcher.get()
        if self.settings.debug_workflows_dir:
            self.settings.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json").write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return workflow

    def patch_lora_inputs(self, patcher: WorkflowPatcher) -> None:
        self.lora_patcher.patch_lora_inputs(patcher)

    def active_loras(self) -> tuple[ResolvedLoraConfig, ...]:
        return self.lora_patcher.active_loras()

    def patch_prompt_inputs(
        self,
        patcher: WorkflowPatcher,
        scene: dict,
        mode: str,
        render_frame_count: int,
        trim_front_frames: int,
        tail_loss_frames: int,
    ) -> None:
        if mode == "single_prompt":
            prompt = (
                scene.get("ltx", {}).get("original_style_i2v_prompt")
                or scene.get("ltx", {}).get("base_prompt")
                or ""
            )
            prompt_title_candidates = [
                self.settings.single_prompt_node_title,
                "#PROMPT_POSITIVE",
                "#PROMPT",
            ]
            last_error = None
            for title in dict.fromkeys(prompt_title_candidates):
                try:
                    patcher.set_input_by_title(
                        title,
                        self.settings.single_prompt_input_name,
                        str(prompt).strip(),
                    )
                    return
                except KeyError as exc:
                    last_error = exc

            raise last_error or KeyError(
                f"No prompt node found for single_prompt mode. Tried: {prompt_title_candidates}"
            )

        global_prompt, local_prompts, segment_lengths = self.build_prompt_relay_payload(
            scene=scene,
            render_frame_count=render_frame_count,
            trim_front_frames=trim_front_frames,
            tail_loss_frames=tail_loss_frames,
        )
        patcher.set_input_by_title(self.settings.prompt_relay_node_title, "global_prompt", global_prompt)
        patcher.set_input_by_title(self.settings.prompt_relay_node_title, "local_prompts", local_prompts)
        patcher.set_input_by_title(self.settings.prompt_relay_node_title, "segment_lengths", segment_lengths)

    def build_prompt_relay_payload(
        self,
        scene: dict,
        render_frame_count: int,
        trim_front_frames: int,
        tail_loss_frames: int,
    ) -> tuple[str, str, str]:
        payload = PromptRelayPayloadBuilder(
            segment_length_mode=self.settings.segment_length_mode,
        ).build(
            scene=scene,
            render_frame_count=render_frame_count,
            trim_front_frames=trim_front_frames,
            tail_loss_frames=tail_loss_frames,
        )
        return payload.global_prompt, payload.local_prompts, payload.segment_lengths

    @classmethod
    def normalize_prompt_relay_segments(cls, segments: list[dict]) -> list[dict]:
        return PromptRelayPayloadBuilder.normalize_segments(segments)
