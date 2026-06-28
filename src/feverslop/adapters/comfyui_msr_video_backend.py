from __future__ import annotations

from pathlib import Path
import json
import re

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.ltx_rendering import PromptRelayPayloadBuilder
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMSRVideoRenderBackend:
    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_offset: int = 100000,
        msr_frame_count: int = 17,
        debug_workflows_dir: str | Path | None = None,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        model_resolver=None,
    ):
        if int(msr_frame_count) not in {17, 25, 33, 41}:
            raise ValueError("msr_frame_count must be one of 17, 25, 33, 41")
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.output_dir = Path(output_dir)
        self.raw_output_dir = self.output_dir
        self.seed_offset = int(seed_offset)
        self.msr_frame_count = int(msr_frame_count)
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_video(self, request: VideoRenderRequest) -> Path:
        scene_number = int(request.scene_number)
        comfy_audio_name = self.asset_uploader.resolve_audio_name(
            request.audio_file,
            upload_audio=request.upload_audio,
            uploaded_audio_name=request.uploaded_audio_name,
        )
        workflow = self.build_workflow(request.scene, prompt=request.prompt, comfy_audio_name=comfy_audio_name)
        workflow = self.model_resolver.resolve_workflow_models(workflow, workflow_path=self.workflow_path)
        return self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=self.raw_output_dir / f"scene_{scene_number:04}_raw.mp4",
        )

    def build_workflow(self, scene: dict, *, prompt: str, comfy_audio_name: str | None = None) -> dict:
        scene_number = int(scene["scene"])
        references = scene.get("references") or {}
        actor_reference_paths = references.get("actor_msr_paths") or references.get("actor_sheet_paths", [])
        actor_paths = [Path(path) for path in actor_reference_paths]
        if not actor_paths:
            raise ValueError(f"Scene {scene_number} references at least 1 actor for ltx_msr")
        if len(actor_paths) > 4:
            raise ValueError(f"Scene {scene_number} references at most 4 actors for ltx_msr")
        location_path = references.get("location_msr_path") or references.get("location_sheet_path")
        if not location_path:
            raise ValueError(f"Scene {scene_number} is missing references.location_msr_path")

        patcher = WorkflowPatcher(self.load_workflow())
        self._patch_actor_reference_inputs(patcher, actor_paths)
        patcher.set_input_by_title(
            "#MSR_BACKGROUND",
            "image",
            self.asset_uploader.resolve_reference_image_name(location_path),
        )
        self._patch_prompt_inputs(patcher, scene, prompt=prompt)
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", f"ltx_msr_raw/scene_{scene_number:04}")
        patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "frame_count", self.msr_frame_count)
        patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "value", self.msr_frame_count)
        patcher.try_set_existing_input_by_title("#SEED", "noise_seed", self.seed_offset + scene_number)
        patcher.try_set_existing_input_by_title("#WIDTH", "value", int(scene.get("width", 0) or 0))
        patcher.try_set_existing_input_by_title("#HEIGHT", "value", int(scene.get("height", 0) or 0))
        patcher.try_set_existing_input_by_title("#FRAMES", "value", int(scene.get("frame_count", 0) or 0))
        patcher.try_set_existing_input_by_title("#FRAMERATE", "value", int(scene.get("fps", 0) or 0))
        if comfy_audio_name:
            self._patch_audio_inputs(patcher, scene, comfy_audio_name=comfy_audio_name)

        workflow = patcher.get()
        if self.debug_workflows_dir:
            self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json").write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return workflow

    def _patch_actor_reference_inputs(self, patcher: WorkflowPatcher, actor_paths: list[Path]) -> None:
        for index, actor_path in enumerate(actor_paths, start=1):
            title = f"#MSR_ACTOR_{index}"
            image_name = self.asset_uploader.resolve_reference_image_name(actor_path)
            if self._has_anchor(patcher, title):
                node_id, node = patcher.find_node_by_meta_title(title)
                node.setdefault("inputs", {})["image"] = image_name
            else:
                node_id = self._add_actor_reference_node(patcher, index=index, image_name=image_name)
            self._connect_msr_actor_input(patcher, actor_index=index, actor_node_id=node_id)

    @staticmethod
    def _add_actor_reference_node(patcher: WorkflowPatcher, *, index: int, image_name: str) -> str:
        workflow = patcher.get()
        numeric_ids = [int(node_id) for node_id in workflow if str(node_id).isdigit()]
        node_id = str(max(numeric_ids, default=0) + 1)
        workflow[node_id] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": f"#MSR_ACTOR_{index}"},
        }
        return node_id

    @staticmethod
    def _connect_msr_actor_input(patcher: WorkflowPatcher, *, actor_index: int, actor_node_id: str) -> None:
        try:
            _, msr_node = patcher.find_node_by_meta_title("#MSR_FRAME_COUNT")
        except KeyError:
            return
        msr_node.setdefault("inputs", {})[str(actor_index)] = [str(actor_node_id), 0]

    @staticmethod
    def _patch_audio_inputs(patcher: WorkflowPatcher, scene: dict, *, comfy_audio_name: str) -> None:
        patcher.try_set_existing_input_by_title("#LOAD_AUDIO", "audio", comfy_audio_name)
        patcher.try_set_existing_input_by_title(
            "#LOAD_AUDIO",
            "audioUI",
            f"/api/view?filename={comfy_audio_name}&type=input",
        )
        fps = int(scene.get("fps", 0) or 0)
        frame_count = int(scene.get("frame_count", 0) or 0)
        duration = scene.get("duration_seconds")
        if duration is None and fps > 0 and frame_count > 0:
            duration = max(0.0, (frame_count - 1) / float(fps))
        patcher.try_set_existing_input_by_title("#TRIM_AUDIO", "start_index", float(scene.get("abs_start_seconds", 0.0) or 0.0))
        patcher.try_set_existing_input_by_title("#TRIM_AUDIO", "duration", float(duration or 0.0))

    def _patch_prompt_inputs(self, patcher: WorkflowPatcher, scene: dict, *, prompt: str) -> None:
        if self._has_anchor(patcher, "#PROMPT_RELAY"):
            global_prompt, local_prompts, segment_lengths = self._build_prompt_relay_payload(scene, prompt=prompt)
            patcher.set_input_by_title("#PROMPT_RELAY", "global_prompt", global_prompt)
            patcher.set_input_by_title("#PROMPT_RELAY", "local_prompts", local_prompts)
            patcher.set_input_by_title("#PROMPT_RELAY", "segment_lengths", segment_lengths)
            return
        patcher.set_input_by_title("#PROMPT", "text", str(prompt).strip())

    @staticmethod
    def _build_prompt_relay_payload(scene: dict, *, prompt: str) -> tuple[str, str, str]:
        frame_count = int(scene.get("frame_count", 1) or 1)
        ltx = scene.get("ltx") or {}
        reference_global_prompt = _build_msr_reference_global_prompt(scene.get("references") or {})
        if ltx.get("prompt_relay"):
            payload = PromptRelayPayloadBuilder().build(
                scene=scene,
                render_frame_count=frame_count,
                trim_front_frames=0,
                tail_loss_frames=0,
            )
            motion_prompt = _build_msr_motion_prompt(scene, fallback_prompt=prompt)
            local_prompts = _clean_msr_local_prompts(
                payload.local_prompts,
                base_prompt=str(ltx.get("base_prompt") or ""),
            )
            if motion_prompt:
                local_prompts = _replace_msr_local_prompt_text(local_prompts, motion_prompt)
            return reference_global_prompt or payload.global_prompt, local_prompts, payload.segment_lengths

        global_prompt = str(ltx.get("base_prompt") or ltx.get("original_style_i2v_prompt") or prompt).strip()
        local_prompts = _build_msr_motion_prompt(scene, fallback_prompt=prompt)
        local_prompts = local_prompts or _clean_msr_local_prompt(str(prompt).strip(), base_prompt=global_prompt)
        local_prompts = local_prompts or "continue the main scene motion with stable subject identity"
        segment_lengths = str(max(1, frame_count - 1))
        return reference_global_prompt or global_prompt, local_prompts, segment_lengths

    @staticmethod
    def _has_anchor(patcher: WorkflowPatcher, title: str) -> bool:
        try:
            patcher.find_node_by_meta_title(title)
            return True
        except KeyError:
            return False


def _build_msr_reference_global_prompt(references: dict) -> str:
    parts: list[str] = []
    for index, actor in enumerate(references.get("actor_reference_descriptions") or [], start=1):
        actor_text = _describe_reference_item(actor)
        if actor_text:
            parts.append(
                f"Reference image {index}: {actor_text}. "
                f"Use reference image {index} for this subject's identity, face, body, wardrobe, and materials."
            )

    location = references.get("location_reference_description") or {}
    location_text = _describe_reference_item(location)
    if location_text:
        parts.append(
            f"Background reference: {location_text}. "
            "Use this image as the scene environment, lighting, color palette, atmosphere, and spatial setting."
        )

    return " ".join(parts).strip()


def _describe_reference_item(item: dict) -> str:
    name = str(item.get("name") or item.get("id") or "").strip()
    role = str(item.get("role") or "").strip()
    visual = str(item.get("visual_description") or item.get("image_prompt") or "").strip()
    chunks = [chunk for chunk in (name, role, visual) if chunk]
    return ", ".join(chunks)


def _clean_msr_local_prompts(local_prompts: str, *, base_prompt: str) -> str:
    cleaned = [
        _clean_msr_local_prompt(prompt, base_prompt=base_prompt)
        for prompt in str(local_prompts).split("\n|")
    ]
    return "\n|".join(prompt for prompt in cleaned if prompt)


def _replace_msr_local_prompt_text(local_prompts: str, replacement: str) -> str:
    segments = str(local_prompts or "").split("\n|")
    return "\n|".join(str(replacement).strip() for segment in segments if str(segment).strip())


def _clean_msr_local_prompt(prompt: str, *, base_prompt: str) -> str:
    cleaned = str(prompt or "").strip()
    base = str(base_prompt or "").strip()
    if base and cleaned.startswith(base):
        cleaned = cleaned[len(base):].lstrip(" .")
    cleaned = re.sub(r"(?is)\bStart frame:\s*", "", cleaned).strip()
    cleaned = re.sub(
        r"(?is)\bLock the first frame\b.*?(?:\.|$)",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(
        r"(?is)\bcontinue directly from it\b.*?(?:\.|$)",
        "",
        cleaned,
    ).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _build_msr_motion_prompt(scene: dict, *, fallback_prompt: str) -> str:
    ltx = scene.get("ltx") or {}
    source = str(ltx.get("base_prompt") or ltx.get("original_style_i2v_prompt") or fallback_prompt or "")
    parts: list[str] = []
    for label in ("Camera motion", "Subject or environment motion", "Story beat"):
        match = re.search(rf"(?is){re.escape(label)}\s*:\s*(.*?)(?:\.\.|\. (?=[A-Z][A-Za-z ]+:)|$)", source)
        if match:
            text = re.sub(r"\s+", " ", match.group(1)).strip(" .")
            if text:
                parts.append(f"{label}: {text}.")
    return " ".join(parts).strip()
