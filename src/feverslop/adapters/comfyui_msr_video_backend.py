from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import random
import re

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import TrimSpec, VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.ltx_rendering import (
    AudioWindowSpec,
    PromptRelayPayload,
    PromptRelayPayloadBuilder,
    build_audio_window_spec,
)
from feverslop.path_utils import coerce_local_path
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMSRVideoRenderBackend:
    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_offset: int = 100000,
        randomize_seed: bool = False,
        msr_frame_count: int = 17,
        debug_workflows_dir: str | Path | None = None,
        preroll_frames: int = 50,
        tail_loss_frames: int = 25,
        round_render_frames_to_8n1: bool = True,
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
        ffmpeg_debug: bool = False,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        project_dir: str | Path | None = None,
        workflow: dict | None = None,
        workflow_label: str | Path | None = None,
    ):
        if int(msr_frame_count) not in {17, 25, 33, 41}:
            raise ValueError("msr_frame_count must be one of 17, 25, 33, 41")
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.workflow = deepcopy(workflow) if workflow is not None else None
        self.workflow_label = Path(workflow_label) if workflow_label is not None else self.workflow_path
        self.output_dir = Path(output_dir)
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.raw_output_dir = self.output_dir / "raw"
        self.seed_offset = int(seed_offset)
        self.randomize_seed = bool(randomize_seed)
        self.msr_frame_count = int(msr_frame_count)
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None
        self.preroll_frames = max(0, int(preroll_frames))
        self.tail_loss_frames = max(0, int(tail_loss_frames))
        self.round_render_frames_to_8n1 = bool(round_render_frames_to_8n1)
        self.postprocess = bool(postprocess)
        self.postprocess_reencode = bool(postprocess_reencode)
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.postprocessor = postprocessor or VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
            debug=ffmpeg_debug,
        )
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def load_workflow(self) -> dict:
        if self.workflow is not None:
            return deepcopy(self.workflow)
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_video(self, request: VideoRenderRequest) -> Path:
        scene_number = int(request.scene_number)
        rolling = self._rolling_spec(request.scene)
        comfy_audio_name = None
        if request.upload_audio or request.uploaded_audio_name:
            comfy_audio_name = self.asset_uploader.resolve_audio_name(
                request.audio_file,
                upload_audio=request.upload_audio,
                uploaded_audio_name=request.uploaded_audio_name,
            )
        workflow = self.build_workflow(
            request.scene,
            prompt=request.prompt,
            comfy_audio_name=comfy_audio_name,
            rolling=rolling,
        )
        workflow = self.model_resolver.resolve_workflow_models(workflow, workflow_path=self.workflow_label)
        self._write_debug_workflow(scene_number, workflow)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=self.raw_output_dir / f"scene_{scene_number:04}_raw.mp4",
        )
        if not self.postprocess:
            return raw_output

        return self.postprocessor.trim_clip(
            TrimSpec(
                source_file=raw_output,
                output_file=self.output_dir / f"scene_{scene_number:04}.mp4",
                fps=int(rolling["fps"]),
                trim_front_frames=int(rolling["trim_front_frames"]),
                keep_frames=int(rolling["scene_frame_count"]),
                scene=scene_number,
            )
        )

    def build_workflow(
        self,
        scene: dict,
        *,
        prompt: str,
        comfy_audio_name: str | None = None,
        rolling: AudioWindowSpec | None = None,
    ) -> dict:
        scene_number = int(scene["scene"])
        render_frame_count = int(rolling["render_frame_count"]) if rolling else int(scene.get("frame_count", 0) or 0)
        references = scene.get("references") or {}
        actor_reference_paths = references.get("actor_msr_paths") or references.get("actor_sheet_paths", [])
        actor_paths = [self._resolve_project_path(path) for path in actor_reference_paths]
        if not actor_paths:
            raise ValueError(f"Scene {scene_number} references at least 1 actor for ltx_msr")
        if len(actor_paths) > 4:
            raise ValueError(f"Scene {scene_number} references at most 4 actors for ltx_msr")
        location_path = references.get("location_msr_path") or references.get("location_sheet_path")
        if not location_path:
            raise ValueError(f"Scene {scene_number} is missing references.location_msr_path")
        location_path = self._resolve_project_path(location_path)

        patcher = WorkflowPatcher(self.load_workflow())
        self._patch_actor_reference_inputs(patcher, actor_paths)
        patcher.set_input_by_title(
            "#MSR_BACKGROUND",
            "image",
            self.asset_uploader.resolve_reference_image_name(location_path),
        )
        self._patch_prompt_inputs(patcher, scene, prompt=prompt, rolling=rolling)
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", f"ltx_msr_raw/scene_{scene_number:04}")
        patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "frame_count", self.msr_frame_count)
        patcher.try_set_existing_input_by_title("#MSR_FRAME_COUNT", "value", self.msr_frame_count)
        self._patch_seed_inputs(patcher, self._seed_for_scene(scene_number))
        patcher.try_set_existing_input_by_title("#WIDTH", "value", int(scene.get("width", 0) or 0))
        patcher.try_set_existing_input_by_title("#HEIGHT", "value", int(scene.get("height", 0) or 0))
        patcher.try_set_existing_input_by_title("#FRAMES", "value", render_frame_count)
        patcher.try_set_existing_input_by_title("#FRAMERATE", "value", int(scene.get("fps", 0) or 0))
        self._patch_startframe_input(patcher, scene)
        if comfy_audio_name:
            self._patch_audio_inputs(patcher, scene, comfy_audio_name=comfy_audio_name, rolling=rolling)

        return patcher.get()

    def _patch_startframe_input(self, patcher: WorkflowPatcher, scene: dict) -> None:
        keyframes = scene.get("keyframes") or {}
        startframe_path = keyframes.get("startframe_path") or keyframes.get("start_frame_path")
        if not startframe_path:
            return
        if not self._has_anchor(patcher, "#STARTFRAME"):
            raise ValueError("Movie MSR-I2V scene provides a startframe, but workflow is missing #STARTFRAME")
        image_name = self.asset_uploader.resolve_reference_image_name(self._resolve_project_path(startframe_path))
        patcher.set_input_by_title("#STARTFRAME", "image", image_name)

    def _write_debug_workflow(self, scene_number: int, workflow: dict) -> None:
        if self.debug_workflows_dir is None:
            return
        self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
    def _patch_audio_inputs(
        patcher: WorkflowPatcher,
        scene: dict,
        *,
        comfy_audio_name: str,
        rolling: AudioWindowSpec | None = None,
    ) -> None:
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
        start_index = float(scene.get("abs_start_seconds", 0.0) or 0.0)
        if rolling:
            start_index = float(rolling["audio_start_seconds"])
            duration = float(rolling["audio_duration_seconds"])
        patcher.try_set_existing_input_by_title("#TRIM_AUDIO", "start_index", start_index)
        patcher.try_set_existing_input_by_title("#TRIM_AUDIO", "duration", float(duration or 0.0))

    def _patch_prompt_inputs(
        self,
        patcher: WorkflowPatcher,
        scene: dict,
        *,
        prompt: str,
        rolling: AudioWindowSpec | None = None,
    ) -> None:
        if self._has_anchor(patcher, "#PROMPT_RELAY"):
            global_prompt, local_prompts, segment_lengths = self._build_prompt_relay_payload(
                scene,
                prompt=prompt,
                rolling=rolling,
            )
            patcher.set_input_by_title("#PROMPT_RELAY", "global_prompt", global_prompt)
            patcher.set_input_by_title("#PROMPT_RELAY", "local_prompts", local_prompts)
            patcher.set_input_by_title("#PROMPT_RELAY", "segment_lengths", segment_lengths)
            return
        patcher.set_input_by_title("#PROMPT", "text", str(prompt).strip())

    @staticmethod
    def _build_prompt_relay_payload(
        scene: dict,
        *,
        prompt: str,
        rolling: AudioWindowSpec | None = None,
    ) -> tuple[str, str, str]:
        frame_count = int(rolling["render_frame_count"]) if rolling else int(scene.get("frame_count", 1) or 1)
        trim_front_frames = int(rolling["trim_front_frames"]) if rolling else 0
        tail_loss_frames = int(rolling["tail_loss_frames"]) if rolling else 0
        ltx = scene.get("ltx") or {}
        if ltx.get("msr_prompt_relay") or ltx.get("msr_global_prompt"):
            payload = _build_msr_prompt_relay_payload(
                scene=scene,
                render_frame_count=frame_count,
                trim_front_frames=trim_front_frames,
                tail_loss_frames=tail_loss_frames,
            )
            return payload.global_prompt, payload.local_prompts, payload.segment_lengths

        reference_global_prompt = _build_msr_reference_global_prompt(scene.get("references") or {})
        if ltx.get("prompt_relay"):
            payload = PromptRelayPayloadBuilder().build(
                scene=scene,
                render_frame_count=frame_count,
                trim_front_frames=trim_front_frames,
                tail_loss_frames=tail_loss_frames,
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

    def _rolling_spec(self, scene: dict) -> AudioWindowSpec:
        return build_audio_window_spec(
            scene_number=int(scene["scene"]),
            fps=int(scene.get("fps", 0) or 24),
            scene_frame_count=int(scene.get("frame_count", 0) or 1),
            scene_start_seconds=float(scene.get("abs_start_seconds", 0.0) or 0.0),
            preroll_frames=self.preroll_frames,
            tail_loss_frames=self.tail_loss_frames,
            round_render_frames_to_8n1=self.round_render_frames_to_8n1,
        )

    def _seed_for_scene(self, scene_number: int) -> int:
        if self.randomize_seed:
            return random.randint(0, 2**63 - 1)
        return self.seed_offset + int(scene_number)

    def _resolve_project_path(self, path: str | Path) -> Path:
        if self.project_dir is None:
            return coerce_local_path(path)
        return coerce_local_path(path, base_dir=self.project_dir)

    @staticmethod
    def _patch_seed_inputs(patcher: WorkflowPatcher, seed: int) -> None:
        for _, node in patcher.find_nodes_by_meta_title("#SEED"):
            inputs = node.setdefault("inputs", {})
            for input_name in ("noise_seed", "seed", "value"):
                if input_name in inputs:
                    inputs[input_name] = seed

        for _, node in patcher.get().items():
            inputs = node.setdefault("inputs", {})
            for input_name in ("noise_seed", "seed"):
                if input_name in inputs:
                    inputs[input_name] = seed

    @staticmethod
    def _has_anchor(patcher: WorkflowPatcher, title: str) -> bool:
        try:
            patcher.find_node_by_meta_title(title)
            return True
        except KeyError:
            return False


def _build_msr_prompt_relay_payload(
    *,
    scene: dict,
    render_frame_count: int,
    trim_front_frames: int,
    tail_loss_frames: int,
) -> PromptRelayPayload:
    ltx = scene.get("ltx") or {}
    global_prompt = str(ltx.get("msr_global_prompt") or ltx.get("base_prompt") or "").strip()
    relays = list(ltx.get("msr_prompt_relay") or ltx.get("prompt_relay") or [])
    timeline_frames = max(1, int(render_frame_count) - 1)
    scene_timeline_frames = max(1, int(scene.get("frame_count", 1)) - 1)
    if str(ltx.get("msr_prompt_relay_mode") or "").strip().lower() == "single":
        prompt = str(relays[0].get("prompt") or "").strip() if relays and isinstance(relays[0], dict) else ""
        prompt = prompt or _msr_gap_prompt(scene)
        return PromptRelayPayload(
            global_prompt=global_prompt,
            local_prompts=prompt,
            segment_lengths=str(timeline_frames),
        )

    relay_segments: list[dict] = []
    if trim_front_frames > 0:
        relay_segments.append({
            "prompt": _msr_preroll_prompt(scene),
            "length": int(trim_front_frames),
        })

    if not relays:
        relay_segments.append({
            "prompt": _msr_gap_prompt(scene),
            "length": scene_timeline_frames,
        })
    else:
        cursor = 0
        for relay in sorted(relays, key=lambda item: int(item["frame_start"])):
            start = max(0, min(int(relay["frame_start"]), scene_timeline_frames))
            end = max(start, min(int(relay["frame_end"]), scene_timeline_frames))
            if start > cursor:
                relay_segments.append({
                    "prompt": _msr_gap_prompt(scene),
                    "length": start - cursor,
                })
                cursor = start

            relay_segments.append({
                "prompt": str(relay.get("prompt") or "").strip() or _msr_gap_prompt(scene),
                "length": max(1, end - start),
            })
            cursor = end

        if cursor < scene_timeline_frames:
            relay_segments.append({
                "prompt": _msr_gap_prompt(scene),
                "length": scene_timeline_frames - cursor,
            })

    if tail_loss_frames > 0:
        relay_segments.append({
            "prompt": _msr_tail_prompt(scene),
            "length": int(tail_loss_frames),
        })

    relay_segments = PromptRelayPayloadBuilder.normalize_segments(relay_segments)
    local_prompts = [segment["prompt"] for segment in relay_segments]
    segment_lengths = [int(segment["length"]) for segment in relay_segments]
    total = sum(segment_lengths)
    if total != timeline_frames:
        raise ValueError(
            f"PromptRelay segment length mismatch for MSR scene {scene.get('scene')}: "
            f"sum={total}, expected={timeline_frames}, render_frame_count={render_frame_count}, "
            f"scene_frame_count={scene.get('frame_count')}, preroll={trim_front_frames}, tail={tail_loss_frames}"
        )

    return PromptRelayPayload(
        global_prompt=global_prompt,
        local_prompts="\n|".join(local_prompts),
        segment_lengths=",".join(str(length) for length in segment_lengths),
    )


def _msr_preroll_prompt(scene: dict) -> str:
    ltx = scene.get("ltx") or {}
    prompt = str(ltx.get("msr_preroll_prompt") or "").strip()
    if prompt:
        return prompt
    return _msr_scene_direction(
        scene,
        fallback=(
            "Cinematic atmosphere gathers around {location}; {actor} remains physically present as particles, "
            "light, and environmental motion build tension before the main action begins."
        ),
    )


def _msr_tail_prompt(scene: dict) -> str:
    ltx = scene.get("ltx") or {}
    prompt = str(ltx.get("msr_tail_prompt") or "").strip()
    if prompt:
        return prompt
    return _msr_scene_direction(
        scene,
        fallback=(
            "{actor} carries the last motion through {location}; camera and atmosphere continue the same dramatic "
            "energy as the action resolves without introducing a new scene."
        ),
    )


def _msr_gap_prompt(scene: dict) -> str:
    ltx = scene.get("ltx") or {}
    prompt = str(ltx.get("msr_gap_prompt") or "").strip()
    if prompt:
        return prompt
    return _msr_scene_direction(
        scene,
        fallback=(
            "{actor} continues the scene action inside {location}; the camera stays active while environmental "
            "details keep moving around the reference subject."
        ),
    )


def _msr_scene_direction(scene: dict, *, fallback: str) -> str:
    metadata = scene.get("metadata") or {}
    references = scene.get("references") or {}
    actor = _first_reference_name(references, default="the reference actor")
    location = _location_reference_name(references)
    base_concept = str(metadata.get("base_concept") or "").strip()
    camera = str(metadata.get("camera_motion") or "").strip()
    character_motion = str(metadata.get("character_motion") or "").strip()
    prompt = fallback.format(actor=actor, location=location)
    extras = [value for value in (character_motion, camera, base_concept) if value]
    if extras:
        prompt = f"{prompt} {' '.join(text.strip(' .') + '.' for text in extras)}"
    return re.sub(r"\s+", " ", prompt).strip()


def _first_reference_name(references: dict, *, default: str) -> str:
    actors = references.get("actor_reference_descriptions") or []
    if actors:
        name = str(actors[0].get("name") or actors[0].get("id") or "").strip()
        if name:
            return name
    return default


def _location_reference_name(references: dict) -> str:
    location = references.get("location_reference_description") or {}
    name = str(location.get("name") or location.get("id") or "").strip()
    return name or "the referenced location"


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
    name = str(item.get("name") or item.get("id") or "").strip(" .")
    role = str(item.get("role") or "").strip(" .")
    visual = str(item.get("visual_description") or "").strip(" .")
    image_prompt = str(item.get("image_prompt") or "").strip(" .")
    chunks = [chunk for chunk in (name, role, visual or image_prompt) if chunk]
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
    for label in ("Camera motion", "Character motion", "Subject or environment motion", "Story beat"):
        match = re.search(rf"(?is){re.escape(label)}\s*:\s*(.*?)(?:\.\.|\. (?=[A-Z][A-Za-z ]+:)|$)", source)
        if match:
            text = re.sub(r"\s+", " ", match.group(1)).strip(" .")
            if text:
                parts.append(f"{label}: {text}.")
    return " ".join(parts).strip()
