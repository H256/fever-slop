from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import random

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import TrimSpec, VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.ltx_rendering import AudioWindowSpec, build_audio_window_spec
from feverslop.errors import FeverSlopValidationError
from feverslop.path_utils import coerce_local_path
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIIngredientsVideoRenderBackend:
    def __init__(
        self,
        *,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_offset: int = 100000,
        randomize_seed: bool = False,
        debug_workflows_dir: str | Path | None = None,
        preroll_frames: int = 0,
        tail_loss_frames: int = 0,
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
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.workflow = deepcopy(workflow) if workflow is not None else None
        self.workflow_label = Path(workflow_label) if workflow_label is not None else self.workflow_path
        self.output_dir = Path(output_dir)
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.raw_output_dir = self.output_dir / "raw"
        self.seed_offset = int(seed_offset)
        self.randomize_seed = bool(randomize_seed)
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
        workflow = self.build_workflow(
            request.scene,
            prompt=request.prompt,
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
        rolling: AudioWindowSpec | None = None,
    ) -> dict:
        scene_number = int(scene["scene"])
        render_frame_count = int(rolling["render_frame_count"]) if rolling else int(scene.get("frame_count", 0) or 0)

        patcher = WorkflowPatcher(self.load_workflow())
        self._patch_ingredients_input(patcher, scene)
        self._patch_prompt_inputs(patcher, scene, prompt=prompt)
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", f"ltx_ingredients_raw/scene_{scene_number:04}")
        self._patch_seed_inputs(patcher, self._seed_for_scene(scene_number))
        patcher.try_set_existing_input_by_title("#WIDTH", "value", int(scene.get("width", 0) or 0))
        patcher.try_set_existing_input_by_title("#HEIGHT", "value", int(scene.get("height", 0) or 0))
        patcher.try_set_existing_input_by_title("#FRAMES", "value", render_frame_count)
        patcher.try_set_existing_input_by_title("#FRAMERATE", "value", int(scene.get("fps", 0) or 0))
        _patch_i2v_latent_lengths(patcher, render_frame_count)

        return patcher.get()

    def _patch_ingredients_input(self, patcher: WorkflowPatcher, scene: dict) -> None:
        if not self._has_anchor(patcher, "#INGREDIENTS"):
            raise FeverSlopValidationError("Ingredients workflow is missing #INGREDIENTS anchor")

        sheet_path = scene.get("ingredients_scene_sheet") or ""
        if not sheet_path:
            raise FeverSlopValidationError(f"Scene {scene.get('scene')} is missing ingredients_scene_sheet path")

        image_name = self.asset_uploader.resolve_reference_image_name(
            self._resolve_project_path(sheet_path)
        )
        patcher.set_input_by_title("#INGREDIENTS", "image", image_name)

    def _patch_prompt_inputs(
        self,
        patcher: WorkflowPatcher,
        scene: dict,
        *,
        prompt: str,
    ) -> None:
        ltx = scene.get("ltx") or {}
        scene_desc = str(ltx.get("ingredients_scene_sheet_description") or "").strip()
        target_prompt = str(ltx.get("ingredients_target_prompt") or "").strip()

        if scene_desc and target_prompt:
            assembled = scene_desc + "\n" + target_prompt
        elif scene_desc:
            assembled = scene_desc
        elif target_prompt:
            assembled = target_prompt
        else:
            assembled = str(prompt).strip()

        patcher.set_input_by_title("#PROMPT_POSITIVE", "text", assembled)

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

    def _write_debug_workflow(self, scene_number: int, workflow: dict) -> None:
        if self.debug_workflows_dir is None:
            return
        self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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


def _patch_i2v_latent_lengths(patcher: WorkflowPatcher, render_frame_count: int) -> None:
    workflow = patcher.get()
    latent_node_ids: set[str] = set()
    for node in workflow.values():
        if node.get("class_type") != "LTXVImgToVideoInplace":
            continue
        latent_input = (node.get("inputs") or {}).get("latent")
        if isinstance(latent_input, list) and latent_input:
            latent_node_ids.add(str(latent_input[0]))

    for node_id in latent_node_ids:
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "EmptyLTXVLatentVideo":
            continue
        node.setdefault("inputs", {})["length"] = int(render_frame_count)
