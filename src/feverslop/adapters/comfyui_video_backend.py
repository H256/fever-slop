from __future__ import annotations

from pathlib import Path
import json
import shutil

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher, LTXWorkflowSettings
from feverslop.ports.rendering import VideoRenderRequest
from feverslop.domain.ltx_rendering import (
    AudioWindowSpec,
    build_audio_window_spec,
    round_up_8n1,
)
from feverslop.adapters.video_postprocessor import VideoPostProcessor, TrimSpec


class ComfyUIVideoRenderBackend:
    min_prompt_relay_frames = 6

    def __init__(
        self,
        client: ComfyUIClient,
        ltx_workflow_path: str | Path,
        output_dir: str | Path,
        single_prompt_workflow_path: str | Path | None = None,
        render_mode: str = "relay",
        width_node_title: str = "#WIDTH",
        height_node_title: str = "#HEIGHT",
        load_audio_node_title: str = "#LOAD_AUDIO",
        trim_audio_node_title: str = "#TRIM_AUDIO",
        startframe_node_title: str = "#STARTFRAME",
        frames_node_title: str = "#FRAMES",
        framerate_node_title: str = "#FRAMERATE",
        seed_node_title: str = "#SEED",
        prompt_relay_node_title: str = "#PROMPT_RELAY",
        single_prompt_node_title: str = "#PROMPT",
        single_prompt_input_name: str = "text",
        save_video_node_title: str = "#SAVE_VIDEO",
        character_lora_node_title: str | None = "#CHARACTER_LORA",
        character_lora_strength: float = 1.0,
        lora_1_enabled: bool = False,
        lora_1_name: str = "",
        lora_1_strength_model: float = 1.0,
        lora_1_strength_clip: float = 1.0,
        lora_1_strengths_explicit: bool = False,
        lora_1_node_title: str = "#LORA_1",
        randomize_seed: bool = False,
        seed_offset: int = 100000,
        segment_length_mode: str = "frames_minus_one",
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        allow_out_of_range_clips: bool = False,
        debug_workflows_dir: str | Path | None = None,
        preroll_frames: int = 0,
        tail_loss_frames: int = 0,
        round_render_frames_to_8n1: bool = False,
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        workflow_patcher: LTXWorkflowPatcher | None = None,
        model_resolver=None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
    ):
        self.client = client
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.ltx_workflow_path = Path(ltx_workflow_path)
        self.single_prompt_workflow_path = Path(single_prompt_workflow_path) if single_prompt_workflow_path else None
        self.output_dir = Path(output_dir)
        self.raw_output_dir = self.output_dir / "raw"
        self.final_output_dir = self.output_dir / "final"

        self.width_node_title = width_node_title
        self.height_node_title = height_node_title
        self.load_audio_node_title = load_audio_node_title
        self.trim_audio_node_title = trim_audio_node_title
        self.startframe_node_title = startframe_node_title
        self.frames_node_title = frames_node_title
        self.framerate_node_title = framerate_node_title
        self.seed_node_title = seed_node_title
        self.prompt_relay_node_title = prompt_relay_node_title
        self.single_prompt_node_title = single_prompt_node_title
        self.single_prompt_input_name = single_prompt_input_name
        self.save_video_node_title = save_video_node_title
        self.character_lora_node_title = character_lora_node_title

        self.character_lora_strength = character_lora_strength
        self.lora_1_enabled = bool(lora_1_enabled)
        self.lora_1_name = lora_1_name
        self.lora_1_strength_model = float(lora_1_strength_model)
        self.lora_1_strength_clip = float(lora_1_strength_clip)
        self.lora_1_strengths_explicit = bool(lora_1_strengths_explicit)
        self.lora_1_node_title = lora_1_node_title
        self.randomize_seed = randomize_seed
        self.seed_offset = seed_offset

        if segment_length_mode not in {"frames_minus_one", "frames"}:
            raise ValueError("segment_length_mode must be 'frames_minus_one' or 'frames'")
        self.segment_length_mode = segment_length_mode
        if render_mode not in {"relay", "single_prompt", "auto"}:
            raise ValueError("render_mode must be 'relay', 'single_prompt', or 'auto'")
        self.render_mode = render_mode

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.allow_out_of_range_clips = allow_out_of_range_clips
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None

        self.preroll_frames = max(0, int(preroll_frames))
        self.tail_loss_frames = max(0, int(tail_loss_frames))
        self.round_render_frames_to_8n1 = bool(round_render_frames_to_8n1)
        self.postprocess = postprocess
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()
        self.postprocessor = postprocessor or VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
        )
        self.workflow_patcher = workflow_patcher or LTXWorkflowPatcher(
            LTXWorkflowSettings(
                ltx_workflow_path=self.ltx_workflow_path,
                single_prompt_workflow_path=self.single_prompt_workflow_path,
                render_mode=self.render_mode,
                width_node_title=self.width_node_title,
                height_node_title=self.height_node_title,
                load_audio_node_title=self.load_audio_node_title,
                trim_audio_node_title=self.trim_audio_node_title,
                startframe_node_title=self.startframe_node_title,
                frames_node_title=self.frames_node_title,
                framerate_node_title=self.framerate_node_title,
                seed_node_title=self.seed_node_title,
                prompt_relay_node_title=self.prompt_relay_node_title,
                single_prompt_node_title=self.single_prompt_node_title,
                single_prompt_input_name=self.single_prompt_input_name,
                save_video_node_title=self.save_video_node_title,
                character_lora_node_title=self.character_lora_node_title,
                character_lora_strength=self.character_lora_strength,
                lora_1_enabled=self.lora_1_enabled,
                lora_1_name=self.lora_1_name,
                lora_1_strength_model=self.lora_1_strength_model,
                lora_1_strength_clip=self.lora_1_strength_clip,
                lora_1_strengths_explicit=self.lora_1_strengths_explicit,
                lora_1_node_title=self.lora_1_node_title,
                randomize_seed=self.randomize_seed,
                seed_offset=self.seed_offset,
                segment_length_mode=self.segment_length_mode,
                debug_workflows_dir=self.debug_workflows_dir,
            )
        )

    def load_workflow(self, mode: str = "relay") -> dict:
        return self.workflow_patcher.load_workflow(mode=mode)

    def validate_workflow(self, mode: str = "relay") -> None:
        self.workflow_patcher.validate_workflow(mode=mode)

    def render_video(self, request: VideoRenderRequest) -> Path:
        one_scene_plan = request.output_dir / "_single_scene_plan.json"
        one_scene_plan.parent.mkdir(parents=True, exist_ok=True)
        one_scene_plan.write_text(
            json.dumps([request.scene], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rendered = self.render_videos(
            render_plan_path=one_scene_plan,
            audio_file=request.audio_file,
            storyboard_dir=request.storyboard_dir,
            skip_existing=request.skip_existing,
            uploaded_audio_name=request.uploaded_audio_name,
            upload_audio=request.upload_audio,
            upload_startframes=request.upload_startframes,
        )
        if not rendered:
            raise RuntimeError(f"No rendered video returned for scene {request.scene_number}")
        return rendered[0]

    def render_videos(
        self,
        render_plan_path: str | Path,
        audio_file: str | Path,
        storyboard_dir: str | Path,
        limit: int | None = None,
        scene_numbers: set[int] | None = None,
        skip_existing: bool = True,
        uploaded_audio_name: str | None = None,
        upload_audio: bool = True,
        upload_startframes: bool = True,
    ) -> list[Path]:
        render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
        audio_file = Path(audio_file)
        storyboard_dir = Path(storyboard_dir)

        if scene_numbers is not None:
            render_plan = [s for s in render_plan if int(s["scene"]) in scene_numbers]
        if limit is not None:
            render_plan = render_plan[:limit]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)

        comfy_audio_name = self.asset_uploader.resolve_audio_name(
            audio_file,
            upload_audio=upload_audio,
            uploaded_audio_name=uploaded_audio_name,
        )

        rendered_files: list[Path] = []
        manifest_entries: list[dict] = []

        for scene in render_plan:
            scene_number = int(scene["scene"])
            final_output_path = self.final_output_dir / f"scene_{scene_number:04}.mp4"

            if skip_existing and final_output_path.exists():
                rendered_files.append(final_output_path)
                continue

            duration = float(scene["duration_seconds"])
            if not self.allow_out_of_range_clips and (duration < self.min_duration or duration > self.max_duration):
                raise ValueError(
                    f"Scene {scene_number} duration {duration:.3f}s is outside "
                    f"{self.min_duration:.3f}s..{self.max_duration:.3f}s."
                )

            startframe_path = storyboard_dir / f"scene_{scene_number:04}.png"
            if not startframe_path.exists():
                raise FileNotFoundError(f"Missing storyboard startframe: {startframe_path}")

            comfy_startframe_name = self.asset_uploader.resolve_startframe_name(
                startframe_path,
                upload_startframes=upload_startframes,
            )

            rolling = self._rolling_spec(scene)
            raw_clip = self.render_scene_video(
                scene=scene,
                comfy_audio_name=comfy_audio_name,
                comfy_startframe_name=comfy_startframe_name,
                rolling=rolling,
            )

            if self.postprocess:
                spec = TrimSpec(
                    source_file=raw_clip,
                    output_file=final_output_path,
                    fps=int(scene["fps"]),
                    trim_front_frames=int(rolling["trim_front_frames"]),
                    keep_frames=int(scene["frame_count"]),
                    scene=scene_number,
                )
                clip_path = self.postprocessor.trim_clip(spec)
            else:
                clip_path = self.output_dir / f"scene_{scene_number:04}.mp4"
                shutil.copy2(raw_clip, clip_path)

            rendered_files.append(clip_path)
            manifest_entries.append({
                "scene": scene_number,
                "raw_clip": str(raw_clip),
                "final_clip": str(clip_path),
                "scene_frame_count": int(scene["frame_count"]),
                "render_frame_count": int(rolling["render_frame_count"]),
                "trim_front_frames": int(rolling["trim_front_frames"]),
                "tail_loss_frames": int(rolling["tail_loss_frames"]),
                "audio_start_seconds": float(rolling["audio_start_seconds"]),
                "audio_duration_seconds": float(rolling["audio_duration_seconds"]),
            })

        self.postprocessor.write_concat_list(rendered_files, self.output_dir / "concat_list.txt")
        self.postprocessor.write_manifest(manifest_entries, self.output_dir / "render_manifest.json")
        return rendered_files

    def render_scene_video(self, scene: dict, comfy_audio_name: str, comfy_startframe_name: str, rolling: AudioWindowSpec) -> Path:
        scene_number = int(scene["scene"])
        workflow = self.workflow_patcher.build_workflow(
            scene=scene,
            comfy_audio_name=comfy_audio_name,
            comfy_startframe_name=comfy_startframe_name,
            rolling=rolling,
        )
        mode = self.workflow_patcher.render_mode_for_scene(scene)
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_patcher.workflow_path_for_mode(mode),
        )
        return self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=self.raw_output_dir / f"scene_{scene_number:04}_raw.mp4",
        )

    def _patch_lora_inputs(self, patcher) -> None:
        self.workflow_patcher.patch_lora_inputs(patcher)

    def _render_mode_for_scene(self, scene: dict) -> str:
        return self.workflow_patcher.render_mode_for_scene(scene)

    def _workflow_path_for_mode(self, mode: str) -> Path:
        return self.workflow_patcher.workflow_path_for_mode(mode)

    def _patch_prompt_inputs(
        self,
        patcher,
        scene: dict,
        mode: str,
        render_frame_count: int,
        trim_front_frames: int,
        tail_loss_frames: int,
    ) -> None:
        self.workflow_patcher.patch_prompt_inputs(
            patcher=patcher,
            scene=scene,
            mode=mode,
            render_frame_count=render_frame_count,
            trim_front_frames=trim_front_frames,
            tail_loss_frames=tail_loss_frames,
        )

    def _rolling_spec(self, scene: dict) -> AudioWindowSpec:
        return build_audio_window_spec(
            scene_number=int(scene["scene"]),
            fps=int(scene["fps"]),
            scene_frame_count=int(scene["frame_count"]),
            scene_start_seconds=float(scene["abs_start_seconds"]),
            preroll_frames=self.preroll_frames,
            tail_loss_frames=self.tail_loss_frames,
            round_render_frames_to_8n1=self.round_render_frames_to_8n1,
        )

    @staticmethod
    def _round_up_8n1(frame_count: int) -> int:
        return round_up_8n1(frame_count)

    def _seed_for_scene(self, scene_number: int) -> int:
        return self.workflow_patcher.seed_for_scene(scene_number)

    @staticmethod
    def _comfy_path_from_upload(upload_response: dict) -> str:
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(upload_response)

    @staticmethod
    def _extract_output_videos(history_entry: dict) -> list[dict]:
        return ComfyUIRenderQueue.extract_output_videos(history_entry)

    def _build_prompt_relay_payload(
        self,
        scene: dict,
        render_frame_count: int,
        trim_front_frames: int,
        tail_loss_frames: int,
    ) -> tuple[str, str, str]:
        return self.workflow_patcher.build_prompt_relay_payload(
            scene=scene,
            render_frame_count=render_frame_count,
            trim_front_frames=trim_front_frames,
            tail_loss_frames=tail_loss_frames,
        )

    @classmethod
    def _normalize_prompt_relay_segments(cls, segments: list[dict]) -> list[dict]:
        return LTXWorkflowPatcher.normalize_prompt_relay_segments(segments)


class ComfyUIVideoBackend(ComfyUIVideoRenderBackend):
    """Compatibility alias for older imports."""
