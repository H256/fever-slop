from __future__ import annotations

from pathlib import Path
import json
import random
import shutil

from autoprompter.adapters.comfyui_client import ComfyUIClient
from autoprompter.ports.rendering import VideoRenderRequest
from autoprompter.domain.ltx_rendering import (
    AudioWindowSpec,
    PromptRelayPayloadBuilder,
    build_audio_window_spec,
    round_up_8n1,
)
from autoprompter.adapters.workflow_patcher import WorkflowPatcher
from autoprompter.adapters.video_postprocessor import VideoPostProcessor, TrimSpec


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
    ):
        self.client = client
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
        self.postprocessor = VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
        )

    def load_workflow(self, mode: str = "relay") -> dict:
        workflow_path = self._workflow_path_for_mode(mode)
        return json.loads(workflow_path.read_text(encoding="utf-8"))

    def validate_workflow(self, mode: str = "relay") -> None:
        workflow_path = self._workflow_path_for_mode(mode)
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        patcher = WorkflowPatcher(workflow)

        required_titles = [
            self.width_node_title,
            self.height_node_title,
            self.load_audio_node_title,
            self.trim_audio_node_title,
            self.startframe_node_title,
            self.frames_node_title,
            self.framerate_node_title,
            self.seed_node_title,
            self.save_video_node_title,
        ]
        if mode != "single_prompt":
            required_titles.append(self.prompt_relay_node_title)
        if self.lora_1_enabled:
            required_titles.append(self.lora_1_node_title)

        for title in dict.fromkeys(required_titles):
            try:
                patcher.find_node_by_meta_title(title)
            except KeyError as exc:
                raise ValueError(f"Missing workflow anchor {title} in workflow file {workflow_path}") from exc

        if mode == "single_prompt":
            prompt_title_candidates = [
                self.single_prompt_node_title,
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

        if upload_audio:
            audio_upload = self.client.upload_file_via_image_endpoint(
                audio_file,
                subfolder="autoprompter/audio",
                file_type="input",
                overwrite=True,
            )
            comfy_audio_name = self._comfy_path_from_upload(audio_upload)
        else:
            comfy_audio_name = uploaded_audio_name or audio_file.name

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

            if upload_startframes:
                image_upload = self.client.upload_image(
                    startframe_path,
                    subfolder="autoprompter/storyboard",
                    file_type="input",
                    overwrite=True,
                )
                comfy_startframe_name = self._comfy_path_from_upload(image_upload)
            else:
                comfy_startframe_name = startframe_path.name

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
        mode = self._render_mode_for_scene(scene)
        self.validate_workflow(mode=mode)
        workflow = self.load_workflow(mode=mode)
        patcher = WorkflowPatcher(workflow)

        scene_number = int(scene["scene"])
        fps = int(scene["fps"])
        width = int(scene["width"])
        height = int(scene["height"])
        render_frame_count = int(rolling["render_frame_count"])

        patcher.set_input_by_title(self.width_node_title, "value", width)
        patcher.set_input_by_title(self.height_node_title, "value", height)
        patcher.set_input_by_title(self.frames_node_title, "value", render_frame_count)
        patcher.set_input_by_title(self.framerate_node_title, "value", fps)
        patcher.set_input_by_title(self.seed_node_title, "noise_seed", self._seed_for_scene(scene_number))

        patcher.set_input_by_title(self.load_audio_node_title, "audio", comfy_audio_name)
        patcher.try_set_existing_input_by_title(
            self.load_audio_node_title,
            "audioUI",
            f"/api/view?filename={comfy_audio_name}&type=input",
        )

        patcher.set_input_by_title(self.trim_audio_node_title, "start_index", float(rolling["audio_start_seconds"]))
        patcher.set_input_by_title(self.trim_audio_node_title, "duration", float(rolling["audio_duration_seconds"]))
        patcher.set_input_by_title(self.startframe_node_title, "image", comfy_startframe_name)

        self._patch_prompt_inputs(
            patcher=patcher,
            scene=scene,
            mode=mode,
            render_frame_count=render_frame_count,
            trim_front_frames=int(rolling["trim_front_frames"]),
            tail_loss_frames=int(rolling["tail_loss_frames"]),
        )

        patcher.set_input_by_title(self.save_video_node_title, "filename_prefix", f"ltx_raw/scene_{scene_number:04}")

        self._patch_lora_inputs(patcher)

        if self.debug_workflows_dir:
            self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json").write_text(
                json.dumps(patcher.get(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        prompt_id = self.client.queue_prompt(patcher.get())
        history = self.client.wait_for_completion(prompt_id)

        videos = self._extract_output_videos(history)
        if not videos:
            raise RuntimeError(f"No video output for scene {scene_number}")

        first = videos[0]
        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=self.raw_output_dir / f"scene_{scene_number:04}_raw.mp4",
        )

    def _patch_lora_inputs(self, patcher: WorkflowPatcher) -> None:
        if self.lora_1_enabled:
            patcher.patch_lora_by_title(
                self.lora_1_node_title,
                lora_name=self.lora_1_name,
                strength_model=self.lora_1_strength_model,
                strength_clip=self.lora_1_strength_clip,
            )
        elif self.lora_1_strengths_explicit:
            patcher.patch_lora_strengths_by_title(
                self.lora_1_node_title,
                strength_model=self.lora_1_strength_model,
                strength_clip=self.lora_1_strength_clip,
            )
        elif self.character_lora_node_title:
            try:
                patcher.patch_lora_strength_by_title(
                    self.character_lora_node_title,
                    self.character_lora_strength,
                )
            except KeyError:
                pass

    def _render_mode_for_scene(self, scene: dict) -> str:
        if self.render_mode != "auto":
            return self.render_mode

        hint = str(scene.get("ltx", {}).get("render_mode_hint", "relay")).strip()
        if hint == "single_prompt":
            return "single_prompt"
        return "relay"

    def _workflow_path_for_mode(self, mode: str) -> Path:
        if mode == "single_prompt":
            return self.single_prompt_workflow_path or self.ltx_workflow_path
        return self.ltx_workflow_path

    def _patch_prompt_inputs(
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
                self.single_prompt_node_title,
                "#PROMPT_POSITIVE",
                "#PROMPT",
            ]
            last_error = None
            for title in dict.fromkeys(prompt_title_candidates):
                try:
                    patcher.set_input_by_title(
                        title,
                        self.single_prompt_input_name,
                        str(prompt).strip(),
                    )
                    return
                except KeyError as exc:
                    last_error = exc

            raise last_error or KeyError(
                f"No prompt node found for single_prompt mode. Tried: {prompt_title_candidates}"
            )
            return

        global_prompt, local_prompts, segment_lengths = self._build_prompt_relay_payload(
            scene=scene,
            render_frame_count=render_frame_count,
            trim_front_frames=trim_front_frames,
            tail_loss_frames=tail_loss_frames,
        )
        patcher.set_input_by_title(self.prompt_relay_node_title, "global_prompt", global_prompt)
        patcher.set_input_by_title(self.prompt_relay_node_title, "local_prompts", local_prompts)
        patcher.set_input_by_title(self.prompt_relay_node_title, "segment_lengths", segment_lengths)

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
        if self.randomize_seed:
            return random.randint(0, 2**63 - 1)
        return self.seed_offset + scene_number

    @staticmethod
    def _comfy_path_from_upload(upload_response: dict) -> str:
        name = upload_response.get("name")
        subfolder = upload_response.get("subfolder", "")
        if not name:
            raise ValueError(f"Unexpected ComfyUI upload response: {upload_response}")
        return f"{subfolder}/{name}" if subfolder else name

    @staticmethod
    def _extract_output_videos(history_entry: dict) -> list[dict]:
        videos = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_output in outputs.items():
            for key in ("videos", "gifs", "files"):
                for item in node_output.get(key, []):
                    filename = item.get("filename")
                    if filename and filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                        videos.append({
                            "node_id": node_id,
                            "filename": filename,
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        })
        return videos

    def _build_prompt_relay_payload(
        self,
        scene: dict,
        render_frame_count: int,
        trim_front_frames: int,
        tail_loss_frames: int,
    ) -> tuple[str, str, str]:
        payload = PromptRelayPayloadBuilder(
            segment_length_mode=self.segment_length_mode,
        ).build(
            scene=scene,
            render_frame_count=render_frame_count,
            trim_front_frames=trim_front_frames,
            tail_loss_frames=tail_loss_frames,
        )
        return payload.global_prompt, payload.local_prompts, payload.segment_lengths

    @classmethod
    def _normalize_prompt_relay_segments(cls, segments: list[dict]) -> list[dict]:
        return PromptRelayPayloadBuilder.normalize_segments(segments)


class ComfyUIVideoBackend(ComfyUIVideoRenderBackend):
    """Compatibility alias for older imports."""
