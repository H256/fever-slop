from __future__ import annotations

from pathlib import Path
import json
import random
import shutil

from comfyui_client import ComfyUIClient
from workflow_patcher import WorkflowPatcher
from video_postprocessor import VideoPostProcessor, TrimSpec


class LTXVideoRenderer:
    def __init__(
        self,
        client: ComfyUIClient,
        ltx_workflow_path: str | Path,
        output_dir: str | Path,
        width_node_title: str = "#WIDTH",
        height_node_title: str = "#HEIGHT",
        load_audio_node_title: str = "#LOAD_AUDIO",
        trim_audio_node_title: str = "#TRIM_AUDIO",
        startframe_node_title: str = "#STARTFRAME",
        frames_node_title: str = "#FRAMES",
        framerate_node_title: str = "#FRAMERATE",
        seed_node_title: str = "#SEED",
        prompt_relay_node_title: str = "#PROMPT_RELAY",
        save_video_node_title: str = "#SAVE_VIDEO",
        character_lora_node_title: str | None = "#CHARACTER_LORA",
        character_lora_strength: float = 1.0,
        randomize_seed: bool = False,
        seed_offset: int = 100000,
        segment_length_mode: str = "frames_minus_one",
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        allow_out_of_range_clips: bool = False,
        debug_workflows_dir: str | Path | None = None,
        preroll_frames: int = 0,
        tail_loss_frames: int = 0,
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
    ):
        self.client = client
        self.ltx_workflow_path = Path(ltx_workflow_path)
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
        self.save_video_node_title = save_video_node_title
        self.character_lora_node_title = character_lora_node_title

        self.character_lora_strength = character_lora_strength
        self.randomize_seed = randomize_seed
        self.seed_offset = seed_offset

        if segment_length_mode not in {"frames_minus_one", "frames"}:
            raise ValueError("segment_length_mode must be 'frames_minus_one' or 'frames'")
        self.segment_length_mode = segment_length_mode

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.allow_out_of_range_clips = allow_out_of_range_clips
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None

        self.preroll_frames = max(0, int(preroll_frames))
        self.tail_loss_frames = max(0, int(tail_loss_frames))
        self.postprocess = postprocess
        self.postprocessor = VideoPostProcessor(
            ffmpeg_path=ffmpeg_path,
            reencode=postprocess_reencode,
        )

    def load_workflow(self) -> dict:
        return json.loads(self.ltx_workflow_path.read_text(encoding="utf-8"))

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

    def render_scene_video(self, scene: dict, comfy_audio_name: str, comfy_startframe_name: str, rolling: dict) -> Path:
        workflow = self.load_workflow()
        patcher = WorkflowPatcher(workflow)

        scene_number = int(scene["scene"])
        fps = int(scene["fps"])
        width = int(scene["width"])
        height = int(scene["height"])
        render_frame_count = int(rolling["render_frame_count"])

        global_prompt, local_prompts, segment_lengths = self._build_prompt_relay_payload(
            scene=scene,
            render_frame_count=render_frame_count,
            trim_front_frames=int(rolling["trim_front_frames"]),
            tail_loss_frames=int(rolling["tail_loss_frames"]),
        )

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

        patcher.set_input_by_title(self.prompt_relay_node_title, "global_prompt", global_prompt)
        patcher.set_input_by_title(self.prompt_relay_node_title, "local_prompts", local_prompts)
        patcher.set_input_by_title(self.prompt_relay_node_title, "segment_lengths", segment_lengths)

        patcher.set_input_by_title(self.save_video_node_title, "filename_prefix", f"ltx_raw/scene_{scene_number:04}")

        if self.character_lora_node_title:
            try:
                patcher.patch_lora_strength_by_title(
                    self.character_lora_node_title,
                    self.character_lora_strength,
                )
            except KeyError:
                pass

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

    def _rolling_spec(self, scene: dict) -> dict:
        scene_number = int(scene["scene"])
        fps = int(scene["fps"])
        scene_frame_count = int(scene["frame_count"])
        scene_start = float(scene["abs_start_seconds"])

        preroll = 0 if scene_number == 1 else self.preroll_frames
        tail = self.tail_loss_frames

        audio_start = max(0.0, scene_start - preroll / float(fps))
        effective_preroll = round((scene_start - audio_start) * fps)

        render_frame_count = scene_frame_count + effective_preroll + tail
        audio_duration = max(0.0, (render_frame_count - 1) / float(fps))

        return {
            "render_frame_count": render_frame_count,
            "trim_front_frames": effective_preroll,
            "tail_loss_frames": tail,
            "audio_start_seconds": audio_start,
            "audio_duration_seconds": audio_duration,
        }

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
        ltx = scene["ltx"]
        global_prompt = ltx["base_prompt"].strip()
        relays = ltx.get("prompt_relay", [])

        timeline_frames = render_frame_count if self.segment_length_mode == "frames" else max(1, render_frame_count - 1)
        scene_timeline_frames = int(scene["frame_count"]) if self.segment_length_mode == "frames" else max(1, int(scene["frame_count"]) - 1)

        local_prompts: list[str] = []
        segment_lengths: list[int] = []

        if trim_front_frames > 0:
            local_prompts.append("pre-roll continuity hold, preserve startframe composition, subtle breathing and atmospheric motion")
            segment_lengths.append(trim_front_frames)

        if not relays:
            local_prompts.append("continue the main scene motion with stable subject identity")
            segment_lengths.append(scene_timeline_frames)
        else:
            relays = sorted(relays, key=lambda item: int(item["frame_start"]))
            cursor = 0

            for relay in relays:
                start = max(0, min(int(relay["frame_start"]), scene_timeline_frames))
                end = max(start, min(int(relay["frame_end"]), scene_timeline_frames))

                if start > cursor:
                    local_prompts.append("hold the same shot, subtle breathing and atmospheric motion")
                    segment_lengths.append(start - cursor)
                    cursor = start

                length = max(1, end - start)
                local_prompts.append(str(relay["prompt"]).strip())
                segment_lengths.append(length)
                cursor = end

            if cursor < scene_timeline_frames:
                local_prompts.append("hold the same shot, subtle breathing and atmospheric motion")
                segment_lengths.append(scene_timeline_frames - cursor)

        if tail_loss_frames > 0:
            local_prompts.append("tail safety continuation, maintain same motion and atmosphere without introducing a new scene")
            segment_lengths.append(tail_loss_frames)

        total = sum(segment_lengths)
        if total != timeline_frames:
            raise ValueError(
                f"PromptRelay segment length mismatch for scene {scene.get('scene')}: "
                f"sum={total}, expected={timeline_frames}, mode={self.segment_length_mode}, "
                f"render_frame_count={render_frame_count}, scene_frame_count={scene.get('frame_count')}, "
                f"preroll={trim_front_frames}, tail={tail_loss_frames}"
            )

        return global_prompt, "\n|".join(local_prompts), ",".join(str(int(x)) for x in segment_lengths)
