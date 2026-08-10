from __future__ import annotations

from pathlib import Path
import json
import random

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_minimax_h3_video_backend import ComfyUIMiniMaxH3VideoRenderBackend
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.postprocessing import TrimSpec
from feverslop.errors import FeverSlopValidationError
from feverslop.config.video_settings import VideoSettings
from feverslop.path_utils import coerce_local_path
from feverslop.ports.rendering import VideoRenderRequest


class ComfyUIMiniMaxH3R2VBackend(ComfyUIMiniMaxH3VideoRenderBackend):
    """MiniMax H3 reference-to-video backend using FeverSlop meta-anchors.

    Subclass of ComfyUIMiniMaxH3VideoRenderBackend that patches workflows by
    meta-title anchors (#PROMPT, #SEED, #FRAMECOUNT, #MEGAPIXELS, #REF_N,
    #LOAD_AUDIO, #TRIM_AUDIO, #SAVE_VIDEO) instead of direct class-type
    patching.
    """

    MAX_REF_IMAGES = 9
    MAX_REF_VIDEOS = 3
    MAX_REF_AUDIOS = 3
    MAX_STEM_AUDIOS = MAX_REF_AUDIOS
    FPS = 24
    pipeline_name = "minimax-h3-r2v"

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
        postprocess: bool = True,
        ffmpeg_path: str = "ffmpeg",
        postprocess_reencode: bool = True,
        ffmpeg_debug: bool = False,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        postprocessor: VideoPostProcessor | None = None,
        model_resolver=None,
        video_settings: VideoSettings | None = None,
        project_dir: str | Path | None = None,
        workflow: dict | None = None,
        workflow_label: str | Path | None = None,
        audio_ref_stems: list[str] | None = None,
    ):
        super().__init__(
            client=client,
            workflow_path=workflow_path,
            output_dir=output_dir,
            preroll_frames=preroll_frames,
            tail_loss_frames=tail_loss_frames,
            postprocess=postprocess,
            ffmpeg_path=ffmpeg_path,
            postprocess_reencode=postprocess_reencode,
            ffmpeg_debug=ffmpeg_debug,
            asset_uploader=asset_uploader,
            render_queue=render_queue,
            postprocessor=postprocessor,
            model_resolver=model_resolver,
            video_settings=video_settings,
            project_dir=project_dir,
            workflow=workflow,
        )
        self.seed_offset = int(seed_offset)
        self.randomize_seed = bool(randomize_seed)
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None
        self.workflow_label = Path(workflow_label) if workflow_label is not None else self.workflow_path
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()
        self.audio_ref_stems = audio_ref_stems

    # -----------------------------------------------------------------------
    # High-level entry points
    # -----------------------------------------------------------------------

    def build_workflow(
        self,
        scene: dict,
        *,
        prompt: str,
        comfy_audio_name: str | None = None,
        duration_seconds: float | None = None,
        frame_count: int | None = None,
        width: int | None = None,
        height: int | None = None,
        ref_image_paths: list[str | Path] | None = None,
        ref_video_paths: list[str | Path] | None = None,
        ref_audio_paths: list[str | Path] | None = None,
    ) -> dict:
        """Build a patched R2V workflow dict from *scene*.

        Patches FeverSlop meta-anchors:
        - ``#PROMPT``  → ``value``
        - ``#SEED``    → ``noise_seed``
        - ``#FRAMECOUNT`` → ``value``
        - ``#MEGAPIXELS`` → ``megapixels`` (computed from width × height when given)
        - ``#REF_1``, ``#REF_2``, … → ``image``
        - ``#VIDEO_1``, ``#VIDEO_2``, ``#VIDEO_3`` → ``video``
        - ``#AUDIO_1``, ``#AUDIO_2``, ``#AUDIO_3`` → ``audio``
        - ``#LOAD_AUDIO`` / ``#TRIM_AUDIO`` (audio workflow only)
        - ``#SAVE_VIDEO`` → ``filename_prefix``
        """
        self._validate_scene(scene)
        scene_number = int(scene.get("scene", 0))

        patcher = WorkflowPatcher(self.load_workflow())

        # MiniMax R2V uses the explicitly numbered reference-audio anchors.
        # The legacy main-audio chain would otherwise occupy ref_audio_0 with
        # a duplicate full mix and shift all prompt references by one slot.
        self._remove_legacy_main_audio_chain(patcher)

        # -- prompt -----------------------------------------------------------
        patcher.set_input_by_title("#PROMPT", "value", str(prompt).strip())

        # -- seed -------------------------------------------------------------
        self._patch_seed(patcher, self._seed_for_scene(scene_number))

        # -- frame count ---------------------------------------------------
        if duration_seconds is not None:
            patcher.set_input_by_title(
                "#FRAMECOUNT", "value", int(round(float(duration_seconds) * 24))
            )

        # -- resolution (megapixels) ------------------------------------------
        if width is not None and height is not None:
            megapixels = (int(width) * int(height)) / 1_000_000
            self._patch_megapixels(patcher, megapixels)

        # -- reference images -------------------------------------------------
        self._patch_reference_images(patcher, ref_image_paths or [])

        # -- reference videos -------------------------------------------------
        self._patch_reference_videos(patcher, ref_video_paths or [])

        # -- reference audios -------------------------------------------------
        self._patch_reference_audios(
            patcher,
            ref_audio_paths or [],
            duration_seconds=duration_seconds,
            abs_start_seconds=scene.get("abs_start_seconds"),
        )

        # -- dynamic ref wiring: fill remaining slots from scene refs -------
        self._patch_dynamic_ref_inputs(patcher, scene)

        # -- output filename --------------------------------------------------
        self._patch_save_video(patcher, scene_number)

        # Remove unused template anchors and their disconnected branches so
        # ComfyUI does not validate stale paths from the original workflow.
        patcher.prune_unreachable_nodes(root_titles=("#SAVE_VIDEO",))

        return patcher.get()

    def render_video(self, request: VideoRenderRequest) -> Path:
        """Complete render flow for one R2V scene."""
        self._validate_scene(request.scene)
        scene_number = int(request.scene_number)

        # -- compute duration / frame count -----------------------------------
        duration_seconds: float | None = None
        raw_duration = request.scene.get("duration_seconds")
        if raw_duration is not None:
            duration_seconds = float(raw_duration)
        frame_count: int | None = None

        # -- resolve reference image paths ------------------------------------
        ref_image_paths = self._resolve_ref_image_paths(request.scene)

        # -- resolve reference video paths ------------------------------------
        ref_video_paths = self._resolve_ref_video_paths(request.scene)

        # -- resolve reference audio paths ------------------------------------
        ref_audio_paths = self._resolve_ref_audio_paths(request.scene)

        # -- resolve stem audio paths (takes priority for slots) ------------------
        stem_audio_paths = self._resolve_stem_audio_paths(request.scene)

        # Merge: stem audio takes priority, then existing scene-level audio refs
        merged_audio: list[Path] = []
        audio_path_set: set[str] = set()
        for p in stem_audio_paths:
            key = str(p)
            if key not in audio_path_set:
                merged_audio.append(p)
                audio_path_set.add(key)
        for p in ref_audio_paths:
            key = str(p)
            if key not in audio_path_set:
                merged_audio.append(p)
                audio_path_set.add(key)
                if len(merged_audio) >= self.MAX_STEM_AUDIOS:
                    break
        ref_audio_paths = merged_audio[: self.MAX_STEM_AUDIOS]

        # -- build workflow ---------------------------------------------------
        workflow = self.build_workflow(
            request.scene,
            prompt=request.prompt,
            duration_seconds=duration_seconds,
            frame_count=frame_count,
            width=int(request.scene.get("width", 0) or 0) or None,
            height=int(request.scene.get("height", 0) or 0) or None,
            ref_image_paths=ref_image_paths,
            ref_video_paths=ref_video_paths,
            ref_audio_paths=ref_audio_paths,
        )

        # -- resolve model references -----------------------------------------
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_label,
        )

        # -- per-scene output directory ---------------------------------------
        scene_dir = self.output_dir / f"scene_{scene_number:04}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # -- scene workflow.json (production artifact) ------------------------
        workflow_path = scene_dir / "workflow.json"
        workflow_path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_scene_manifest(
            request,
            workflow_path,
            pipeline=self.pipeline_name,
            workflow=workflow,
            assets=self._manifest_assets(request),
        )

        # -- debug write ------------------------------------------------------
        self._write_debug_workflow(scene_number, workflow)

        # -- queue and download -------------------------------------------------
        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=scene_dir / "raw.mp4",
        )

        if not self.postprocess:
            return raw_output

        # -- postprocess trim -------------------------------------------------
        # use render plan frame_count for audio sync, fall back to 17N+5
        scene_frame_count = request.scene.get("frame_count")
        if scene_frame_count:
            keep_frames = int(scene_frame_count)
        else:
            keep_frames = self._frames_from_duration(
                duration_seconds if duration_seconds else 5.0
            )
        return self._postprocess_with_audio(
            raw_output,
            TrimSpec(
                source_file=raw_output,
                output_file=scene_dir / "final.mp4",
                fps=self.FPS,
                trim_front_frames=int(self.preroll_frames),
                keep_frames=keep_frames,
                scene=scene_number,
            ),
        )

    # -----------------------------------------------------------------------
    # Patching helpers
    # -----------------------------------------------------------------------

    def _patch_reference_images(
        self,
        patcher: WorkflowPatcher,
        ref_image_paths: list[str | Path] | None,
    ) -> None:
        """Map reference image paths to ``#REF_1``, ``#REF_2``, … anchors."""
        self._clear_reference_group(patcher, "ref_images")
        if not ref_image_paths:
            return
        if len(ref_image_paths) > self.MAX_REF_IMAGES:
            raise FeverSlopValidationError(
                f"At most {self.MAX_REF_IMAGES} reference images allowed, "
                f"got {len(ref_image_paths)}"
            )
        for index, path in enumerate(ref_image_paths, start=1):
            title = f"#REF_{index}"
            image_name = self.asset_uploader.resolve_reference_image_name(path)
            self._patch_reference_asset(
                patcher, title, "LoadImage", "image", image_name, "ref_images", index - 1
            )

    def _patch_reference_videos(
        self,
        patcher: WorkflowPatcher,
        ref_video_paths: list[str | Path] | None,
    ) -> None:
        """Map reference video paths to ``#VIDEO_1``, ``#VIDEO_2``, ``#VIDEO_3`` anchors."""
        self._clear_reference_group(patcher, "ref_videos")
        if not ref_video_paths:
            return
        if len(ref_video_paths) > self.MAX_REF_VIDEOS:
            raise FeverSlopValidationError(
                f"At most {self.MAX_REF_VIDEOS} reference videos allowed, "
                f"got {len(ref_video_paths)}"
            )
        for index, path in enumerate(ref_video_paths, start=1):
            title = f"#VIDEO_{index}"
            video_name = self.asset_uploader.resolve_reference_video_name(path)
            self._patch_reference_asset(
                patcher, title, "LoadVideo", "video", video_name, "ref_videos", index - 1
            )

    @staticmethod
    def _wire_ref_audio_slot(
        patcher: WorkflowPatcher,
        source_node_id: str,
        index: int,
    ) -> None:
        """Wire a node to ref_audios.ref_audio_{index} on the core R2V node."""
        for _, core in patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo"):
            core.setdefault("inputs", {})[
                f"ref_audios.ref_audio_{index}"
            ] = [source_node_id, 0]

    def _patch_reference_audios(
        self,
        patcher: WorkflowPatcher,
        ref_audio_paths: list[str | Path] | None,
        *,
        duration_seconds: float | None = None,
        abs_start_seconds: float | None = None,
    ) -> None:
        """Map reference audio paths through LoadAudio → [TrimAudioDuration] → R2V slots.

        When *duration_seconds* is given, each audio path is trimmed to the scene
        duration before reaching the core node, keeping stem audio in sync with
        the main (comfy_audio) reference.
        """
        self._clear_reference_group(patcher, "ref_audios")
        if not ref_audio_paths:
            return
        if len(ref_audio_paths) > self.MAX_REF_AUDIOS:
            raise FeverSlopValidationError(
                f"At most {self.MAX_REF_AUDIOS} reference audio clips allowed, "
                f"got {len(ref_audio_paths)}"
            )
        for slot_index, path in enumerate(ref_audio_paths):
            title = f"#AUDIO_{slot_index + 1}"
            trim_title = f"#TRIM_AUDIO_{slot_index + 1}"
            audio_name = self.asset_uploader.resolve_reference_audio_name(path)

            # Step 1: Update or create LoadAudio anchor
            try:
                loader_id, loader = patcher.find_node_by_meta_title(title)
                loader.setdefault("inputs", {})["audio"] = audio_name
            except KeyError:
                numeric_ids = [int(nid) for nid in patcher.get() if nid.isdigit()]
                loader_id = str(max(numeric_ids, default=0) + 1)
                patcher.get()[loader_id] = {
                    "class_type": "LoadAudio",
                    "_meta": {"title": title},
                    "inputs": {"audio": audio_name},
                }

            # Step 2: If trimming, wire through TrimAudioDuration
            if duration_seconds is not None:
                try:
                    trim_id, trim = patcher.find_node_by_meta_title(trim_title)
                except KeyError:
                    numeric_ids = [int(nid) for nid in patcher.get() if nid.isdigit()]
                    trim_id = str(max(numeric_ids, default=0) + 1)
                    patcher.get()[trim_id] = {
                        "class_type": "TrimAudioDuration",
                        "_meta": {"title": trim_title},
                        "inputs": {
                            "start_index": 0.0,
                            "duration": float(duration_seconds),
                            "audio": [loader_id, 0],
                        },
                    }
                    trim = patcher.get()[trim_id]

                else:
                    # Update existing trim node to point at the loader
                    trim.setdefault("inputs", {})["audio"] = [loader_id, 0]

                start = float(abs_start_seconds) if abs_start_seconds is not None else 0.0
                trim.setdefault("inputs", {})["start_index"] = start
                trim.setdefault("inputs", {})["duration"] = float(duration_seconds)

                self._wire_ref_audio_slot(patcher, trim_id, slot_index)
            else:
                self._wire_ref_audio_slot(patcher, loader_id, slot_index)

    @staticmethod
    def _remove_legacy_main_audio_chain(patcher: WorkflowPatcher) -> None:
        """Remove the template's duplicate full-mix reference-audio chain."""
        removed_ids: set[str] = set()
        for title in ("#LOAD_AUDIO", "#TRIM_AUDIO"):
            for node_id, _ in patcher.find_nodes_by_meta_title(title):
                removed_ids.add(node_id)
                del patcher.get()[node_id]

        if not removed_ids:
            return
        for _, core in patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo"):
            inputs = core.setdefault("inputs", {})
            for input_name, value in list(inputs.items()):
                if (
                    input_name.startswith("ref_audios.")
                    and isinstance(value, list)
                    and value
                    and str(value[0]) in removed_ids
                ):
                    del inputs[input_name]

    @staticmethod
    def _clear_reference_group(patcher: WorkflowPatcher, input_group: str) -> None:
        for _, core in patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo"):
            inputs = core.setdefault("inputs", {})
            for input_name in list(inputs):
                if input_name.startswith(f"{input_group}."):
                    del inputs[input_name]

    def _patch_reference_asset(
        self,
        patcher: WorkflowPatcher,
        title: str,
        class_type: str,
        loader_input: str,
        asset_name: str,
        input_group: str,
        index: int,
    ) -> None:
        try:
            loader_id, loader = patcher.find_node_by_meta_title(title)
            loader.setdefault("inputs", {})[loader_input] = asset_name
        except KeyError:
            core_nodes = patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo")
            if not core_nodes:
                return
            numeric_ids = [int(node_id) for node_id in patcher.get() if node_id.isdigit()]
            loader_id = str(max(numeric_ids, default=0) + 1)
            patcher.get()[loader_id] = {
                "class_type": class_type,
                "_meta": {"title": title},
                "inputs": {loader_input: asset_name},
            }

        core_nodes = patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo")
        if core_nodes:
            if input_group == "ref_audios":
                self._wire_ref_audio_slot(patcher, loader_id, index)
            else:
                core_nodes[0][1].setdefault("inputs", {})[
                    f"{input_group}.{input_group[:-1]}_{index}"
                ] = [loader_id, 0]

    def _patch_audio_inputs(
        self,
        patcher: WorkflowPatcher,
        comfy_audio_name: str,
        duration_seconds: float | None = None,
        scene: dict | None = None,
    ) -> None:
        """Patch ``#LOAD_AUDIO`` and ``#TRIM_AUDIO`` anchors (audio workflow only).

        Also wires the trimmed audio output to ref_audios.ref_audio_0 on the
        MiniMaxH3ReferenceToVideo node and patches start_index from scene data.
        """
        if patcher.try_set_existing_input_by_title(
            "#LOAD_AUDIO", "audio", comfy_audio_name
        ):
            patcher.try_set_existing_input_by_title(
                "#LOAD_AUDIO",
                "audioUI",
                f"/api/view?filename={comfy_audio_name}&type=input",
            )
        # -- patch start_index and duration on #TRIM_AUDIO --------------------
        start_index: float = 0.0
        if scene is not None:
            raw_start = scene.get("abs_start_seconds")
            if raw_start is not None:
                start_index = float(raw_start)
        patcher.try_set_existing_input_by_title("#TRIM_AUDIO", "start_index", start_index)
        if duration_seconds is not None:
            patcher.try_set_existing_input_by_title(
                "#TRIM_AUDIO", "duration", float(duration_seconds)
            )
        # -- wire trimmed audio to MiniMaxH3ReferenceToVideo ------------------
        self._wire_trimmed_audio_to_r2v(patcher)

    @staticmethod
    def _wire_trimmed_audio_to_r2v(patcher: WorkflowPatcher) -> None:
        """Connect the #TRIM_AUDIO output to ref_audios.ref_audio_0 on the core node."""
        try:
            trim_node_id, _ = patcher.find_node_by_meta_title("#TRIM_AUDIO")
        except KeyError:
            return
        core_nodes = patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo")
        if not core_nodes:
            return
        core_nodes[0][1].setdefault("inputs", {})["ref_audios.ref_audio_0"] = [
            trim_node_id,
            0,
        ]

    # -----------------------------------------------------------------------
    # Dynamic ref wiring
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_occupied_ref_slots(
        patcher: WorkflowPatcher,
        slot_type: str,
    ) -> set[int]:
        """Return set of occupied input slot indices on the core node.

        *slot_type* is one of ``ref_images``, ``ref_videos``, ``ref_audios``.
        """
        occupied: set[int] = set()
        for _, core in patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo"):
            inputs = core.get("inputs", {})
            for input_name in inputs:
                if input_name.startswith(f"{slot_type}."):
                    suffix = input_name[len(f"{slot_type}."):]
                    # suffix is like "ref_image_0" -> extract 0
                    try:
                        idx = int(suffix.split("_")[-1])
                        occupied.add(idx)
                    except (ValueError, IndexError):
                        continue
        return occupied

    def _add_ref_node_and_wire(
        self,
        patcher: WorkflowPatcher,
        slot_type: str,
        slot_index: int,
        source_path: str | Path,
    ) -> str:
        """Create a loader node for *source_path*, wire to core node at *slot_index*.

        Returns the prompt tag string (e.g. ``"<Picture 5>"`` for images).
        """
        class_type_map = {
            "ref_images": ("LoadImage", "image", "ref_image"),
            "ref_videos": ("LoadVideo", "video", "ref_video"),
            "ref_audios": ("LoadAudio", "audio", "ref_audio"),
        }
        class_type, loader_input, input_singular = class_type_map[slot_type]

        # Resolve asset name
        if slot_type == "ref_images":
            asset_name = self.asset_uploader.resolve_reference_image_name(source_path)
        elif slot_type == "ref_videos":
            asset_name = self.asset_uploader.resolve_reference_video_name(source_path)
        else:
            asset_name = self.asset_uploader.resolve_reference_audio_name(source_path)

        # Create loader node with fresh ID
        loader_id = str(patcher.find_free_node_id())
        title = f"#DYN_{slot_type}_{slot_index}"
        patcher.add_node(loader_id, {
            "class_type": class_type,
            "_meta": {"title": title},
            "inputs": {loader_input: asset_name},
        })

        # Wire to core node
        for _, core in patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo"):
            core.setdefault("inputs", {})[
                f"{slot_type}.{input_singular}_{slot_index}"
            ] = [loader_id, 0]

        # Build prompt tag
        if slot_type == "ref_images":
            return f"<Picture {slot_index + 1}>"
        elif slot_type == "ref_videos":
            return f"<Video {slot_index + 1}>"
        else:
            return f"<Audio {slot_index + 1}>"

    def _collect_scene_references(
        self,
        scene: dict,
    ) -> tuple[list[str], list[str], list[str]]:
        """Collect all scene references by type.

        Returns ``(image_paths, video_paths, audio_paths)`` — each already clamped to max.
        """
        image_paths = [str(p) for p in self._resolve_ref_image_paths(scene)]
        video_paths = [str(p) for p in self._resolve_ref_video_paths(scene)]
        audio_paths = [str(p) for p in self._resolve_ref_audio_paths(scene)]
        return image_paths, video_paths, audio_paths

    def _patch_dynamic_ref_inputs(
        self,
        patcher: WorkflowPatcher,
        scene: dict,
    ) -> list[str]:
        """Wire all scene references to unoccupied core node slots.

        Collects refs from *scene*, finds occupied slots, fills empty slots with
        new loader nodes. Returns list of prompt tags.
        """
        prompt_tags: list[str] = []

        image_paths, video_paths, audio_paths = self._collect_scene_references(scene)

        for slot_type, paths, max_count in [
            ("ref_images", image_paths, self.MAX_REF_IMAGES),
            ("ref_videos", video_paths, self.MAX_REF_VIDEOS),
            ("ref_audios", audio_paths, self.MAX_REF_AUDIOS),
        ]:
            occupied = self._find_occupied_ref_slots(patcher, slot_type)
            for i, path in enumerate(paths):
                if i in occupied:
                    continue  # slot already filled by pre-wired anchor
                if i >= max_count:
                    break
                tag = self._add_ref_node_and_wire(patcher, slot_type, i, path)
                prompt_tags.append(tag)

        return prompt_tags

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _resolve_stem_audio_paths(
        self,
        scene: dict,
    ) -> list[Path]:
        """Resolve stem-based audio reference paths from scene stem_audio section.

        Uses instance-level audio_ref_stems if set, otherwise scene-level stem list.
        Returns empty list if no stem audio available.
        """
        stem_audio = (scene.get("stem_audio") or {})
        stem_names: list[str] = self.audio_ref_stems or list(stem_audio.get("stems", []))
        paths_map: dict[str, str] = stem_audio.get("paths", {})
        if not stem_audio and self.audio_ref_stems:
            paths_map = self._fallback_stem_paths()
        if not paths_map or not stem_names:
            return []

        # Priority ordering: lip-sync-critical stems (vocals, full_mix) first,
        # then any additional stems in original order.
        priority_order = ["vocals", "full_mix"]
        ordered_names = [n for n in priority_order if n in stem_names and n in paths_map]
        for name in stem_names:
            if name not in ordered_names and name in paths_map:
                ordered_names.append(name)

        result: list[Path] = []
        for stem_name in ordered_names:
            path_str = paths_map.get(stem_name)
            if path_str:
                p = self._resolve_project_path(path_str)
                if p.exists():
                    result.append(p)
        return result[: self.MAX_REF_AUDIOS]

    def _manifest_assets(self, request: VideoRenderRequest) -> list[tuple]:
        assets = [
            self._reference_asset("reference_image", path)
            for path in self._resolve_ref_image_paths(request.scene)
        ]
        assets.extend(
            self._reference_asset("reference_video", path)
            for path in self._resolve_ref_video_paths(request.scene)
        )
        audio_paths = [
            *self._resolve_stem_audio_paths(request.scene),
            *self._resolve_ref_audio_paths(request.scene),
        ]
        unique_audio = list(dict.fromkeys(audio_paths))[: self.MAX_REF_AUDIOS]
        assets.extend(
            self._reference_asset("reference_audio", path)
            for path in unique_audio
        )
        return assets

    def _fallback_stem_paths(self) -> dict[str, str]:
        """Find generated Demucs stems when an older render plan lacks metadata."""
        if self.project_dir is None:
            return {}
        stem_dir = self.project_dir / "output" / "stems"
        result: dict[str, str] = {}
        for stem_name in self.audio_ref_stems or []:
            if stem_name == "full_mix":
                continue
            matches = sorted(stem_dir.glob(f"{stem_name}_*.wav"))
            if matches:
                result[stem_name] = str(matches[0])
        return result

    def _validate_scene(self, scene: dict) -> None:
        """Validate that the scene has at least one actor reference."""
        references = scene.get("references") or {}
        actor_paths = (
            references.get("actor_sheet_paths", [])
            or references.get("actor_msr_paths", [])
        )
        if not actor_paths:
            scene_number = scene.get("scene", "?")
            raise FeverSlopValidationError(
                f"Scene {scene_number} requires at least one actor reference"
            )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _seed_for_scene(self, scene_number: int) -> int:
        if self.randomize_seed:
            return random.randint(0, 2**63 - 1)
        return self.seed_offset + int(scene_number)

    def _resolve_ref_image_paths(self, scene: dict) -> list[Path]:
        """Extract and resolve reference image paths from a scene dict."""
        references = scene.get("references") or {}

        actor_paths = (
            references.get("actor_sheet_paths", [])
            or references.get("actor_msr_paths", [])
        )
        paths: list[Path] = [self._resolve_project_path(p) for p in actor_paths]

        location_path = (
            references.get("location_sheet_path")
            or references.get("location_msr_path")
        )
        if location_path:
            paths.append(self._resolve_project_path(location_path))

        style_paths = references.get("style_reference_paths", [])
        if self.project_dir is not None:
            style_paths = [self.project_dir / p for p in style_paths]
        paths.extend(str(p) for p in style_paths)

        return paths[: self.MAX_REF_IMAGES]

    def _resolve_ref_video_paths(self, scene: dict) -> list[Path]:
        """Extract and resolve reference video paths from a scene dict."""
        references = scene.get("references") or {}
        video_paths = references.get("reference_video_paths", [])[:self.MAX_REF_VIDEOS]
        if self.project_dir is not None:
            video_paths = [self._resolve_project_path(p) for p in video_paths]
        return list(video_paths)

    def _resolve_ref_audio_paths(self, scene: dict) -> list[Path]:
        """Extract and resolve reference audio paths from a scene dict."""
        references = scene.get("references") or {}
        audio_paths = references.get("reference_audio_paths", [])[:self.MAX_REF_AUDIOS]
        if self.project_dir is not None:
            audio_paths = [self._resolve_project_path(p) for p in audio_paths]
        return list(audio_paths)

    def _resolve_project_path(self, path: str | Path) -> Path:
        if self.project_dir is None:
            return coerce_local_path(path)
        return coerce_local_path(path, base_dir=self.project_dir)

    def _write_debug_workflow(self, scene_number: int, workflow: dict) -> None:
        if self.debug_workflows_dir is None:
            return
        self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
        (
            self.debug_workflows_dir / f"scene_{scene_number:04}_workflow.json"
        ).write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _has_anchor(patcher: WorkflowPatcher, title: str) -> bool:
        try:
            patcher.find_node_by_meta_title(title)
            return True
        except KeyError:
            return False
