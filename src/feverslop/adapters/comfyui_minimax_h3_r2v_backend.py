from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_minimax_h3_video_backend import (
    ComfyUIMiniMaxH3VideoRenderBackend,
)
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.config.video_settings import VideoSettings
from feverslop.domain.postprocessing import TrimSpec
from feverslop.domain.h3_two_pass import H3TwoPassSpec, apply_h3_two_pass_patch
from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.continuity import BoundaryFrameManifest
from feverslop.errors import FeverSlopValidationError
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
        input_audio: str | Path | None = None,
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
        self.input_audio = Path(input_audio) if input_audio is not None else None

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
        megapixels: float | None = None,
        ref_image_paths: list[str | Path] | None = None,
        ref_video_paths: list[str | Path] | None = None,
        ref_audio_paths: list[str | Path] | None = None,
        two_pass_spec: H3TwoPassSpec | dict | None = None,
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
        continuity_manifest = self._continuity_manifest(scene)
        continuity_anchor = self._resolve_continuity_anchor_path(scene)
        if continuity_anchor:
            scene = dict(scene)
            keyframes = dict(scene.get("keyframes") or {})
            keyframes["continuity_anchor_path"] = continuity_anchor.as_posix()
            scene["keyframes"] = keyframes
            if continuity_manifest is not None:
                refs = list(ref_image_paths or [])
                refs = [path for path in refs if Path(path).resolve() != continuity_anchor]
                ref_image_paths = [*refs[: self.MAX_REF_IMAGES - 1], continuity_anchor]

        patcher = WorkflowPatcher(self.load_workflow())

        required_titles = ("#PROMPT", "#SAVE_VIDEO")
        missing_titles = [
            title for title in required_titles
            if not patcher.find_nodes_by_meta_title(title)
        ]
        if missing_titles:
            raise FeverSlopValidationError(
                "MiniMax H3 R2V workflow is incompatible: missing required anchors "
                + ", ".join(missing_titles)
                + ". Select a MiniMax H3 R2V workflow with the native anchor contract.",
            )

        # MiniMax R2V uses the explicitly numbered reference-audio anchors.
        # The legacy main-audio chain would otherwise occupy ref_audio_0 with
        # a duplicate full mix and shift all prompt references by one slot.
        self._remove_legacy_main_audio_chain(patcher)

        # -- prompt -----------------------------------------------------------
        resolved_prompt = self._append_continuity_anchor_prompt(
            str(prompt).strip(), scene, len(ref_image_paths or []),
            manifest=continuity_manifest,
        )
        patcher.set_input_by_title("#PROMPT", "value", resolved_prompt)

        # -- seed -------------------------------------------------------------
        self._patch_seed(patcher, self._seed_for_scene(scene))

        # -- frame count ---------------------------------------------------
        if duration_seconds is not None:
            patcher.set_input_by_title(
                "#FRAMECOUNT", "value", int(round(float(duration_seconds) * 24)),
            )

        # -- resolution (megapixels) ------------------------------------------
        if megapixels is not None:
            self._patch_megapixels(patcher, megapixels, explicit=True)
        elif width is not None and height is not None:
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

        if two_pass_spec is not None:
            spec = two_pass_spec if isinstance(two_pass_spec, H3TwoPassSpec) else H3TwoPassSpec.from_dict(two_pass_spec)
            self._progress("h3_passes_validating")
            patcher = WorkflowPatcher(apply_h3_two_pass_patch(patcher.get(), spec))
            self._progress("h3_passes_ready")

        # -- output filename --------------------------------------------------
        self._patch_save_video(patcher, scene_number)

        # Remove unused template anchors and their disconnected branches so
        # ComfyUI does not validate stale paths from the original workflow.
        patcher.prune_unreachable_nodes(root_titles=("#SAVE_VIDEO",))

        return patcher.get()

    @staticmethod
    def _append_continuity_anchor_prompt(
        prompt: str,
        scene: dict,
        reference_count: int,
        *,
        manifest: BoundaryFrameManifest | None = None,
    ) -> str:
        """Declare the reserved final R2V image slot as a prior-scene anchor."""
        if manifest is None or reference_count < 1:
            return prompt
        return f"{prompt}\n\n{ComfyUIMiniMaxH3R2VBackend._compile_continuity_start_state(manifest, picture_slot=reference_count)}".strip()

    @staticmethod
    def _compile_continuity_start_state(
        manifest: BoundaryFrameManifest,
        *,
        picture_slot: int,
    ) -> str:
        """Compile deterministic H3 instructions from a verified boundary."""
        return (
            f"Start state: <Picture {picture_slot}> is the verified predecessor boundary "
            f"frame {manifest.frame_index} from extractor {manifest.extractor_revision}. "
            "Preserve composition, subject identity, spatial layout, lighting, "
            "and motion direction."
        )

    def _continuity_manifest(self, scene: dict) -> BoundaryFrameManifest | None:
        payload = (scene.get("keyframes") or {}).get("boundary_frame_manifest")
        if payload is None:
            transition = (scene.get("visual_consistency") or {}).get("transition_from_previous")
            if transition == "continuous" and int(scene.get("scene") or 0) > 1:
                raise ValueError("continuous H3 successor requires a verified boundary frame manifest")
            return None
        return BoundaryFrameManifest.from_dict(payload)

    def _resolve_continuity_anchor_path(self, scene: dict) -> Path | None:
        manifest = self._continuity_manifest(scene)
        if manifest is None:
            return None
        if self.project_dir is None:
            raise ValueError("verified boundary frame requires a project directory")
        frame = self._resolve_project_path(manifest.frame_path)
        source_clip = self._resolve_project_path(manifest.source_clip_path)
        if not frame.is_file() or sha256_file(frame) != manifest.frame_sha256:
            raise ValueError("verified boundary frame is missing or stale")
        if not source_clip.is_file() or sha256_file(source_clip) != manifest.source_clip_sha256:
            raise ValueError("verified boundary source clip is missing or stale")
        return frame

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
        ref_audio_paths = self._filter_audio_paths_for_window(
            merged_audio[: self.MAX_STEM_AUDIOS],
            start_seconds=float(request.scene.get("abs_start_seconds", 0.0) or 0.0),
            duration_seconds=duration_seconds,
        )

        # -- build workflow ---------------------------------------------------
        workflow = self.build_workflow(
            request.scene,
            prompt=request.prompt,
            duration_seconds=duration_seconds,
            frame_count=frame_count,
            width=int(request.scene.get("width", 0) or 0) or None,
            height=int(request.scene.get("height", 0) or 0) or None,
            megapixels=(
                float(request.scene["megapixels"])
                if request.scene.get("megapixels") is not None
                else (self.video_settings.megapixels if self.video_settings else None)
            ),
            ref_image_paths=ref_image_paths,
            ref_video_paths=ref_video_paths,
            ref_audio_paths=ref_audio_paths,
        )

        # -- resolve model references -----------------------------------------
        workflow = self.model_resolver.resolve_workflow_models(
            workflow,
            workflow_path=self.workflow_label,
        )
        self._preflight_comfy_node_classes(workflow)

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
        self._progress("h3_render_submitting")
        raw_output = self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=scene_dir / "raw.mp4",
        )
        self._progress("h3_render_completed")

        if not self.postprocess:
            return raw_output

        # -- postprocess trim -------------------------------------------------
        # use render plan frame_count for audio sync, fall back to 17N+5
        scene_frame_count = request.scene.get("frame_count")
        if scene_frame_count:
            keep_frames = int(scene_frame_count)
        else:
            keep_frames = self._frames_from_duration(
                duration_seconds or 5.0,
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
                extract_boundary_frames=True,
            ),
        )

    def _preflight_comfy_node_classes(self, workflow: dict) -> None:
        """Fail before queueing when ComfyUI cannot provide workflow nodes."""
        get_object_info = getattr(self.client, "get_object_info", None)
        if not callable(get_object_info):
            return
        available = set(get_object_info())
        required = {
            str(node.get("class_type"))
            for node in workflow.values()
            if node.get("class_type")
        }
        missing = sorted(required - available)
        if missing:
            raise FeverSlopValidationError(
                "ComfyUI is missing required H3 workflow nodes: "
                + ", ".join(missing)
                + ". Install/load the matching native H3 node package before rendering.",
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
                f"got {len(ref_image_paths)}",
            )
        for index, path in enumerate(ref_image_paths, start=1):
            title = f"#REF_{index}"
            image_name = self.asset_uploader.resolve_reference_image_name(path)
            self._patch_reference_asset(
                patcher, title, "LoadImage", "image", image_name, "ref_images", index - 1,
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
                f"got {len(ref_video_paths)}",
            )
        for index, path in enumerate(ref_video_paths, start=1):
            title = f"#VIDEO_{index}"
            video_name = self.asset_uploader.resolve_reference_video_name(path)
            self._patch_reference_asset(
                patcher, title, "LoadVideo", "video", video_name, "ref_videos", index - 1,
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
                f"got {len(ref_audio_paths)}",
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
            "#LOAD_AUDIO", "audio", comfy_audio_name,
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
                "#TRIM_AUDIO", "duration", float(duration_seconds),
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
        if slot_type == "ref_videos":
            return f"<Video {slot_index + 1}>"
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

    @staticmethod
    def _audio_duration(path: Path) -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    @classmethod
    def _filter_audio_paths_for_window(
        cls,
        paths: list[Path],
        *,
        start_seconds: float,
        duration_seconds: float | None,
    ) -> list[Path]:
        if duration_seconds is None:
            return paths

        window_end = float(start_seconds) + float(duration_seconds)
        valid_paths: list[Path] = []
        for path in paths:
            source_duration = cls._audio_duration(path)
            if source_duration is None or source_duration + 0.05 >= window_end:
                valid_paths.append(path)
        return valid_paths

    def _resolve_stem_audio_paths(
        self,
        scene: dict,
    ) -> list[Path]:
        """Resolve stem-based audio reference paths from scene stem_audio section.

        Uses instance-level audio_ref_stems if set, otherwise scene-level stem list.
        Returns empty list if no stem audio available.
        """
        stem_audio = (scene.get("stem_audio") or {})
        stem_names: list[str] = (
            list(stem_audio.get("stems", []))
            if stem_audio
            else list(self.audio_ref_stems or [])
        )
        paths_map: dict[str, str] = stem_audio.get("paths", {})
        if not stem_audio and self.audio_ref_stems:
            paths_map = self._fallback_stem_paths()
        if not paths_map or not stem_names:
            return []

        silent_mode = bool(
            scene.get("silent_mode") or (scene.get("metadata") or {}).get("silent_mode"),
        )
        if silent_mode:
            stem_names = [name for name in stem_names if name != "vocals"]
        ordered_names = list(dict.fromkeys(
            name for name in stem_names if name in paths_map
        ))

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
            if self.input_audio is None:
                matches = sorted(stem_dir.glob(f"{stem_name}_*.wav"))
            else:
                matches = [stem_dir / f"{stem_name}_{self.input_audio.stem}.wav"]
                matches = [path for path in matches if path.is_file()]
            if matches:
                result[stem_name] = str(matches[0])
        return result

    def _validate_scene(self, scene: dict) -> None:
        """Validate that the scene has an actor or location reference."""
        references = scene.get("references") or {}
        actor_paths = (
            references.get("actor_sheet_paths", [])
            or references.get("actor_msr_paths", [])
        )
        if str(references.get("subject_mode") or "").strip().lower() == "location_only":
            actor_paths = []
        location_paths = (
            references.get("location_sheet_path", "")
            or references.get("location_msr_path", "")
        )
        if not actor_paths and not location_paths:
            scene_number = scene.get("scene", "?")
            raise FeverSlopValidationError(
                f"Scene {scene_number} requires at least one actor or location reference",
            )
        self._validate_h3_reference_contract(scene)

    def _validate_h3_reference_contract(self, scene: dict) -> None:
        """Reject structured H3 prompts that do not match bound workflow slots."""
        prompt = str(((scene.get("h3") or {}).get("prompt")) or "").strip()
        if not prompt.startswith("subject_definitions:"):
            return
        scene_number = scene.get("scene", "?")
        definitions = prompt.split("summary:", 1)[0]
        defined_subjects = set(re.findall(r"<Subject\s+\d+>", definitions))
        used_subjects = set(re.findall(r"<Subject\s+\d+>", prompt))
        undefined_subjects = used_subjects - defined_subjects

        picture_count = len(self._resolve_ref_image_paths(scene))
        expected_pictures = {f"<Picture {index}>" for index in range(1, picture_count + 1)}
        bound_picture_list = re.findall(r"<Picture\s+\d+>", definitions)
        bound_pictures = set(bound_picture_list)
        used_pictures = set(re.findall(r"<Picture\s+\d+>", prompt))
        unbound_pictures = expected_pictures - bound_pictures
        unknown_pictures = used_pictures - expected_pictures

        video_count = len(self._resolve_ref_video_paths(scene))
        expected_videos = {f"<Video {index}>" for index in range(1, video_count + 1)}
        used_videos = set(re.findall(r"<Video\s+\d+>", prompt))
        missing_videos = expected_videos - used_videos
        unknown_videos = used_videos - expected_videos

        audio_paths = [
            *self._resolve_stem_audio_paths(scene),
            *self._resolve_ref_audio_paths(scene),
        ]
        audio_count = min(len(dict.fromkeys(Path(path) for path in audio_paths)), self.MAX_REF_AUDIOS)
        expected_audio = {f"<Audio {index}>" for index in range(1, audio_count + 1)}
        used_audio = set(re.findall(r"<Audio\s+\d+>", prompt))
        defined_audio = set(re.findall(r"<Audio\s+\d+>", definitions))
        missing_audio = expected_audio - defined_audio
        unknown_audio = used_audio - expected_audio

        if any((
            undefined_subjects,
            unbound_pictures,
            unknown_pictures,
            missing_videos,
            unknown_videos,
            missing_audio,
            unknown_audio,
        )):
            raise FeverSlopValidationError(
                f"Scene {scene_number} H3 reference contract mismatch: "
                f"undefined_subjects={sorted(undefined_subjects)!r}; "
                f"unbound_pictures={sorted(unbound_pictures)!r}; "
                f"unknown_pictures={sorted(unknown_pictures)!r}; "
                f"missing_videos={sorted(missing_videos)!r}; "
                f"unknown_videos={sorted(unknown_videos)!r}; "
                f"missing_audio={sorted(missing_audio)!r}; "
                f"unknown_audio={sorted(unknown_audio)!r}",
            )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _seed_for_scene(self, scene: int | dict) -> int:
        if self.randomize_seed:
            return random.randint(0, 2**63 - 1)
        if isinstance(scene, dict) and scene.get("seed") is not None:
            return int(scene["seed"])
        scene_number = int(scene.get("scene", 0)) if isinstance(scene, dict) else int(scene)
        return self.seed_offset + int(scene_number)

    def _resolve_ref_image_paths(self, scene: dict) -> list[Path]:
        """Extract and resolve reference image paths from a scene dict."""
        self._continuity_manifest(scene)
        references = scene.get("references") or {}

        actor_paths = (
            references.get("actor_sheet_paths", [])
            or references.get("actor_msr_paths", [])
        )
        if str(references.get("subject_mode") or "").strip().lower() == "location_only":
            actor_paths = []
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

        anchor_path = self._resolve_continuity_anchor_path(scene)
        if anchor_path:
            paths = paths[: self.MAX_REF_IMAGES - 1]
            paths.append(anchor_path)
            return paths

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
        audio_paths = list(references.get("reference_audio_paths", []))
        silent_mode = bool(
            scene.get("silent_mode") or (scene.get("metadata") or {}).get("silent_mode"),
        )
        vocal_path = str(
            ((scene.get("stem_audio") or {}).get("paths") or {}).get("vocals") or "",
        )
        if silent_mode and vocal_path:
            audio_paths = [path for path in audio_paths if str(path) != vocal_path]
        audio_paths = audio_paths[:self.MAX_REF_AUDIOS]
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
