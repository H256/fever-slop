from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.utils.io import read_json


class ComfyUIStartframeDirectorVisualAdapter:
    def __init__(
        self,
        *,
        client,
        director_workflow_path: str | Path,
        mask_workflow_path: str | Path,
        identity_repair_workflow_path: str | Path,
        detail_workflow_path: str | Path,
        video_use_case,
        validator,
        i2v_workflow_path: str | Path = "workflows/video_ltxv_i2v_native_audio_v2.json",
        model_resolver=None,
        debug_workflows_dir: str | Path | None = None,
    ):
        self.client = client
        self.director_workflow_path = Path(director_workflow_path)
        self.mask_workflow_path = Path(mask_workflow_path)
        self.identity_repair_workflow_path = Path(identity_repair_workflow_path)
        self.detail_workflow_path = Path(detail_workflow_path)
        self.i2v_workflow_path = Path(i2v_workflow_path)
        self.video_use_case = video_use_case
        self.validator = validator
        self.model_resolver = model_resolver
        self.debug_workflows_dir = Path(debug_workflows_dir) if debug_workflows_dir else None

    def render_movie(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        selected_scenes: list[int] | None = None,
        concat_only: bool = False,
        continuity_keyframes: str = "none",
        on_clip_rendered: Callable[[int, int, int], None] | None = None,
        on_startframe_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        project_dir = Path(project_dir)
        storyboard_dir = self._render_startframes(
            project_dir=project_dir,
            selected_scenes=set(selected_scenes or []),
            on_startframe_step=on_startframe_step,
        )
        ltx_dir = project_dir / "output" / "movie" / "ltx_startframe_director"
        rendered = self.video_use_case.execute(
            SimpleNamespace(
                render_plan_path=render_plan_path,
                workflow_path=self.i2v_workflow_path,
                single_prompt_workflow_path=self.i2v_workflow_path,
                audio_file=project_dir / "movie" / "ltx_native_audio.wav",
                storyboard_dir=storyboard_dir,
                output_dir=ltx_dir,
                render_mode="single_prompt",
                limit=None,
                scene_numbers=selected_scenes,
                skip_existing=True,
                uploaded_audio_name=None,
                upload_audio=False,
                upload_startframes=True,
                anchors=WorkflowAnchorConfig(),
                on_scene_complete=on_clip_rendered,
            ),
        )
        final = project_dir / "output" / "movie" / "startframe-director.mp4"
        final.parent.mkdir(parents=True, exist_ok=True)
        if rendered:
            shutil.copyfile(Path(rendered[0]), final)
        else:
            final.write_bytes(b"")
        return final

    def _render_startframes(
        self,
        *,
        project_dir: Path,
        selected_scenes: set[int],
        on_startframe_step: Callable[[dict[str, Any]], None] | None,
    ) -> Path:
        plan = read_json(project_dir / "movie" / "startframe_plan.json")
        prompts = read_json(project_dir / "movie" / "startframe_director_prompts.json")
        identity = read_json(project_dir / "movie" / "identity_ledger.json")
        prompt_by_shot = {str(item.get("shot_id")): item for item in prompts.get("shots") or []}
        shots = [shot for shot in plan.get("shots") or [] if not selected_scenes or int(shot.get("scene") or 0) in selected_scenes]
        total = sum(1 + len(shot.get("actors") or []) * 2 + 2 for shot in shots)
        completed = 0
        final_dir = project_dir / "output" / "movie" / "storyboard" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        validations = []

        for shot in shots:
            scene = int(shot.get("scene") or len(validations) + 1)
            prompt = prompt_by_shot.get(str(shot.get("shot_id"))) or {}
            current = self._render_director(project_dir=project_dir, shot=shot, prompt=prompt)
            completed = _notify(on_startframe_step, completed, total, scene, "director")

            for actor in shot.get("actors") or []:
                actor_id = str(actor.get("actor_id") or "")
                mask = self._render_mask(
                    project_dir=project_dir,
                    scene=scene,
                    actor=actor,
                    source_image=current,
                    identity_ledger=identity,
                )
                completed = _notify(on_startframe_step, completed, total, scene, "mask", actor_id)
                current = self._render_identity_repair(
                    project_dir=project_dir,
                    scene=scene,
                    actor=actor,
                    source_image=current,
                    mask_image=mask,
                    identity_ledger=identity,
                )
                completed = _notify(on_startframe_step, completed, total, scene, "repair", actor_id)

            current = self._render_detail(project_dir=project_dir, scene=scene, source_image=current, shot=shot)
            completed = _notify(on_startframe_step, completed, total, scene, "detail")
            final_path = final_dir / f"scene_{scene:04}.png"
            shutil.copyfile(current, final_path)
            validation = self.validator.validate_startframe(
                image_path=final_path,
                shot_contract=shot,
                identity_ledger=identity,
            )
            validations.append({"scene": scene, "shot_id": str(shot.get("shot_id") or ""), **validation})
            completed = _notify(on_startframe_step, completed, total, scene, "validation")

        validation_path = project_dir / "movie" / "startframe_validation.json"
        validation_path.write_text(
            json.dumps({"version": 1, "validator": _validator_label(self.validator), "shots": validations}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return final_dir

    def _render_director(self, *, project_dir: Path, shot: dict[str, Any], prompt: dict[str, Any]) -> Path:
        scene = int(shot.get("scene") or 0)
        patcher = WorkflowPatcher(read_json(self.director_workflow_path))
        patcher.set_existing_input_by_title_any("#PROMPT_POSITIVE", "text", str(prompt.get("positive_prompt") or ""))
        patcher.set_existing_input_by_title_any("#PROMPT_NEGATIVE", "text", str(prompt.get("negative_prompt") or ""))
        _patch_dimensions(
            patcher,
            width=int(prompt.get("width") or shot.get("width") or 1280),
            height=int(prompt.get("height") or shot.get("height") or 704),
        )
        patcher.set_input_by_title("#SAVE_IMAGE", "filename_prefix", f"startframe/director/scene_{scene:04}")
        _patch_seed_inputs(patcher, 300000 + scene)
        return self._queue_and_download(
            patcher.get(),
            self.director_workflow_path,
            project_dir / "output" / "movie" / "startframes" / "director" / f"scene_{scene:04}.png",
            debug_name=f"scene_{scene:04}_director",
        )

    def _render_mask(
        self,
        *,
        project_dir: Path,
        scene: int,
        actor: dict[str, Any],
        source_image: Path,
        identity_ledger: dict[str, Any],
    ) -> Path:
        actor_id = str(actor.get("actor_id") or "")
        actor_contract = (identity_ledger.get("actors") or {}).get(actor_id) or {}
        patcher = WorkflowPatcher(read_json(self.mask_workflow_path))
        patcher.set_input_by_title("#INPUT_IMAGE", "image", self._upload(source_image, "feverslop/startframe/director"))
        patcher.set_existing_input_by_title("#SEGMENT_PROMPT", "prompt", _mask_prompt(actor_id, actor_contract))
        patcher.set_input_by_title("#SAVE_MASK", "filename_prefix", f"startframe/masks/scene_{scene:04}_{actor_id}")
        return self._queue_and_download(
            patcher.get(),
            self.mask_workflow_path,
            project_dir / "output" / "movie" / "startframes" / "masks" / f"scene_{scene:04}_{actor_id}.png",
            debug_name=f"scene_{scene:04}_mask_{actor_id}",
        )

    def _render_identity_repair(
        self,
        *,
        project_dir: Path,
        scene: int,
        actor: dict[str, Any],
        source_image: Path,
        mask_image: Path,
        identity_ledger: dict[str, Any],
    ) -> Path:
        actor_id = str(actor.get("actor_id") or "")
        actor_contract = (identity_ledger.get("actors") or {}).get(actor_id) or {}
        reference = project_dir / str((actor_contract.get("reference_paths") or {}).get("full_body") or "")
        prompt = _repair_prompt(actor_id, actor_contract)
        patcher = WorkflowPatcher(read_json(self.identity_repair_workflow_path))
        patcher.set_input_by_title("#INPUT_IMAGE", "image", self._upload(source_image, "feverslop/startframe/director"))
        patcher.set_input_by_title("#IDENTITY_REFERENCE", "image", self._upload(reference, "feverslop/startframe/references"))
        patcher.set_input_by_title("#REGION_MASK_IMAGE", "image", self._upload(mask_image, "feverslop/startframe/masks"))
        patcher.set_existing_input_by_title_any("#PROMPT_POSITIVE", "text", prompt)
        patcher.set_existing_input_by_title_any(
            "#PROMPT_NEGATIVE",
            "text",
            "wrong person, changed outfit, changed hair, extra people, artifacts, split screen, contact sheet, multiple panels, reference sheet",
        )
        patcher.set_existing_input_by_title("#DENOISE", "denoise", 0.36)
        _patch_seed_inputs(patcher, 400000 + scene)
        patcher.set_input_by_title("#SAVE_IMAGE", "filename_prefix", f"startframe/repair/scene_{scene:04}_{actor_id}")
        return self._queue_and_download(
            patcher.get(),
            self.identity_repair_workflow_path,
            project_dir / "output" / "movie" / "startframes" / "repair" / f"scene_{scene:04}_{actor_id}.png",
            debug_name=f"scene_{scene:04}_repair_{actor_id}",
        )

    def _render_detail(self, *, project_dir: Path, scene: int, source_image: Path, shot: dict[str, Any]) -> Path:
        patcher = WorkflowPatcher(read_json(self.detail_workflow_path))
        patcher.set_input_by_title("#INPUT_IMAGE", "image", self._upload(source_image, "feverslop/startframe/repair"))
        try:
            patcher.set_existing_input_by_title("#SDXL_CHECKPOINT", "positive", str((shot.get("ltx_motion") or {}).get("prompt") or "cinematic startframe"))
            patcher.set_existing_input_by_title("#SDXL_CHECKPOINT", "negative", "changed identity, changed wardrobe, artifacts")
        except KeyError:
            pass
        _patch_seed_inputs(patcher, 500000 + scene)
        patcher.set_input_by_title("#SAVE_IMAGE", "filename_prefix", f"startframe/detail/scene_{scene:04}")
        return self._queue_and_download(
            patcher.get(),
            self.detail_workflow_path,
            project_dir / "output" / "movie" / "startframes" / "detail" / f"scene_{scene:04}.png",
            debug_name=f"scene_{scene:04}_detail",
        )

    def _queue_and_download(self, workflow: dict[str, Any], workflow_path: Path, output_path: Path, *, debug_name: str) -> Path:
        if self.model_resolver is not None:
            workflow = self.model_resolver.resolve_workflow_models(workflow, workflow_path=workflow_path)
        self._write_debug_workflow(debug_name, workflow)
        prompt_id = self.client.queue_prompt(workflow)
        history = self.client.wait_for_completion(prompt_id)
        images = self.client.extract_output_images(history)
        if not images:
            raise RuntimeError(f"No image output from workflow: {workflow_path}")
        first = images[0]
        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=output_path,
        )

    def _upload(self, image_path: Path, subfolder: str) -> str:
        upload = self.client.upload_image(image_path, subfolder=subfolder, file_type="input", overwrite=True)
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(upload)

    def _write_debug_workflow(self, debug_name: str, workflow: dict[str, Any]) -> None:
        if self.debug_workflows_dir is None:
            return
        self.debug_workflows_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in debug_name)
        (self.debug_workflows_dir / f"{safe_name}.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _patch_seed_inputs(patcher: WorkflowPatcher, seed: int) -> None:
    for node in patcher.get().values():
        inputs = node.setdefault("inputs", {})
        if "seed" in inputs:
            inputs["seed"] = seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = seed


def _patch_dimensions(patcher: WorkflowPatcher, *, width: int, height: int) -> None:
    try:
        patcher.set_existing_input_by_title("#WIDTH", "value", width)
        patcher.set_existing_input_by_title("#HEIGHT", "value", height)
        return
    except KeyError:
        pass
    patcher.set_existing_input_by_title("#DIMENSIONS", "width", width)
    patcher.set_existing_input_by_title("#DIMENSIONS", "height", height)


def _repair_prompt(actor_id: str, actor_contract: dict[str, Any]) -> str:
    wardrobe = (actor_contract.get("wardrobe") or {}).get("description") or ""
    face = (actor_contract.get("face") or {}).get("description") or ""
    return (
        f"Repair only actor {actor_id} inside the supplied mask. "
        f"Preserve pose, action, lighting, background, hands, and unmasked pixels. "
        f"Identity: {face}. Wardrobe must remain: {wardrobe}."
    )


def _mask_prompt(actor_id: str, actor_contract: dict[str, Any]) -> str:
    name = str(actor_contract.get("name") or actor_id)
    wardrobe = (actor_contract.get("wardrobe") or {}).get("description") or ""
    face = (actor_contract.get("face") or {}).get("description") or ""
    details = ", ".join(part for part in (face, wardrobe) if part)
    if details:
        return f"the visible full body, face, hair, and clothing of {name}: {details}"
    return f"the visible full body, face, hair, and clothing of {name}"


def _notify(callback, completed: int, total: int, scene: int, kind: str, actor_id: str = "") -> int:
    completed += 1
    if callback is not None:
        event = {"kind": kind, "completed": completed, "total": total, "scene": scene}
        if actor_id:
            event["actor_id"] = actor_id
        callback(event)
    return completed


def _validator_label(validator: object) -> str:
    model = getattr(validator, "model", "")
    base_url = getattr(validator, "base_url", "")
    if model or base_url:
        return f"{model}@{base_url}"
    return validator.__class__.__name__
