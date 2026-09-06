from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from feverslop.domain.artifact_hash import is_sha256_hex
from feverslop.domain.postprocessing import TrimSpec
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies
from feverslop.domain.prepared_workflow import (
    PreparedSceneWorkflow,
    SceneWorkflowManifest,
    StoredArtifact,
    sha256_file,
)
from feverslop.domain.scene_duration_limits import validate_render_frame_budget
from feverslop.domain.visual_consistency import SceneConsistencyContract
from feverslop.ports.workflow import WorkflowMaterializationRequest
from feverslop.scene_artifacts import SceneArtifactLayout


def _write_json_temp(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    return temporary


class WorkflowMaterializer:
    """Build and persist a render-ready workflow without submitting it."""

    def __init__(self, backend: Any, layout: SceneArtifactLayout):
        self.backend = backend
        self.layout = layout

    def prepare(self, request: WorkflowMaterializationRequest) -> PreparedSceneWorkflow:
        scene = request.scene
        scene_number = int(scene["scene"])
        consistency = self._consistency_contract(scene, scene_number)
        manifest_path = self.layout.scene_manifest(scene_number)
        if request.seed is not None:
            seed = int(request.seed)
        elif scene.get("seed") is not None and not getattr(self.backend, "randomize_seed", False):
            seed = int(scene["seed"])
        else:
            seed = int(self.backend._seed_for_scene(scene_number))
        original_uploader = self.backend.asset_uploader
        recording_uploader = _RecordingUploader(original_uploader, self.layout.project_dir)
        self.backend.asset_uploader = recording_uploader
        try:
            audio_name = None
            if request.audio_file is not None:
                audio_name = recording_uploader.resolve_audio_name(
                    request.audio_file, upload_audio=True, uploaded_audio_name=None,
                )
            rolling = self.backend._rolling_spec(scene)
            render_frame_count = _coerce_int(
                _rolling_value(rolling, "render_frame_count"),
                scene.get("render_frame_count", 0),
            )
            fps = _coerce_int(_rolling_value(rolling, "fps"), scene.get("fps", 0))
            validate_render_frame_budget(
                scene_number=scene_number,
                render_frame_count=render_frame_count,
                fps=fps,
                workflow_path=(
                    getattr(self.backend, "render_budget_workflow_path", None)
                    or self.backend.workflow_label
                ),
                max_render_frames=getattr(self.backend, "max_render_frames", None),
                max_render_duration_seconds=getattr(
                    self.backend, "max_render_duration_seconds", None,
                ),
                round_render_frames_to_8n1=bool(
                    getattr(self.backend, "round_render_frames_to_8n1", False),
                ),
            )
            old_offset = self.backend.seed_offset
            old_randomize = self.backend.randomize_seed
            try:
                self.backend.seed_offset = seed - scene_number
                self.backend.randomize_seed = False
                workflow = self.backend.build_workflow(
                    scene, prompt=request.prompt, comfy_audio_name=audio_name, rolling=rolling,
                )
            finally:
                self.backend.seed_offset = old_offset
                self.backend.randomize_seed = old_randomize
            workflow = self.backend.model_resolver.resolve_workflow_models(
                workflow, workflow_path=self.backend.workflow_label,
            )
        finally:
            self.backend.asset_uploader = original_uploader

        workflow_path = self.layout.scene_workflow(scene_number)
        temporary_workflow = _write_json_temp(workflow_path, workflow)
        temporary_manifest: Path | None = None
        try:
            assets = self._manifest_assets(
                scene,
                request.audio_file,
                recording_uploader.names,
                consistency,
            )
            keyframes = scene.get("keyframes") or {}
            source_clip_value = keyframes.get("startframe_source_clip_path")
            source_clip_path = (
                None
                if source_clip_value is None
                else self._project_path(source_clip_value)
            )
            manifest = SceneWorkflowManifest.create(
                project_dir=self.layout.project_dir,
                scene=scene_number,
                pipeline=request.pipeline,
                workflow_path=temporary_workflow,
                template_path=self.backend.workflow_label,
                render_plan_path=request.render_plan_path,
                assets=assets,
                seed=seed,
                fps=_coerce_int(scene.get("fps"), _rolling_value(rolling, "fps")),
                frame_count=_coerce_int(
                    scene.get("frame_count"), _rolling_value(rolling, "render_frame_count"),
                ),
                render_frame_count=_coerce_int(
                    _rolling_value(rolling, "render_frame_count"), scene.get("frame_count"),
                ),
                trim_front_frames=_coerce_int(_rolling_value(rolling, "trim_front_frames")),
                width=int(scene.get("width") or 0),
                height=int(scene.get("height") or 0),
                max_render_frames=getattr(self.backend, "max_render_frames", None),
                max_render_duration_seconds=getattr(
                    self.backend, "max_render_duration_seconds", None,
                ),
                render_budget_workflow_path=getattr(
                    self.backend, "render_budget_workflow_path", None,
                ),
                round_render_frames_to_8n1=bool(
                    getattr(self.backend, "round_render_frames_to_8n1", False),
                ),
                canonical_dependencies=request.canonical_dependencies,
                consistency=consistency,
                startframe_mode=keyframes.get("startframe_mode"),
                startframe_source_scene=keyframes.get("startframe_source_scene"),
                startframe_source_clip_path=source_clip_path,
                startframe_extractor=keyframes.get("startframe_extractor"),
                startframe_sha256=keyframes.get("startframe_sha256"),
            )
            provenance_mismatches = manifest.verify_consistency_provenance()
            claimed_source_clip_sha = str(
                keyframes.get("startframe_source_clip_sha256") or "",
            )
            requires_source_clip_claim = (
                consistency is not None
                and consistency.transition_from_previous == "continuous"
                and consistency.mode in {"msr", "i2v"}
            )
            if requires_source_clip_claim and not is_sha256_hex(claimed_source_clip_sha):
                provenance_mismatches.append(
                    "consistency: continuous handoff requires a valid "
                    "startframe source clip SHA-256 claim",
                )
            if (
                claimed_source_clip_sha
                and (
                    manifest.startframe_source_clip is None
                    or manifest.startframe_source_clip.sha256
                    != claimed_source_clip_sha
                )
            ):
                provenance_mismatches.append(
                    "consistency: startframe source clip SHA-256 metadata "
                    "does not match the predecessor clip",
                )
            if provenance_mismatches:
                raise ValueError(
                    "Prepared workflow consistency provenance failed: "
                    + "; ".join(provenance_mismatches),
                )
            workflow_artifact = StoredArtifact.from_path(
                workflow_path, project_dir=self.layout.project_dir,
            ) if workflow_path.is_file() else replace(
                manifest.workflow,
                path=workflow_path.resolve().relative_to(self.layout.project_dir.resolve()).as_posix(),
            )
            manifest = replace(
                manifest,
                workflow=replace(workflow_artifact, sha256=manifest.workflow.sha256),
            )
            temporary_manifest = manifest.write(
                _write_json_temp(manifest_path, manifest.to_dict()),
            )
            os.replace(temporary_workflow, workflow_path)
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_workflow.unlink(missing_ok=True)
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)
        return PreparedSceneWorkflow(scene_number, workflow_path.parent, workflow_path, manifest_path)

    def _manifest_assets(
        self,
        scene: dict[str, Any],
        audio_file: Path | None,
        uploaded_names: dict[Path, str],
        consistency: SceneConsistencyContract | None,
    ) -> list[tuple]:
        assets: list[tuple] = []
        if audio_file is not None:
            audio_path = self._project_path(audio_file).resolve()
            assets.append(("audio", audio_path, uploaded_names[audio_path]))

        references = scene.get("references") or {}
        candidates: list[tuple[str, str | Path, bool, str]] = []
        ingredients = (scene.get("ingredients") or {}).get("sheet_path") or scene.get("ingredients_scene_sheet")
        if ingredients:
            candidates.append(("ingredients_sheet", ingredients, False, ""))
        actor_paths = references.get("actor_msr_paths") or references.get("actor_sheet_paths") or []
        for index, path in enumerate(actor_paths):
            reference_id = (
                consistency.actors[index].id
                if consistency is not None and index < len(consistency.actors)
                else ""
            )
            candidates.append(("actor_sheet", path, False, reference_id))
        location = references.get("location_msr_path") or references.get("location_sheet_path")
        if location:
            candidates.append((
                "location_sheet",
                location,
                False,
                (
                    consistency.location.id
                    if consistency is not None and consistency.location is not None
                    else ""
                ),
            ))
        keyframes = scene.get("keyframes") or {}
        startframe = keyframes.get("startframe_path") or keyframes.get("start_frame_path")
        if startframe:
            candidates.append(("startframe", startframe, False, ""))
        consistency_sources = scene.get("visual_consistency_sources") or {}
        for source in consistency_sources.get("actors") or []:
            if isinstance(source, dict) and source.get("path"):
                candidates.append((
                    "actor_sheet",
                    source["path"],
                    True,
                    str(source.get("id") or ""),
                ))
        location_source = consistency_sources.get("location")
        if isinstance(location_source, dict) and location_source.get("path"):
            candidates.append((
                "location_sheet",
                location_source["path"],
                True,
                str(location_source.get("id") or ""),
            ))
        seen: set[tuple[str, Path, str]] = set()
        provenance_keys = {
            (role, self._project_path(value).resolve(), reference_id)
            for role, value, provenance, reference_id in candidates
            if provenance
        }
        for role, value, _provenance, reference_id in candidates:
            path = self._project_path(value).resolve()
            key = (role, path, reference_id)
            if key not in seen and (path in uploaded_names or key in provenance_keys):
                assets.append((
                    role,
                    path,
                    uploaded_names.get(path, ""),
                    reference_id,
                ))
                seen.add(key)
        ingredients_metadata = scene.get("ingredients") or {}
        expected_sheet_sha = str(ingredients_metadata.get("sheet_sha256") or "")
        if expected_sheet_sha:
            sheets = [
                self._project_path(value).resolve()
                for role, value, _provenance, _reference_id in candidates
                if role == "ingredients_sheet"
            ]
            if len(sheets) != 1 or sha256_file(sheets[0]) != expected_sheet_sha:
                raise ValueError(
                    "Ingredients sheet hash does not match runtime metadata",
                )
        return assets

    @staticmethod
    def _consistency_contract(
        scene: dict[str, Any], scene_number: int,
    ) -> SceneConsistencyContract | None:
        payload = scene.get("visual_consistency")
        if payload is None:
            return None
        try:
            contract = SceneConsistencyContract.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid visual consistency contract: {exc}") from exc
        if contract.scene != scene_number:
            raise ValueError(
                "visual consistency scene does not match materialized scene",
            )
        return contract

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.layout.project_dir / path


class PreparedWorkflowRenderer:
    """Verify stored JSON and adapt an in-memory copy for the current ComfyUI server."""

    def __init__(
        self, *, project_dir: str | Path, render_queue: Any, postprocessor: Any,
        expected_pipeline: str,
        expected_workflow_profile: str | None = None,
        max_render_frames: int | None = None,
        max_render_duration_seconds: float | None = None,
        render_budget_workflow_path: str | Path | None = None,
        round_render_frames_to_8n1: bool = False,
        asset_uploader: Any | None = None,
        model_resolver: Any | None = None,
        model_workflow_path: str | Path | None = None,
    ):
        self.project_dir = Path(project_dir)
        self.render_queue = render_queue
        self.postprocessor = postprocessor
        self.expected_pipeline = expected_pipeline
        self.expected_workflow_profile = expected_workflow_profile
        self.max_render_frames = max_render_frames
        self.max_render_duration_seconds = max_render_duration_seconds
        self.render_budget_workflow_path = render_budget_workflow_path
        self.round_render_frames_to_8n1 = bool(round_render_frames_to_8n1)
        self.asset_uploader = asset_uploader
        self.model_resolver = model_resolver
        self.model_workflow_path = model_workflow_path

    @staticmethod
    def verify_canonical_dependencies(
        prepared_workflow_path: str | Path,
        canonical_dependencies: CanonicalSceneDependencies,
    ) -> None:
        manifest = SceneWorkflowManifest.read(
            Path(prepared_workflow_path).with_name("manifest.json"),
        )
        dependency_mismatches = manifest.compare_canonical_dependencies(
            canonical_dependencies,
        )
        if dependency_mismatches:
            raise ValueError(
                "Stale prepared workflow from "
                f"{canonical_dependencies.source} for scene {manifest.scene}: "
                + "; ".join(dependency_mismatches)
                + ". Run --stage ltx_prepare_workflows first.",
            )

    def render(
        self,
        prepared_workflow_path: str | Path,
        *,
        canonical_dependencies: CanonicalSceneDependencies | None = None,
    ) -> Path:
        workflow_path = Path(prepared_workflow_path)
        manifest = SceneWorkflowManifest.read(workflow_path.with_name("manifest.json"))
        if manifest.pipeline != self.expected_pipeline:
            raise ValueError(
                f"Prepared workflow pipeline {manifest.pipeline!r} does not match "
                f"expected pipeline {self.expected_pipeline!r}",
            )
        if (
            manifest.consistency is not None
            and manifest.consistency.workflow_profile
            != self.expected_workflow_profile
        ):
            raise ValueError(
                "Prepared workflow profile "
                f"{manifest.consistency.workflow_profile!r} does not match "
                f"active workflow profile {self.expected_workflow_profile!r}",
            )
        manifest_workflow_path = manifest.workflow.resolve(self.project_dir).resolve()
        if workflow_path.resolve() != manifest_workflow_path:
            raise ValueError(
                f"Prepared path {workflow_path} does not match manifest workflow {manifest_workflow_path}",
            )
        if canonical_dependencies is not None:
            self.verify_canonical_dependencies(
                workflow_path,
                canonical_dependencies,
            )
        mismatches = manifest.verify(self.project_dir)
        if mismatches:
            raise ValueError("Prepared workflow verification failed: " + "; ".join(mismatches))
        validate_render_frame_budget(
            scene_number=manifest.scene,
            render_frame_count=manifest.render_frame_count,
            fps=manifest.fps,
            workflow_path=(
                manifest.render_budget_workflow_path or manifest.template.path
            ),
            max_render_frames=manifest.max_render_frames,
            max_render_duration_seconds=manifest.max_render_duration_seconds,
            round_render_frames_to_8n1=manifest.round_render_frames_to_8n1,
        )
        validate_render_frame_budget(
            scene_number=manifest.scene,
            render_frame_count=manifest.render_frame_count,
            fps=manifest.fps,
            workflow_path=(
                self.render_budget_workflow_path
                or manifest.render_budget_workflow_path
                or manifest.template.path
            ),
            max_render_frames=self.max_render_frames,
            max_render_duration_seconds=self.max_render_duration_seconds,
            round_render_frames_to_8n1=self.round_render_frames_to_8n1,
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
        workflow = self._prepare_for_current_server(workflow, manifest)
        layout = SceneArtifactLayout(self.project_dir)
        raw_path = layout.scene_raw_video(manifest.scene)
        final_path = layout.scene_final_video(manifest.scene)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=raw_path.parent, suffix=".mp4", delete=False) as handle:
            temporary_raw = Path(handle.name)
        try:
            downloaded = self.render_queue.queue_workflow_and_download_first_video(
                workflow, scene_number=manifest.scene, output_path=temporary_raw,
            )
            os.replace(downloaded, raw_path)
        finally:
            temporary_raw.unlink(missing_ok=True)
        with NamedTemporaryFile(dir=final_path.parent, suffix=".mp4", delete=False) as handle:
            temporary_final = Path(handle.name)
        try:
            self.postprocessor.trim_clip(TrimSpec(
                source_file=raw_path,
                output_file=temporary_final,
                fps=manifest.fps,
                trim_front_frames=manifest.trim_front_frames,
                keep_frames=manifest.frame_count,
                scene=manifest.scene,
                extract_boundary_frames=True,
            ))
            os.replace(temporary_final, final_path)
        finally:
            temporary_final.unlink(missing_ok=True)
        manifest_path = final_path.with_name("manifest.json")
        manifest = SceneWorkflowManifest.read(manifest_path)
        first_frame_path = layout.scene_dir(manifest.scene) / "firstframe.png"
        last_frame_path = layout.scene_dir(manifest.scene) / "lastframe.png"
        if not first_frame_path.is_file() or not last_frame_path.is_file():
            return final_path
        manifest = replace(
            manifest,
            first_frame_path=StoredArtifact.from_path(
                first_frame_path,
                project_dir=self.project_dir,
            ),
            last_frame_path=StoredArtifact.from_path(
                last_frame_path,
                project_dir=self.project_dir,
            ),
        )
        manifest.write(manifest_path)
        return final_path

    def _prepare_for_current_server(
        self, workflow: dict[str, Any], manifest: SceneWorkflowManifest,
    ) -> dict[str, Any]:
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "MinimaxH3LatentUpscaler3D":
                continue
            inputs = node.get("inputs") or {}
            if "mode.scale" not in inputs and "scale" in inputs:
                inputs["mode.scale"] = inputs.pop("scale")
        replacements: dict[str, str] = {}
        if self.asset_uploader is not None:
            for asset in manifest.assets:
                stored_name = asset.comfyui_name
                if not stored_name:
                    continue
                if stored_name in replacements:
                    continue
                normalized_name = stored_name.replace("\\", "/")
                if self.asset_uploader.client.input_file_exists(normalized_name):
                    current_name = normalized_name
                else:
                    local_path = asset.resolve(self.project_dir)
                    if asset.role == "audio":
                        current_name = self.asset_uploader.resolve_audio_name(
                            local_path,
                            upload_audio=True,
                            uploaded_audio_name=None,
                        )
                    elif asset.role == "startframe":
                        current_name = self.asset_uploader.resolve_startframe_name(
                            local_path,
                            upload_startframes=True,
                        )
                    else:
                        current_name = self.asset_uploader.resolve_reference_image_name(
                            local_path,
                            upload_references=True,
                        )
                replacements[stored_name] = current_name.replace("\\", "/")
        if replacements:
            workflow = _replace_asset_names_in_workflow(workflow, replacements)
        if self.model_resolver is not None:
            workflow = self.model_resolver.resolve_workflow_models(
                workflow,
                workflow_path=(self.model_workflow_path or manifest.template.path),
            )
        return workflow


def _rolling_value(rolling: Any, key: str) -> Any:
    if isinstance(rolling, dict):
        return rolling.get(key)
    return getattr(rolling, key, None)


def _coerce_int(value: Any, fallback: Any = 0) -> int:
    """Convert numeric rolling metadata while tolerating loose test doubles."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(fallback)
        except (TypeError, ValueError):
            return 0


def _replace_asset_names_in_workflow(
    workflow: dict[str, Any],
    replacements: dict[str, str],
) -> dict[str, Any]:
    """Replace asset file names only in ComfyUI node input values.

    Only leaf string values inside each node's ``"inputs"`` dict are
    candidates for replacement.  Structural fields (``class_type``,
    ``_meta``) and wire references (lists like ``["node_id", 0]``) are
    left untouched.
    """
    result: dict[str, Any] = {}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            result[node_id] = node
            continue
        new_node: dict[str, Any] = {}
        for key, value in node.items():
            if key == "inputs" and isinstance(value, dict):
                new_node[key] = _replace_in_inputs(value, replacements)
            else:
                new_node[key] = deepcopy(value)
        result[node_id] = new_node
    return result


def _replace_in_inputs(
    inputs: dict[str, Any],
    replacements: dict[str, str],
) -> dict[str, Any]:
    """Recursively replace asset names inside node input values."""
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        result[key] = _replace_value(value, replacements)
    return result


def _replace_value(value: Any, replacements: dict[str, str]) -> Any:
    """Replace asset name in a single input value, skipping wire refs."""
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        # Skip ComfyUI wire references [node_id, output_index]
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
            return list(value)
        return [_replace_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {k: _replace_value(v, replacements) for k, v in value.items()}
    return value


class _RecordingUploader:
    def __init__(self, delegate: Any, project_dir: Path):
        self.delegate = delegate
        self.project_dir = project_dir
        self.names: dict[Path, str] = {}

    def resolve_audio_name(self, path: str | Path, **kwargs: Any) -> str:
        name = self.delegate.resolve_audio_name(path, **kwargs)
        self.names[self._path(path)] = name
        return name

    def resolve_reference_image_name(self, path: str | Path, **kwargs: Any) -> str:
        name = self.delegate.resolve_reference_image_name(path, **kwargs)
        self.names[self._path(path)] = name
        return name

    def resolve_startframe_name(self, path: str | Path, **kwargs: Any) -> str:
        name = self.delegate.resolve_startframe_name(path, **kwargs)
        self.names[self._path(path)] = name
        return name

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.project_dir / path).resolve()
