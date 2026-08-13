from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from feverslop.domain.artifact_hash import sha256_file
from feverslop.domain.visual_consistency import SceneConsistencyContract


SCHEMA_V1 = "feverslop.scene-workflow/v1"
SCHEMA_V2 = "feverslop.scene-workflow/v2"
SCHEMA = SCHEMA_V2


@dataclass(frozen=True)
class StoredArtifact:
    path: str
    sha256: str
    external: bool = False

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        project_dir: str | Path,
        allow_external: bool = False,
    ) -> StoredArtifact:
        resolved = Path(path).resolve()
        project = Path(project_dir).resolve()
        try:
            stored_path = resolved.relative_to(project).as_posix()
            external = False
        except ValueError:
            if not allow_external:
                raise ValueError(f"Artifact path is outside project: {resolved}") from None
            stored_path = str(resolved)
            external = True
        return cls(path=stored_path, sha256=sha256_file(resolved), external=external)

    def resolve(self, project_dir: str | Path) -> Path:
        return Path(self.path) if self.external else Path(project_dir) / self.path

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.external:
            payload["external"] = True
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StoredArtifact:
        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            external=bool(payload.get("external", False)),
        )


@dataclass(frozen=True)
class ManifestAsset(StoredArtifact):
    role: str = ""
    comfyui_name: str = ""
    reference_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "role": self.role,
            **super().to_dict(),
            "comfyui_name": self.comfyui_name,
        }
        if self.reference_id:
            payload["reference_id"] = self.reference_id
        return payload

    @classmethod
    def create(
        cls,
        role: str,
        path: str | Path,
        comfyui_name: str,
        *,
        project_dir: str | Path,
        reference_id: str = "",
    ) -> ManifestAsset:
        artifact = StoredArtifact.from_path(path, project_dir=project_dir)
        return cls(
            **asdict(artifact),
            role=role,
            comfyui_name=comfyui_name,
            reference_id=reference_id,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestAsset:
        return cls(
            path=str(payload["path"]), sha256=str(payload["sha256"]),
            external=bool(payload.get("external", False)), role=str(payload["role"]),
            comfyui_name=str(payload["comfyui_name"]),
            reference_id=str(payload.get("reference_id") or ""),
        )


@dataclass(frozen=True)
class PreparedSceneWorkflow:
    scene: int
    scene_dir: Path
    workflow_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SceneWorkflowManifest:
    schema: str
    scene: int
    pipeline: str
    workflow: StoredArtifact
    template: StoredArtifact
    render_plan: StoredArtifact
    assets: tuple[ManifestAsset, ...]
    seed: int
    fps: int
    frame_count: int
    render_frame_count: int
    trim_front_frames: int
    width: int
    height: int
    consistency: SceneConsistencyContract | None = None
    startframe_mode: str | None = None
    startframe_source_scene: int | None = None
    startframe_source_clip: StoredArtifact | None = None
    startframe_extractor: str | None = None
    startframe_sha256: str | None = None
    first_frame_path: StoredArtifact | None = None
    last_frame_path: StoredArtifact | None = None
    max_render_frames: int | None = None
    max_render_duration_seconds: float | None = None
    render_budget_workflow_path: str | None = None
    round_render_frames_to_8n1: bool = False

    @classmethod
    def create(
        cls, *, project_dir: str | Path, scene: int, pipeline: str,
        workflow_path: str | Path, template_path: str | Path,
        render_plan_path: str | Path,
        assets: list[tuple], seed: int, fps: int,
        frame_count: int, width: int, height: int,
        render_frame_count: int | None = None, trim_front_frames: int = 0,
        max_render_frames: int | None = None,
        max_render_duration_seconds: float | None = None,
        render_budget_workflow_path: str | Path | None = None,
        round_render_frames_to_8n1: bool = False,
        consistency: SceneConsistencyContract | None = None,
        startframe_mode: str | None = None,
        startframe_source_scene: int | None = None,
        startframe_source_clip_path: str | Path | None = None,
        startframe_extractor: str | None = None,
        startframe_sha256: str | None = None,
        first_frame_path: str | Path | None = None,
        last_frame_path: str | Path | None = None,
    ) -> SceneWorkflowManifest:
        return cls(
            schema=SCHEMA_V2,
            scene=int(scene),
            pipeline=str(pipeline),
            workflow=StoredArtifact.from_path(workflow_path, project_dir=project_dir),
            template=StoredArtifact.from_path(template_path, project_dir=project_dir, allow_external=True),
            render_plan=StoredArtifact.from_path(render_plan_path, project_dir=project_dir),
            assets=tuple(
                ManifestAsset.create(
                    item[0],
                    item[1],
                    item[2],
                    project_dir=project_dir,
                    reference_id=(str(item[3]) if len(item) > 3 else ""),
                )
                for item in assets
            ),
            seed=int(seed), fps=int(fps), frame_count=int(frame_count),
            render_frame_count=int(render_frame_count if render_frame_count is not None else frame_count),
            trim_front_frames=int(trim_front_frames),
            width=int(width), height=int(height),
            consistency=consistency,
            startframe_mode=(
                None if startframe_mode is None else str(startframe_mode)
            ),
            startframe_source_scene=(
                None
                if startframe_source_scene is None
                else int(startframe_source_scene)
            ),
            startframe_source_clip=(
                None
                if startframe_source_clip_path is None
                else StoredArtifact.from_path(
                    startframe_source_clip_path,
                    project_dir=project_dir,
                )
            ),
            startframe_extractor=(
                None if startframe_extractor is None else str(startframe_extractor)
            ),
            startframe_sha256=(
                None if startframe_sha256 is None else str(startframe_sha256)
            ),
            first_frame_path=(
                None if first_frame_path is None else StoredArtifact.from_path(first_frame_path, project_dir=project_dir)
            ),
            last_frame_path=(
                None if last_frame_path is None else StoredArtifact.from_path(last_frame_path, project_dir=project_dir)
            ),
            max_render_frames=(None if max_render_frames is None else int(max_render_frames)),
            max_render_duration_seconds=(
                None
                if max_render_duration_seconds is None
                else float(max_render_duration_seconds)
            ),
            render_budget_workflow_path=(
                None
                if render_budget_workflow_path is None
                else str(render_budget_workflow_path)
            ),
            round_render_frames_to_8n1=bool(round_render_frames_to_8n1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "scene": self.scene, "pipeline": self.pipeline,
            "workflow": self.workflow.to_dict(), "template": self.template.to_dict(),
            "render_plan": self.render_plan.to_dict(),
            "assets": [asset.to_dict() for asset in self.assets],
            "seed": self.seed, "fps": self.fps, "frame_count": self.frame_count,
            "render_frame_count": self.render_frame_count,
            "trim_front_frames": self.trim_front_frames,
            "width": self.width, "height": self.height,
            "consistency": (
                None if self.consistency is None else self.consistency.to_dict()
            ),
            "startframe_mode": self.startframe_mode,
            "startframe_source_scene": self.startframe_source_scene,
            "startframe_source_clip": (
                None
                if self.startframe_source_clip is None
                else self.startframe_source_clip.to_dict()
            ),
            "startframe_extractor": self.startframe_extractor,
            "startframe_sha256": self.startframe_sha256,
            "first_frame_path": None if self.first_frame_path is None else self.first_frame_path.to_dict(),
            "last_frame_path": None if self.last_frame_path is None else self.last_frame_path.to_dict(),
            "max_render_frames": self.max_render_frames,
            "max_render_duration_seconds": self.max_render_duration_seconds,
            "render_budget_workflow_path": self.render_budget_workflow_path,
            "round_render_frames_to_8n1": self.round_render_frames_to_8n1,
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return destination

    @classmethod
    def read(cls, path: str | Path) -> SceneWorkflowManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema not in {SCHEMA_V1, SCHEMA_V2}:
            raise ValueError(f"Unsupported scene workflow schema: {payload.get('schema')}")
        if "assets" not in payload or not isinstance(payload["assets"], list):
            raise ValueError("Scene workflow manifest requires an assets list")
        consistency = (
            None
            if schema == SCHEMA_V1 or payload.get("consistency") is None
            else SceneConsistencyContract.from_dict(payload["consistency"])
        )
        roles = {str(item.get("role", "")) for item in payload["assets"]}
        required_roles = (
            {
                "ltx_ingredients": {"ingredients_sheet"},
                "ltx_msr": {"actor_sheet", "location_sheet"},
            }.get(str(payload.get("pipeline")), set())
            if consistency is None
            else set()
        )
        missing_roles = sorted(required_roles - roles)
        if missing_roles:
            raise ValueError(f"Scene workflow manifest is missing required asset roles: {', '.join(missing_roles)}")
        return cls(
            schema=schema, scene=int(payload["scene"]), pipeline=str(payload["pipeline"]),
            workflow=StoredArtifact.from_dict(payload["workflow"]),
            template=StoredArtifact.from_dict(payload["template"]),
            render_plan=StoredArtifact.from_dict(payload["render_plan"]),
            assets=tuple(ManifestAsset.from_dict(item) for item in payload["assets"]),
            seed=int(payload["seed"]), fps=int(payload["fps"]),
            frame_count=int(payload["frame_count"]),
            render_frame_count=int(payload.get("render_frame_count", payload["frame_count"])),
            trim_front_frames=int(payload.get("trim_front_frames", 0)),
            width=int(payload["width"]),
            height=int(payload["height"]),
            consistency=consistency,
            startframe_mode=(
                None
                if schema == SCHEMA_V1 or payload.get("startframe_mode") is None
                else str(payload["startframe_mode"])
            ),
            startframe_source_scene=(
                None
                if schema == SCHEMA_V1
                or payload.get("startframe_source_scene") is None
                else int(payload["startframe_source_scene"])
            ),
            startframe_source_clip=(
                None
                if schema == SCHEMA_V1
                or payload.get("startframe_source_clip") is None
                else StoredArtifact.from_dict(payload["startframe_source_clip"])
            ),
            startframe_extractor=(
                None
                if schema == SCHEMA_V1
                or payload.get("startframe_extractor") is None
                else str(payload["startframe_extractor"])
            ),
            startframe_sha256=(
                None
                if schema == SCHEMA_V1
                or payload.get("startframe_sha256") is None
                else str(payload["startframe_sha256"])
            ),
            first_frame_path=(
                None if payload.get("first_frame_path") is None else StoredArtifact.from_dict(payload["first_frame_path"])
            ),
            last_frame_path=(
                None if payload.get("last_frame_path") is None else StoredArtifact.from_dict(payload["last_frame_path"])
            ),
            max_render_frames=(
                None
                if payload.get("max_render_frames") is None
                else int(payload["max_render_frames"])
            ),
            max_render_duration_seconds=(
                None
                if payload.get("max_render_duration_seconds") is None
                else float(payload["max_render_duration_seconds"])
            ),
            render_budget_workflow_path=payload.get("render_budget_workflow_path"),
            round_render_frames_to_8n1=bool(payload.get("round_render_frames_to_8n1", False)),
        )

    def verify(self, project_dir: str | Path) -> list[str]:
        mismatches: list[str] = []
        artifacts = [("workflow", self.workflow), ("template", self.template), ("render_plan", self.render_plan)]
        artifacts.extend((f"asset[{asset.role}]", asset) for asset in self.assets)
        if self.startframe_source_clip is not None:
            artifacts.append(("startframe source clip", self.startframe_source_clip))
        if self.first_frame_path is not None:
            artifacts.append(("first frame", self.first_frame_path))
        if self.last_frame_path is not None:
            artifacts.append(("last frame", self.last_frame_path))
        for label, artifact in artifacts:
            path = artifact.resolve(project_dir)
            if not path.is_file():
                mismatches.append(f"{label}: missing {path}")
            elif sha256_file(path) != artifact.sha256:
                mismatches.append(f"{label}: sha256 mismatch for {path}")
        mismatches.extend(self.verify_consistency_provenance())
        return mismatches

    def verify_consistency_provenance(self) -> list[str]:
        contract = self.consistency
        if contract is None:
            return []
        mismatches: list[str] = []
        if contract.scene != self.scene:
            mismatches.append(
                "consistency: contract scene does not match manifest scene"
            )
        expected_mode = {
            "ltx_ingredients": "ingredients",
            "ltx_msr": "msr",
            "ltx_i2v": "i2v",
        }.get(self.pipeline)
        if expected_mode is not None and contract.mode != expected_mode:
            mismatches.append(
                "consistency: contract mode does not match manifest pipeline"
            )

        by_role: dict[str, list[ManifestAsset]] = {}
        for asset in self.assets:
            by_role.setdefault(asset.role, []).append(asset)

        actor_bindings = [
            (asset.reference_id, asset.sha256)
            for asset in by_role.get("actor_sheet", [])
        ]
        expected_actor_bindings = [
            (anchor.id, anchor.asset_sha256) for anchor in contract.actors
        ]
        if Counter(actor_bindings) != Counter(expected_actor_bindings):
            mismatches.append(
                "consistency: actor_sheet IDs, roles, or SHA-256 values do not match contract"
            )

        location_bindings = [
            (asset.reference_id, asset.sha256)
            for asset in by_role.get("location_sheet", [])
        ]
        expected_location_bindings = (
            []
            if contract.location is None
            else [(contract.location.id, contract.location.asset_sha256)]
        )
        if location_bindings != expected_location_bindings:
            mismatches.append(
                "consistency: location_sheet ID, role, or SHA-256 does not match contract"
            )

        if contract.mode == "ingredients" and len(by_role.get("ingredients_sheet", [])) != 1:
            mismatches.append(
                "consistency: exactly one ingredients_sheet SHA-256 is required"
            )
        if (
            contract.transition_from_previous == "continuous"
            and contract.mode in {"msr", "i2v"}
            and len(by_role.get("startframe", [])) != 1
        ):
            mismatches.append(
                "consistency: continuous handoff requires exactly one startframe asset"
            )
        startframes = by_role.get("startframe", [])
        if (
            contract.transition_from_previous == "continuous"
            and contract.mode in {"msr", "i2v"}
            and (
                len(startframes) != 1
                or self.startframe_sha256 is None
                or startframes[0].sha256 != self.startframe_sha256
            )
        ):
            mismatches.append(
                "consistency: startframe SHA-256 does not match extracted frame"
            )
        if (
            contract.transition_from_previous == "continuous"
            and contract.mode in {"msr", "i2v"}
            and (
                self.startframe_mode != "last_frame_from_previous"
                or self.startframe_source_scene != contract.scene - 1
                or self.startframe_extractor != "last-frame-v1"
                or self.startframe_source_clip is None
            )
        ):
            mismatches.append(
                "consistency: continuous handoff startframe lineage is invalid"
            )
        if (
            contract.transition_from_previous == "continuous"
            and contract.mode in {"msr", "i2v"}
            and self.startframe_source_clip is not None
            and self.startframe_source_clip.path
            != (
                f"output/render/scenes/"
                f"scene_{contract.scene - 1:04d}/final.mp4"
            )
        ):
            mismatches.append(
                "consistency: continuous handoff source clip path is invalid"
            )
        return mismatches
