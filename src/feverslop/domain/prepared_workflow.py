from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


SCHEMA = "feverslop.scene-workflow/v1"


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, **super().to_dict(), "comfyui_name": self.comfyui_name}

    @classmethod
    def create(
        cls,
        role: str,
        path: str | Path,
        comfyui_name: str,
        *,
        project_dir: str | Path,
    ) -> ManifestAsset:
        artifact = StoredArtifact.from_path(path, project_dir=project_dir)
        return cls(**asdict(artifact), role=role, comfyui_name=comfyui_name)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestAsset:
        return cls(
            path=str(payload["path"]), sha256=str(payload["sha256"]),
            external=bool(payload.get("external", False)), role=str(payload["role"]),
            comfyui_name=str(payload["comfyui_name"]),
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
    max_render_frames: int | None = None
    max_render_duration_seconds: float | None = None
    render_budget_workflow_path: str | None = None
    round_render_frames_to_8n1: bool = False

    @classmethod
    def create(
        cls, *, project_dir: str | Path, scene: int, pipeline: str,
        workflow_path: str | Path, template_path: str | Path,
        render_plan_path: str | Path,
        assets: list[tuple[str, str | Path, str]], seed: int, fps: int,
        frame_count: int, width: int, height: int,
        render_frame_count: int | None = None, trim_front_frames: int = 0,
        max_render_frames: int | None = None,
        max_render_duration_seconds: float | None = None,
        render_budget_workflow_path: str | Path | None = None,
        round_render_frames_to_8n1: bool = False,
    ) -> SceneWorkflowManifest:
        return cls(
            schema=SCHEMA,
            scene=int(scene),
            pipeline=str(pipeline),
            workflow=StoredArtifact.from_path(workflow_path, project_dir=project_dir),
            template=StoredArtifact.from_path(template_path, project_dir=project_dir, allow_external=True),
            render_plan=StoredArtifact.from_path(render_plan_path, project_dir=project_dir),
            assets=tuple(
                ManifestAsset.create(role, path, comfyui_name, project_dir=project_dir)
                for role, path, comfyui_name in assets
            ),
            seed=int(seed), fps=int(fps), frame_count=int(frame_count),
            render_frame_count=int(render_frame_count if render_frame_count is not None else frame_count),
            trim_front_frames=int(trim_front_frames),
            width=int(width), height=int(height),
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
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported scene workflow schema: {payload.get('schema')}")
        if "assets" not in payload or not isinstance(payload["assets"], list):
            raise ValueError("Scene workflow manifest requires an assets list")
        roles = {str(item.get("role", "")) for item in payload["assets"]}
        required_roles = {
            "ltx_ingredients": {"ingredients_sheet"},
            "ltx_msr": {"actor_sheet", "location_sheet"},
        }.get(str(payload.get("pipeline")), set())
        missing_roles = sorted(required_roles - roles)
        if missing_roles:
            raise ValueError(f"Scene workflow manifest is missing required asset roles: {', '.join(missing_roles)}")
        return cls(
            schema=payload["schema"], scene=int(payload["scene"]), pipeline=str(payload["pipeline"]),
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
        for label, artifact in artifacts:
            path = artifact.resolve(project_dir)
            if not path.is_file():
                mismatches.append(f"{label}: missing {path}")
            elif sha256_file(path) != artifact.sha256:
                mismatches.append(f"{label}: sha256 mismatch for {path}")
        return mismatches
