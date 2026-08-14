"""Workflow-independent guided global asset generation service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
import uuid
from typing import Any, Callable

from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset

_os = __import__("os")


@dataclass(frozen=True, slots=True)
class AssetIdea:
    kind: str
    asset_id: str
    name: str
    visual_concept: str
    identity_constraints: tuple[str, ...] = ()
    looks: tuple[dict[str, Any], ...] = ()
    intended_use: str = ""
    negative_constraints: tuple[str, ...] = ()
    output_requirements: tuple[str, ...] = ("hero_image",)

    def __post_init__(self) -> None:
        try:
            AssetKind(self.kind.strip().lower())
        except (AttributeError, ValueError) as exc:
            raise ValueError("kind must be character, location, style, or prop") from exc
        for field_name in ("asset_id", "name", "visual_concept"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.strip().lower()
        payload["asset_id"] = self.asset_id.strip()
        payload["name"] = self.name.strip()
        payload["visual_concept"] = self.visual_concept.strip()
        return payload


@dataclass(frozen=True, slots=True)
class GeneratedAssetRun:
    run_id: str
    status: str
    workflow_profile: str
    asset: GlobalAsset


class GuidedAssetGenerator:
    def __init__(self, library: Any, *, profiles: dict[str, Callable[..., dict[str, Any]]], runs_root: Any = None):
        self.library = library
        self.profiles = dict(profiles)
        self.runs_root = _os.fspath(runs_root) if runs_root is not None else _os.path.join(_os.fspath(library.root), "runs")

    def preview(self, idea: AssetIdea, *, profile_id: str) -> dict[str, Any]:
        if profile_id not in self.profiles:
            raise ValueError(f"unknown workflow profile '{profile_id}'; choose one of: {', '.join(sorted(self.profiles))}")
        return {"request": idea.to_dict(), "asset_id": idea.asset_id.strip(), "workflow_profile": profile_id}

    def generate(self, idea: AssetIdea, *, profile_id: str) -> GeneratedAssetRun:
        preview = self.preview(idea, profile_id=profile_id)
        run_id = uuid.uuid4().hex
        run_dir = _os.path.join(self.runs_root, run_id)
        _os.makedirs(run_dir, exist_ok=False)
        state = {"run_id": run_id, "status": "running", "workflow_profile": profile_id, "request": preview["request"]}
        self._write_state(run_dir, state)
        try:
            output = self.profiles[profile_id](request=preview["request"], run_dir=run_dir)
            if not isinstance(output, dict):
                raise ValueError("workflow profile must return an object")
            media = {}
            for field_name in idea.output_requirements:
                relative = output.get(field_name)
                if not isinstance(relative, str) or not relative:
                    raise ValueError(f"workflow output is missing required field '{field_name}'")
                source = _os.path.realpath(_os.path.join(run_dir, relative))
                if not source.startswith(_os.path.realpath(run_dir) + _os.sep) or not _os.path.isfile(source):
                    raise ValueError(f"workflow output '{relative}' is missing or escapes the run directory")
                media[field_name] = source
            asset_dir = _os.path.join(_os.fspath(self.library.root), AssetKind(idea.kind.strip().lower()).value, idea.asset_id.strip())
            _os.makedirs(asset_dir, exist_ok=False)
            for field_name, source in media.items():
                target = _os.path.join(asset_dir, _os.path.basename(source))
                shutil.copy2(source, target)
            look = AssetLook("default", "Default", description=idea.visual_concept, hero_image=_os.path.basename(media.get("hero_image", "")))
            asset = GlobalAsset(
                id=idea.asset_id.strip(), kind=AssetKind(idea.kind.strip().lower()), name=idea.name.strip(),
                description=idea.visual_concept.strip(), looks=(look,),
                metadata=(("generator_run_id", run_id), ("workflow_profile", profile_id)),
            )
            self.library.create(asset)
            state.update({"status": "completed", "asset": asset.to_dict()})
            self._write_state(run_dir, state)
            return GeneratedAssetRun(run_id, "completed", profile_id, asset)
        except Exception as exc:
            state.update({"status": "failed", "error": str(exc)})
            self._write_state(run_dir, state)
            raise

    def resume(self, run_id: str) -> GeneratedAssetRun:
        run_dir = _os.path.join(self.runs_root, run_id)
        with open(_os.path.join(run_dir, "run.json"), encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("status") != "completed":
            raise ValueError(f"run '{run_id}' is {state.get('status', 'unknown')}; resume requires a completed run")
        return GeneratedAssetRun(run_id, state["status"], state["workflow_profile"], GlobalAsset.from_dict(state["asset"]))

    @staticmethod
    def _write_state(run_dir: str, state: dict[str, Any]) -> None:
        with open(_os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
