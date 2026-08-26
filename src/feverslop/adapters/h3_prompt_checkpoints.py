from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.domain.canonical_render_plan import (
    PromptRole,
    stable_scene_id,
    validate_canonical_plan,
)
from feverslop.domain.h3_prompt_checkpoint import (
    H3_CHECKPOINT_SCHEMA,
    H3PromptCheckpoint,
    H3PromptCheckpointInput,
    checkpoint_status,
)
from feverslop.errors import FeverSlopDataError
from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.utils.io import atomic_write_json, read_json_document

if TYPE_CHECKING:
    from feverslop.ports.reporting import Reporter


class H3PromptCheckpointStore:
    def __init__(self, project_dir: str | Path, *, reporter: Reporter | None = None) -> None:
        self.project_dir = Path(project_dir)
        self.layout = SceneArtifactLayout(self.project_dir)
        self._asset_hashes: dict[Path, tuple[tuple[int, int] | None, dict[str, Any]]] = {}
        self.reporter = reporter

    def load(self, request: H3PromptCheckpointInput) -> H3PromptCheckpoint | None:
        path = self.layout.scene_h3_prompt(request.scene_number)
        if not path.is_file():
            return None
        value = read_json_document(path)
        checkpoint = self._parse(path, value)
        expected_scene_id = stable_scene_id(request.segment_id)
        if (
            checkpoint.scene_number != request.scene_number
            or checkpoint.scene_id != expected_scene_id
            or checkpoint.segment_id != request.segment_id
            or checkpoint.input_fingerprint != self._fingerprint(request)
        ):
            return None
        self._sync_canonical(checkpoint)
        self._report("reused", checkpoint)
        return checkpoint

    def save(
        self,
        request: H3PromptCheckpointInput,
        generated: dict[str, Any],
    ) -> H3PromptCheckpoint:
        scene_id = stable_scene_id(request.segment_id)
        fingerprint = self._fingerprint(request)
        status = checkpoint_status(generated)
        path = self.layout.scene_h3_prompt(request.scene_number)
        payload = {
            "schema": H3_CHECKPOINT_SCHEMA,
            "scene": request.scene_number,
            "scene_id": scene_id,
            "segment_id": request.segment_id,
            "status": status,
            "input_fingerprint": fingerprint,
            "generated": deepcopy(generated),
            "judge": deepcopy(generated.get("prompt_judge")),
            "judge_attempts": deepcopy(generated.get("prompt_judge_attempts") or []),
            "provenance": {
                "source": "dspy-h3-prompt-builder",
                "generator": _json_value(request.generator_revision),
                **_structured_provenance(generated),
            },
        }
        atomic_write_json(path, payload)
        checkpoint = H3PromptCheckpoint(
            path,
            request.scene_number,
            scene_id,
            request.segment_id,
            status,
            fingerprint,
            deepcopy(generated),
        )
        self._sync_canonical(checkpoint)
        self._report("generated", checkpoint)
        return checkpoint

    def _report(self, action: str, checkpoint: H3PromptCheckpoint) -> None:
        if self.reporter is None:
            return
        verdict = {
            "good": "GOOD",
            "bad_exhausted": "BAD",
            "unjudged": "UNJUDGED",
        }[checkpoint.status]
        self.reporter.message(
            f"H3 prompt checkpoint {action}: scene {checkpoint.scene_number}, "
            f"judge {verdict}, status {checkpoint.status}, path {checkpoint.path}",
        )

    def _sync_canonical(self, checkpoint: H3PromptCheckpoint) -> None:
        store = CanonicalPlanStore(self.project_dir)
        snapshot = store.capture_regeneration()
        if not snapshot.exists:
            return
        scenes = deepcopy(list(snapshot.scenes))
        validate_canonical_plan(scenes)
        for scene in scenes:
            canonical = scene.get("canonical")
            if not isinstance(canonical, dict) or canonical.get("scene_id") != checkpoint.scene_id:
                continue
            if canonical.get("segment_id") != checkpoint.segment_id:
                raise FeverSlopDataError(
                    "H3 checkpoint canonical identity conflict: matching scene_id has a different segment_id",
                )
            roles = canonical.get("roles")
            if not isinstance(roles, dict):
                raise FeverSlopDataError("H3 checkpoint canonical roles must be an object")
            role = roles.setdefault(PromptRole.H3_VIDEO, {})
            if not isinstance(role, dict):
                raise FeverSlopDataError("H3 checkpoint canonical H3 role must be an object")
            generated = {
                "value": checkpoint.generated["prompt"],
                "provenance": {
                    "source": "h3-scene-checkpoint",
                    "input_fingerprint": checkpoint.input_fingerprint,
                },
            }
            if role.get("generated") == generated:
                return
            role["generated"] = generated
            store.commit_regeneration(snapshot, scenes)
            return

    def _fingerprint(self, request: H3PromptCheckpointInput) -> str:
        material = {
            "scene": request.scene_number,
            "scene_id": stable_scene_id(request.segment_id),
            "segment_id": request.segment_id,
            "segment": request.segment,
            "concept": request.concept,
            "scene_details": request.scene_details,
            "global_context": request.global_context,
            "mode": request.mode,
            "video_type": request.video_type,
            "audio_paths": request.audio_paths,
            "generator_revision": request.generator_revision,
            "assets": self._asset_evidence(request),
        }
        encoded = json.dumps(
            _json_value(material),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def _asset_evidence(self, request: H3PromptCheckpointInput) -> list[dict[str, Any]]:
        candidates = list(request.audio_paths.values())
        references = request.segment.get("references")
        if isinstance(references, Mapping):
            for key, value in references.items():
                if key == "_stem_audio_tags" and isinstance(value, Mapping):
                    candidates.extend(value.keys())
                elif key.endswith("_path"):
                    candidates.append(value)
                elif key.endswith("_paths") and isinstance(value, (list, tuple)):
                    candidates.extend(value)
        evidence = {
            self._asset_identity(candidate)["path"]: self._asset_identity(candidate)
            for candidate in candidates
            if isinstance(candidate, (str, Path)) and str(candidate).strip()
        }
        return [evidence[key] for key in sorted(evidence)]

    def _asset_identity(self, value: str | Path) -> dict[str, Any]:
        path = Path(value)
        resolved = path if path.is_absolute() else self.project_dir / path
        resolved = resolved.resolve(strict=False)
        stat = resolved.stat() if resolved.is_file() else None
        signature = (stat.st_size, stat.st_mtime_ns) if stat is not None else None
        cached = self._asset_hashes.get(resolved)
        if cached is not None and cached[0] == signature:
            return cached[1]
        identity: dict[str, Any] = {"path": _project_path(resolved, self.project_dir)}
        if stat is not None:
            identity.update({
                "size": stat.st_size,
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            })
        else:
            identity["missing"] = True
        self._asset_hashes[resolved] = (signature, identity)
        return identity

    @staticmethod
    def _parse(path: Path, value: Any) -> H3PromptCheckpoint:
        if not isinstance(value, Mapping) or value.get("schema") != H3_CHECKPOINT_SCHEMA:
            raise FeverSlopDataError(f"Invalid H3 prompt checkpoint schema: {path}")
        generated = value.get("generated")
        if not isinstance(generated, dict) or not str(generated.get("prompt") or "").strip():
            raise FeverSlopDataError(f"Invalid H3 prompt checkpoint generated payload: {path}")
        status = value.get("status")
        if status not in {"good", "bad_exhausted", "unjudged"}:
            raise FeverSlopDataError(f"Invalid H3 prompt checkpoint status: {path}")
        try:
            return H3PromptCheckpoint(
                path,
                int(value["scene"]),
                str(value["scene_id"]),
                str(value["segment_id"]),
                status,
                str(value["input_fingerprint"]),
                deepcopy(generated),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FeverSlopDataError(f"Invalid H3 prompt checkpoint identity: {path}") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _structured_provenance(generated: Mapping[str, Any]) -> dict[str, Any]:
    provenance = generated.get("prompt_provenance")
    result: dict[str, Any] = {}
    if isinstance(provenance, Mapping):
        if provenance.get("compiler"):
            result["compiler"] = str(provenance["compiler"])
        if provenance.get("compiler_version") is not None:
            result["compiler_version"] = provenance["compiler_version"]
    for source_key, output_key in (
        ("creative_sections", "creative_sections_sha256"),
        ("locked_facts", "locked_facts_sha256"),
    ):
        value = generated.get(source_key)
        if value is None:
            continue
        encoded = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result[output_key] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return result


def _project_path(path: Path, project_dir: Path) -> str:
    try:
        return path.relative_to(project_dir.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()
