"""Validate the tracked, media-free benchmark project contract."""

from __future__ import annotations
from feverslop.ports.reporting import report_message

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "feverslop.benchmark-project/v1"


def _relative_path(value: Any, field: str, *, allow_parent: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or (not allow_parent and ".." in path.parts):
        raise ValueError(f"{field} must be portable and relative")
    return path


def validate_benchmark_project(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    manifest_path = root / "benchmark.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported benchmark manifest schema: {manifest.get('schema')!r}")
    project = manifest.get("project")
    if not isinstance(project, dict):
        raise ValueError("benchmark manifest requires project metadata")
    config_path = _relative_path(project.get("config"), "project.config")
    audio_path = _relative_path(project.get("input_audio"), "project.input_audio")
    if not (root / config_path).is_file():
        raise ValueError(f"benchmark config is missing: {config_path}")
    audio = root / audio_path
    if not audio.is_file():
        raise ValueError(f"benchmark input is missing: {audio_path}")
    expected_hash = project.get("input_audio_sha256")
    actual_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(f"benchmark input hash mismatch: {audio_path}")
    for item in manifest.get("expected_planning_artifacts", []):
        _relative_path(item, "expected_planning_artifacts", allow_parent=True)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("root") != "output/" or outputs.get("gitignored") is not True:
        raise ValueError("benchmark outputs must be isolated under ignored output/")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    validate_benchmark_project(args.project)
    report_message(f"validated benchmark project: {args.project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
