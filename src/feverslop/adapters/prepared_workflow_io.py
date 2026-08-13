"""Filesystem adapter for prepared workflow manifests.

The manifest model itself exposes ``to_dict``/``from_dict`` and remains usable
without a filesystem. This module owns JSON persistence for pipeline callers.
"""
from __future__ import annotations

import json
from pathlib import Path

from feverslop.domain.prepared_workflow import SceneWorkflowManifest


def read_manifest(path: str | Path) -> SceneWorkflowManifest:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return SceneWorkflowManifest.from_dict(payload)


def write_manifest(manifest: SceneWorkflowManifest, path: str | Path) -> Path:
    return manifest.write(path)

