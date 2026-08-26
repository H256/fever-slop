from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_PIPELINE_MODE = {
    "ltx_i2v": "i2v",
    "ltx_msr": "msr",
    "ltx_ingredients": "ingredients",
}


@dataclass(frozen=True)
class LTX25MigrationResult:
    profile_id: str
    mode: str
    quality: str
    changed: bool
    warnings: tuple[str, ...] = ()


def diagnose_and_migrate_config(payload: dict[str, Any]) -> tuple[dict[str, Any], LTX25MigrationResult]:
    """Return a migrated copy; never mutates input and never silently chooses 2.3."""
    if not isinstance(payload, dict):
        raise ValueError("project config must be an object")
    result = dict(payload)
    pipeline = str(result.get("video_pipeline") or "ltx_i2v").strip().lower()
    mode = _PIPELINE_MODE.get(pipeline)
    if mode is None:
        raise ValueError(f"unsupported LTX pipeline for migration: {pipeline}")
    raw = result.get("render_profile")
    quality = str(raw.get("quality", "draft") if isinstance(raw, dict) else "draft").strip().lower()
    if quality not in {"draft", "standard", "final"}:
        raise ValueError("render profile quality must be draft, standard, or final")
    profile_id = f"ltx25-{mode}-{quality}"
    changed = raw != profile_id
    result["render_profile"] = profile_id
    warnings = () if not changed else ("profile normalized to LTX 2.5",)
    return result, LTX25MigrationResult(profile_id, mode, quality, changed, warnings)
