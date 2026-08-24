from __future__ import annotations

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.domain.canonical_render_plan import validate_canonical_plan
from feverslop.domain.project_render_settings import ProjectRenderSettings


def sync_project_render_settings(
    store: CanonicalPlanStore,
    settings: ProjectRenderSettings,
) -> bool:
    snapshot = store.capture_regeneration()
    if not snapshot.exists:
        raise FileNotFoundError("Cannot synchronize project settings without base.json")
    updated = settings.apply_to_scenes(list(snapshot.scenes))
    if updated == list(snapshot.scenes):
        return False
    validate_canonical_plan(updated)
    store.commit_regeneration(snapshot, updated)
    return True
