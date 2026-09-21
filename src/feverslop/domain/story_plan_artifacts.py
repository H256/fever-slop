"""Artifact manifest contract for story plan artifacts (issue #1384).

Defines the versioned manifest that records how a story plan artifact was
produced (class, regeneration policy, input and dependency fingerprints),
plus atomic read/write helpers and a pure staleness check.

Scope (per #1384):
- Strict, versioned ``StoryPlanArtifactManifest`` models.
- Atomic JSON persistence for manifests and story plans.
- ``manifest_is_stale``: a pure staleness decision (no I/O).

Non-goals (per #1384):
- Production pipeline wiring.
- Computing dependency digests (callers use ``sha256_file``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from feverslop.domain.artifact_hash import is_sha256_hex
from feverslop.domain.story_plan import StoryPlan
from feverslop.errors import FeverSlopDataError
from feverslop.utils.io import atomic_write_json, read_json_document

#: Version of the artifact manifest serialization contract.
STORY_PLAN_ARTIFACT_SCHEMA_VERSION = "story-plan-artifact/v1"

#: Schema versions this contract can load.
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({STORY_PLAN_ARTIFACT_SCHEMA_VERSION})

#: Revision of the artifact producer.
ARTIFACT_PRODUCER_REVISION = "artifact-producer/v1"


class ArtifactClass(str, Enum):
    """What the artifact is authoritative for."""

    authoritative = "authoritative"
    resume_cache = "resume_cache"
    review_export = "review_export"


class RegenerationPolicy(str, Enum):
    """When the artifact may be regenerated."""

    never = "never"
    on_input_change = "on_input_change"
    on_request = "on_request"


class ArtifactDependency(BaseModel):
    """One input the artifact was produced from, by content digest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        return value


class StoryPlanArtifactManifest(BaseModel):
    """Versioned manifest for one story plan artifact."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = STORY_PLAN_ARTIFACT_SCHEMA_VERSION
    producer_revision: str = ARTIFACT_PRODUCER_REVISION
    artifact_class: ArtifactClass
    regeneration_policy: RegenerationPolicy
    input_fingerprint: str
    dependencies: list[ArtifactDependency] = Field(default_factory=list)
    plan_fingerprint: str | None = None

    @field_validator("input_fingerprint")
    @classmethod
    def _check_input_fingerprint(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("plan_fingerprint")
    @classmethod
    def _check_plan_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not is_sha256_hex(value):
            raise ValueError("plan_fingerprint must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def _check(self) -> "StoryPlanArtifactManifest":
        if self.schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported story plan artifact schema: {self.schema_version}"
            )
        seen: set[str] = set()
        for dependency in self.dependencies:
            if dependency.path in seen:
                raise ValueError(f"duplicate dependency path: {dependency.path}")
            seen.add(dependency.path)
        return self


def write_manifest(
    path: str | Path, manifest: StoryPlanArtifactManifest | dict
) -> Path:
    """Persist a manifest atomically (accepts a model or a raw payload)."""
    if isinstance(manifest, dict):
        manifest = _strict_manifest_from_data(manifest)
    return atomic_write_json(Path(path), manifest.model_dump(mode="json"))


def read_manifest(path: str | Path) -> StoryPlanArtifactManifest:
    """Load a manifest; parse/I/O failures map to ``FeverSlopDataError``."""
    data = read_json_document(path)
    if not isinstance(data, dict):
        raise FeverSlopDataError(f"story plan artifact manifest must be a JSON object: {path}")
    try:
        return _strict_manifest_from_data(data)
    except ValidationError as exc:
        raise FeverSlopDataError(f"malformed story plan artifact manifest: {path}") from exc


def write_story_plan(path: str | Path, plan: StoryPlan | dict) -> Path:
    """Persist a story plan atomically (accepts a model or a raw payload)."""
    if isinstance(plan, dict):
        plan = StoryPlan.from_json(json.dumps(plan))
    return atomic_write_json(Path(path), plan.model_dump(mode="json"))


def read_story_plan(path: str | Path) -> StoryPlan:
    """Load a story plan; parse/I/O failures map to ``FeverSlopDataError``."""
    data = read_json_document(path)
    if not isinstance(data, dict):
        raise FeverSlopDataError(f"story plan must be a JSON object: {path}")
    try:
        return StoryPlan.from_json(json.dumps(data))
    except (FeverSlopDataError, ValidationError) as exc:
        raise FeverSlopDataError(f"malformed story plan: {path}") from exc


def story_plan_fingerprint(plan: StoryPlan) -> str:
    """Digest canonical plan content for its adjacent manifest."""
    payload = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_matches_story_plan(
    manifest: StoryPlanArtifactManifest,
    plan: StoryPlan,
) -> bool:
    """Whether a manifest records the exact canonical plan it accompanies."""
    return manifest.plan_fingerprint == story_plan_fingerprint(plan)


def _strict_manifest_from_data(data: object) -> StoryPlanArtifactManifest:
    """Validate parsed JSON with JSON strictness, including enum values."""
    return StoryPlanArtifactManifest.model_validate_json(json.dumps(data), strict=True)


def manifest_is_stale(
    manifest: StoryPlanArtifactManifest,
    *,
    input_fingerprint: str,
    dependency_digests: Mapping[str, str] | None = None,
) -> bool:
    """Pure staleness decision: no I/O.

    Stale when the input fingerprint differs, a recorded dependency is
    missing from the digest mapping, or its digest differs. A missing file
    is a missing key, which is stale.
    """
    if manifest.input_fingerprint != input_fingerprint:
        return True
    digests = dict(dependency_digests or {})
    for dependency in manifest.dependencies:
        recorded = digests.get(dependency.path)
        if recorded is None or recorded != dependency.sha256:
            return True
    return False
