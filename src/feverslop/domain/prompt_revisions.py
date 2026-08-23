from __future__ import annotations

import datetime
import difflib
import hashlib
from dataclasses import dataclass, field
from enum import Enum


class PromptField(Enum):
    Z_IMAGE_PROMPT = "z_image_prompt"
    I2V_PROMPT = "i2v_prompt"


class DuplicateRevisionError(ValueError):
    def __init__(self, revision_id: str) -> None:
        super().__init__(f"Revision {revision_id!r} already exists")
        self.revision_id = revision_id


def _compute_content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compute_revision_id(
    scene_number: int,
    field_value: str,
    value: str,
    parent_id: str | None,
    now: datetime.datetime,
) -> str:
    canonical = (
        f"{scene_number}|{field_value}|{value}|{parent_id or ''}|{now.isoformat()}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PromptRevision:
    id: str
    project_id: str
    scene_number: int
    field: PromptField
    value: str
    parent_id: str | None
    restored_from: str | None
    content_hash: str
    created_at: datetime.datetime


@dataclass(frozen=True)
class PromptHistory:
    scene_number: int
    field: PromptField
    revisions: tuple[PromptRevision, ...] = field(default_factory=tuple)

    def __post_init__(self):
        sorted_revs = sorted(self.revisions, key=lambda r: r.created_at)
        object.__setattr__(self, "revisions", tuple(sorted_revs))

    @property
    def latest_value(self) -> str | None:
        if not self.revisions:
            return None
        return self.revisions[-1].value

    def diff_with_previous(self, revision_id: str) -> str | None:
        idx = next(
            (i for i, r in enumerate(self.revisions) if r.id == revision_id),
            None,
        )
        if idx is None:
            raise ValueError(f"Revision {revision_id!r} not found")
        if idx == 0:
            return None
        old_value = self.revisions[idx - 1].value
        new_value = self.revisions[idx].value
        lines = list(
            difflib.unified_diff(
                old_value.splitlines(keepends=True),
                new_value.splitlines(keepends=True),
                fromfile="previous",
                tofile="current",
            ),
        )
        return "".join(lines)


def build_revision(
    *,
    project_id: str,
    scene_number: int,
    field: PromptField,
    value: str,
    parent_id: str | None,
    now: datetime.datetime,
) -> PromptRevision:
    if not isinstance(field, PromptField):
        raise ValueError(f"Unknown prompt field: {field!r}")
    if scene_number <= 0:
        raise ValueError("Scene number must be a positive integer")
    if not value.strip():
        raise ValueError("Prompt value must not be blank")

    revision_id = _compute_revision_id(
        scene_number, field.value, value, parent_id, now,
    )
    content_hash = _compute_content_hash(value)

    return PromptRevision(
        id=revision_id,
        project_id=project_id,
        scene_number=scene_number,
        field=field,
        value=value,
        parent_id=parent_id,
        restored_from=None,
        content_hash=content_hash,
        created_at=now,
    )


def restore_revision(
    history: PromptHistory,
    *,
    revision_id: str,
    now: datetime.datetime,
    with_parent: bool = False,
) -> PromptRevision:
    if not history.revisions:
        raise ValueError("Cannot restore from empty history")

    source = next(
        (r for r in history.revisions if r.id == revision_id),
        None,
    )
    if source is None:
        raise ValueError(f"Revision {revision_id!r} not found in history")

    parent_id = source.id if with_parent else history.revisions[-1].id

    canonical = (
        f"{source.scene_number}|{source.field.value}|{source.value}"
        f"|{parent_id or ''}|{now.isoformat()}|restore:{revision_id}"
    )
    new_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    content_hash = _compute_content_hash(source.value)

    return PromptRevision(
        id=new_id,
        project_id=source.project_id,
        scene_number=source.scene_number,
        field=source.field,
        value=source.value,
        parent_id=parent_id,
        restored_from=revision_id,
        content_hash=content_hash,
        created_at=now,
    )
