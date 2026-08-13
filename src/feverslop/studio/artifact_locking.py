"""Backward-compatible import location for artifact locking."""

from feverslop.adapters.artifact_locking import (  # noqa: F401
    _LOCKS,
    _LOCKS_GUARD,
    artifact_write_lock,
)

__all__ = ["artifact_write_lock"]
