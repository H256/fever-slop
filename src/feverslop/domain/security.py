"""Central security utilities for FeverSlop.

Provides path containment checks and sanitization for user-supplied paths
and identifiers, as well as audit helpers that ensure security patterns
are never regressed.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath


# ---------------------------------------------------------------------------
# Path containment helpers
# ---------------------------------------------------------------------------


class SecurityError(ValueError):
    """Raised when a security guard detects a violation."""
    pass


def guard_path_under_root(
    path: str | Path,
    root: str | Path,
    /,
) -> Path:
    """Resolve *path* and ensure it stays under *root*.

    Raises :class:`SecurityError` if the resolved path escapes the root
    directory (e.g. via ``..`` traversal).
    """
    target = Path(path).resolve()
    base = Path(root).resolve()
    if not target.is_relative_to(base):
        raise SecurityError(
            f"Path escapes root: {target} is not under {base}"
        )
    return target


def sanitize_path_component(value: str) -> str:
    """Strip directory separators and traversal tokens from a single path component.

    Intended for user-supplied slugs, identifiers, and filename stems where
    *any* path-like input must be reduced to a safe single-component name.
    """
    # Remove all directory separators and traversal patterns
    cleaned = re.sub(r'[\\/]', "", value)
    cleaned = re.sub(r"\.{2,}", "", cleaned)  # collapse ".." and longer
    cleaned = cleaned.strip("./")
    return cleaned


def is_safe_identifier(value: str) -> bool:
    """Return ``True`` if *value* contains no path traversal tokens.

    Rejects ``..`` segments and absolute path forms on every supported host.
    """
    if value.startswith(("/", "\\")):
        return False
    paths = (PurePosixPath(value), PureWindowsPath(value))
    return not any(
        path.is_absolute() or path.drive or ".." in path.parts
        for path in paths
    )


# ---------------------------------------------------------------------------
# Audit helpers – verify the codebase stays secure
# ---------------------------------------------------------------------------

def _scan_source_files(src_root: str | Path, pattern: str) -> list[tuple[str, int, str]]:
    """Return (file, line, text) for every regex match in ``.py`` files under *src_root*."""
    src = Path(src_root)
    matches: list[tuple[str, int, str]] = []
    if not src.is_dir():
        return matches
    for py_file in sorted(src.rglob("*.py")):
        try:
            text = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(pattern, line):
                matches.append((str(py_file), lineno, line.rstrip()))
    return matches


def audit_no_pickle(src_root: str | Path) -> list[str]:
    """Assert the codebase uses no pickle (RCE risk)."""
    matches = _scan_source_files(
        src_root,
        r"pickle\.(loads?|dumps?|load|dump)\b",
    )
    return [f"  {f}:{lineno}" for f, lineno, _ in matches]


def audit_no_shell_true(src_root: str | Path) -> list[str]:
    """Assert subprocess calls never use unquoted shell mode."""
    matches = _scan_source_files(
        src_root,
        r"shell\s*=\s*True\b",
    )
    return [f"  {f}:{lineno}" for f, lineno, _ in matches]


def audit_no_fstring_sql(directory: str | Path) -> list[str]:
    """Assert ``cursor.execute()`` or ``connection.execute()`` never use f-strings."""
    matches = _scan_source_files(
        directory,
        r"\.(execute|executescript)\s*\(\s*f[\"']",
    )
    return [f"  {f}:{lineno}" for f, lineno, _ in matches]


def run_all_audits(
    src_root: str | Path = "src",
    infra_root: str | Path | None = None,
) -> dict[str, list[str]]:
    """Run all audit checks and return findings.

    Returns a dict mapping check name to a list of issue strings.
    An empty list means the check passed.
    """
    infra = infra_root or str(Path(src_root) / "feverslop" / "infra")
    return {
        "no_pickle": audit_no_pickle(src_root),
        "no_shell_true": audit_no_shell_true(src_root),
        "no_fstring_sql": audit_no_fstring_sql(infra),
    }
