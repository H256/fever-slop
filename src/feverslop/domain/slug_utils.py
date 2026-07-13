from __future__ import annotations
import re


def slugify_project_name(value: str) -> str:
    """Convert arbitrary string to filesystem-safe project slug.

    Rules:
    - Lowercase alphanumeric, hyphens only.
    - Collapse consecutive separators into single hyphen.
    - Strip leading/trailing separators.
    """
    raw = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug
