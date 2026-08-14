from __future__ import annotations

import re

from feverslop.prompting.guide_loader import load_markdown_guide


def _krea_reference_guides(reference_hero_workflow: str | None) -> tuple[str, str]:
    """Return Krea guides only for a Krea reference-image workflow."""
    workflow = str(reference_hero_workflow or "").casefold()
    if "krea" not in workflow:
        return "", ""
    return load_markdown_guide("krea-location"), load_markdown_guide("krea-actor")


def _sanitize_location_image_prompt(value: str) -> str:
    text = re.sub(r"(?i)\b(?:cinematic\s+)?environment reference sheet(?:\s+for)?\b", "", value)
    text = re.sub(r"(?i)\breference sheet\b", "", text)
    return " ".join(text.split()).strip(" .,-")
