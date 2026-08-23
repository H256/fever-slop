from __future__ import annotations


def build_location_constraint(locations: list[str] | tuple[str, ...] | None) -> str:
    cleaned = [str(location).strip() for location in (locations or []) if str(location).strip()]
    if not cleaned:
        return ""

    lines = ["Allowed locations:"]
    lines.extend(f"- {location}" for location in cleaned)
    lines.extend(
        [
            "",
            "Every scene concept and every Z-Image prompt must visibly take place in one of the allowed locations listed above.",
            "Each scene concept and Z-Image prompt must explicitly name one allowed location or a direct visual variant of it.",
            "Do not invent other locations.",
        ],
    )
    return "\n".join(lines)
