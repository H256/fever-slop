from __future__ import annotations

from collections.abc import Iterable, Mapping


def render_reference_contract(
    references: Iterable[Mapping[str, object]],
    *,
    profile: str = "",
    actor_roles: Mapping[str, object] | None = None,
    prop_bindings: Mapping[str, Iterable[object]] | None = None,
) -> str:
    """Render deterministic reference rules without inventing domain semantics."""
    items = [dict(reference) for reference in references]
    subject_names = [
        str(reference.get("name") or reference.get("label") or "").strip()
        for reference in items
        if str(reference.get("role") or "").strip().lower() == "subject"
    ]
    environment_names = [
        str(reference.get("name") or reference.get("label") or "").strip()
        for reference in items
        if str(reference.get("role") or "").strip().lower() == "environment"
    ]

    lines = [
        "Reference identity and continuity contract:",
        "Each referenced character represents exactly one persistent physical individual.",
        "Do not duplicate, clone, mirror, split, replace, or reassign a referenced character.",
        "Camera movement, shot changes, and temporal transitions do not create additional subject instances.",
        "Explicitly bound props remain attached to their subject unless the shot plan explicitly changes that binding.",
    ]
    for name in subject_names:
        if name:
            lines.append(f"Exactly one persistent instance of {name} is present unless the shot plan explicitly says otherwise.")
    if environment_names:
        lines.extend([
            "Ambient background people may exist as anonymous extras in an environment reference.",
            "Ambient background people must not replace, duplicate, or impersonate an explicitly referenced character.",
        ])

    if str(profile or "").strip().lower() == "live_concert":
        lines.extend([
            "Live-concert staging contract: performers remain on the main festival stage.",
            "There is no catwalk, podium, or satellite platform unless the shot plan explicitly specifies one.",
            "The environment may contain an anonymous audience, but it must not contain identifiable duplicates of the referenced band members.",
        ])
        role_map = actor_roles or {}
        binding_map = {str(actor): list(props) for actor, props in (prop_bindings or {}).items()}
        for actor, role in role_map.items():
            role_text = str(role).casefold()
            inferred_prop = next(
                (
                    prop
                    for needles, prop in (
                        (("singer", "frontman", "vocal"), "microphone"),
                        (("drummer", "drum"), "drum kit"),
                        (("bass",), "bass guitar"),
                        (("keyboard", "keys"), "keyboard"),
                        (("guitar",), "guitar"),
                    )
                    if any(needle in role_text for needle in needles)
                ),
                None,
            )
            if inferred_prop and not binding_map.get(str(actor).strip()):
                binding_map[str(actor)] = [inferred_prop]
        for actor, props in binding_map.items():
            actor_name = str(actor).strip()
            for prop in props:
                prop_name = str(prop).strip()
                if actor_name and prop_name:
                    lines.append(f"{actor_name} remains bound to {prop_name} for the shot unless explicitly changed.")
        for actor, role in role_map.items():
            actor_name = str(actor).strip()
            role_name = str(role).strip()
            if actor_name and role_name:
                lines.append(f"{actor_name} retains the role {role_name}; do not transfer that role to another subject.")

    return "\n".join(lines)
