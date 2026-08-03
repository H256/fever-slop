from __future__ import annotations

from feverslop.errors import FeverSlopValidationError


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_R2V_IMAGE_REFS = 9
MAX_R2V_VIDEO_REFS = 3
MAX_R2V_AUDIO_REFS = 3
MAX_T2V_FRAME_REFS = 2  # first + last frame


# ---------------------------------------------------------------------------
# Helpers — R2V image refs
# ---------------------------------------------------------------------------

def _collect_r2v_image_refs(scene: dict) -> list[tuple[str, str]]:
    """Collect image reference (label, path) tuples for r2v.

    Order: actors, location, style-references — clamped to MAX_R2V_IMAGE_REFS.
    """
    references = scene.get("references")
    if not references:
        return []

    refs: list[tuple[str, str]] = []

    # -- actor sheets --------------------------------------------------------
    actor_paths = references.get("actor_sheet_paths", [])
    actor_descs = references.get("actor_reference_descriptions")

    for i, path in enumerate(actor_paths):
        if actor_descs and i < len(actor_descs):
            name = actor_descs[i].get("name", f"Actor {i + 1}")
        else:
            name = f"Actor {i + 1}"
        refs.append((name, path))

    # -- location sheet ------------------------------------------------------
    location_path = references.get("location_sheet_path")
    if location_path:
        location_desc = references.get("location_reference_description")
        name = location_desc.get("name", "Location") if location_desc else "Location"
        refs.append((name, location_path))

    # -- style references ----------------------------------------------------
    for entry in references.get("style_reference_paths", []):
        if isinstance(entry, dict):
            name = entry.get("name", "Style ref")
            path = entry.get("path", "")
        else:
            name = "Style ref"
            path = entry
        refs.append((name, path))

    return refs[:MAX_R2V_IMAGE_REFS]


# ---------------------------------------------------------------------------
# Helpers — T2V frame refs
# ---------------------------------------------------------------------------

def _collect_t2v_frame_refs(scene: dict) -> list[tuple[str, str]]:
    """Collect keyframe (label, path) tuples for t2v / i2v with frames.

    Requires *startframe_path*; *lastframe_path* is optional.
    """
    keyframes = scene.get("keyframes")
    if not keyframes:
        return []

    start = keyframes.get("startframe_path")
    if not start:
        return []

    refs: list[tuple[str, str]] = [("first_frame", start)]

    last = keyframes.get("lastframe_path")
    if last:
        refs.append(("last_frame", last))

    return refs[:MAX_T2V_FRAME_REFS]


# ---------------------------------------------------------------------------
# R2V prompt builder
# ---------------------------------------------------------------------------

def _build_r2v_prompt(scene: dict) -> str:
    """Build a MiniMax H3 reference-tag prompt from *scene* (reference-to-video).

    Image refs are rendered as ``<Picture N> {label} `` tags followed by the
    scene description.  Video refs become ``<Video N>`` and audio refs become
    ``<Audio N>`` tags.
    """
    if not isinstance(scene, dict):
        raise FeverSlopValidationError("scene must be a dict")
    if "description" not in scene:
        raise FeverSlopValidationError("scene must contain a 'description' key")

    parts: list[str] = []

    # picture (image reference) tags
    image_refs = _collect_r2v_image_refs(scene)
    for i, (label, _path) in enumerate(image_refs, start=1):
        parts.append(f"<Picture {i}> {label} ")

    # video reference tags
    references = scene.get("references") or {}
    video_refs = references.get("reference_video_paths", [])[:MAX_R2V_VIDEO_REFS]
    for _ in video_refs:
        parts.append("<Video> ")

    # audio reference tags
    audio_refs = references.get("reference_audio_paths", [])[:MAX_R2V_AUDIO_REFS]
    for _ in audio_refs:
        parts.append("<Audio> ")

    prompt = "".join(parts) + scene["description"]
    return prompt


# ---------------------------------------------------------------------------
# T2V prompt builder
# ---------------------------------------------------------------------------

def _build_t2v_prompt(scene: dict) -> str:
    """Build a MiniMax H3 reference-tag prompt from *scene* (text-to-video with optional frame refs).

    If frame refs are present they are rendered as ``<Picture N> {label} `` tags
    preceding the scene description.  Otherwise the raw description is returned.
    """
    if not isinstance(scene, dict):
        raise FeverSlopValidationError("scene must be a dict")
    if "description" not in scene:
        raise FeverSlopValidationError("scene must contain a 'description' key")

    frame_refs = _collect_t2v_frame_refs(scene)

    if not frame_refs:
        return scene["description"]

    parts: list[str] = []
    for i, (label, _path) in enumerate(frame_refs, start=1):
        parts.append(f"<Picture {i}> {label} ")

    prompt = "".join(parts) + scene["description"]
    return prompt


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_r2v_prompt(scene: dict) -> str:
    """Build a reference-to-video prompt for MiniMax H3.

    Args:
        scene: Scene dict with at least ``"description"``.  ``"references"`` is
            optional and defaults to an empty set.

    Returns:
        The final prompt string with reference tags and the scene description.

    Raises:
        FeverSlopValidationError: If *scene* is not a dict or lacks ``"description"``.
    """
    return _build_r2v_prompt(scene)


def build_t2v_prompt(scene: dict) -> str:
    """Build a text-to-video prompt for MiniMax H3 (with optional frame references).

    If no keyframe references are provided the prompt is identical to the scene
    description (pure T2V).  When ``scene["keyframes"]`` is present the prompt
    includes frame-reference tags.

    Args:
        scene: Scene dict with at least ``"description"``.

    Returns:
        The final prompt string.

    Raises:
        FeverSlopValidationError: If *scene* is not a dict or lacks ``"description"``.
    """
    return _build_t2v_prompt(scene)
