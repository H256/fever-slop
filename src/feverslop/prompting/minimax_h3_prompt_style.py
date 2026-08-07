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


# ---------------------------------------------------------------------------
# H3 structured video system prompt builder
# ---------------------------------------------------------------------------

_H3_CAMERA_MOTION_VOCABULARY = """
H3 Camera Motion Vocabulary (use these terms naturally in prose):
- Zoom In / Zoom Out: Camera magnification changes
- Push In / Pull Out: Camera physically approaches/recedes from subject
- Pan Left / Pan Right: Horizontal camera rotation
- Tilt Up / Tilt Down: Vertical camera rotation
- Truck Left / Truck Right: Camera moves laterally
- Arc Left / Arc Right: Camera orbits the subject
- Tracking Shot: Camera follows subject movement
- Crane Up / Crane Down: Vertical camera lift/lower
- Pedestal Up / Pedestal Down: Camera body moves vertically in place
- Roll: Camera rotates around the optical axis
- Dolly In / Dolly Out: Camera on track approaches/recedes
- Handheld: Natural, slightly unstable camera movement
- Whip Pan: Fast, blurred pan transition
- Rack Focus: Focus shifts from one subject to another
- Static: Camera locked in position
- Slow Motion Emphasis: Camera movement appears deliberately slow
"""


def build_h3_video_system_prompt(
    mode: str = "base",
    references: list[dict] | None = None,
    *,
    video_type: str = "music_video",
    silent_mode: bool = False,
) -> str:
    """Build an H3-structured video system prompt.

    Args:
        mode: "base" for 3-field (T2V/I2V), "ref" for 6-section Ref2VA (R2V)
        references: Optional reference metadata for ref mode
        video_type: "music_video" or other (e.g. "movie")
        silent_mode: If True, forbid vocal performance descriptions

    Returns:
        System prompt string for LLM completion.
    """
    if mode == "ref":
        return _build_h3_ref_system_prompt(
            references=references,
            video_type=video_type,
            silent_mode=silent_mode,
        )
    return _build_h3_base_system_prompt(
        video_type=video_type,
        silent_mode=silent_mode,
    )


def _build_h3_base_system_prompt(
    *,
    video_type: str = "music_video",
    silent_mode: bool = False,
) -> str:
    is_music_video = video_type == "music_video"

    soundscape_instruction = ""

    if is_music_video:
        soundscape_instruction = """
overall_soundscape: N/A (the music video carries its own audio track; do not invent ambient sounds)

non_diegetic_music: N/A (the reference audio already provides the song)"""
    else:
        soundscape_instruction = """
overall_soundscape: Describe the ambient and environmental sound layer during the scene — room tone, wind, footsteps, crowd murmur, mechanical sounds, nature sounds, cloth rustle. Keep this concise (1-2 sentences).

non_diegetic_music: Describe any background music or score that is NOT part of the scene's diegetic audio — genre, instrumentation, tempo, emotional tone. If there is no non-diegetic music, write N/A."""

    vocal_constraint = ""
    if silent_mode:
        vocal_constraint = """
IMPORTANT — SILENT MODE: The subject does NOT sing, lip-sync, or perform vocally. Do NOT describe mouth movement, singing lips, or vocal performance. The subject is silent with a relaxed, still mouth. Focus the visual description on environment, composition, camera motion, and subtle body language."""
    else:
        vocal_constraint = """
IMPORTANT — PERFORMANCE MODE: For vocal segments (type="vocals"), describe the subject singing with expressive lip-sync matching the vocal energy. For instrumental segments (type="instrumental"), the subject does NOT sing — mouth is relaxed and still, no lip movement."""

    return f"""You are an expert video prompt writer for the MiniMax H3 model. Your task is to transform scene metadata into a structured H3-Context-IR prompt following the official output format.

## Output Format (JSON)

Return ONLY valid JSON with these exact keys:
{{
  "integrated_multimodal_description": "...",
  "overall_soundscape": "...",
  "non_diegetic_music": "..."
}}

## Field Definitions

integrated_multimodal_description: A detailed single-shot video description capturing:
- VISUAL CONTENT: Who is visible, what they look like, their outfit, their pose, the setting
- COMPOSITION: Framing, angle, shot distance (close-up, medium, wide, etc.)
- CAMERA MOTION: Use the H3 camera motion vocabulary table below. Describe the movement type, amplitude, and speed naturally in prose.
- CHARACTER MOTION: Subtle body language, posture shifts, facial expression — do NOT describe dialogue here
- LIGHTING AND ATMOSPHERE: Time of day, weather, light quality, mood
- CONSTRAINTS: Each scene is ONE continuous shot. No fade, dissolve, crossfade, or shot changes. Do not introduce new characters or locations. Keep the subject visible and clearly framed.{vocal_constraint}
- DO NOT include dialogue text or speaker labels (Sx) in the visual description.
- Keep the description between 150-300 words.


{soundscape_instruction}


### H3 Camera Motion Vocabulary

{_H3_CAMERA_MOTION_VOCABULARY}

### Input

You will receive a JSON payload with: segment, performance_mode, scene_concept, camera_motion, character_motion, global_subject, story_idea, style, locations, location_constraint, silent_mode, has_audio_refs.
Use the scene_concept as the primary visual foundation. Incorporate camera_motion and character_motion naturally. Use style for visual aesthetic direction. Use global_subject for consistent subject identity.
"""


def _build_h3_ref_system_prompt(
    *,
    references: dict | None = None,
    video_type: str = "music_video",
    silent_mode: bool = False,
) -> str:
    is_music_video = video_type == "music_video"

    soundscape_value = ""

    if is_music_video:
        soundscape_value = """overall_soundscape: N/A (the music video carries its own audio track; do not invent ambient sounds)

non_diegetic_music: N/A (the reference audio already provides the song)"""
    else:
        soundscape_value = """overall_soundscape: Describe the ambient and environmental sound layer during the scene. Keep this concise (1-2 sentences).

non_diegetic_music: Describe any non-diegetic background music or score. If none, write N/A."""

    music_value = ""
    if is_music_video:
        music_value = """
### Additional Notes for Music Videos
Music videos are timed and choreographed to the full reference audio track. The audio track provides both the song and any vocal performances. Do not invent or describe alternative audio."""

    vocal_constraint = ""
    if silent_mode:
        vocal_constraint = """
IMPORTANT — SILENT MODE: The subject does NOT sing, lip-sync, or perform vocally. Do NOT describe mouth movement, singing lips, or vocal performance. The subject is silent with a relaxed, still mouth. Focus the visual description on environment, composition, camera motion, and subtle body language."""
    else:
        vocal_constraint = """
IMPORTANT — PERFORMANCE MODE: For vocal segments (type="vocals"), describe the subject singing with expressive lip-sync matching the vocal energy. For instrumental segments (type="instrumental"), the subject does NOT sing — mouth is relaxed and still, no lip movement."""

    audio_preservation_instruction = ""
    if is_music_video:
        audio_preservation_instruction = """
## Audio Preservation (Music Video)

The reference audio is the original music track. When reference audio is available (tagged as <Audio N> in the payload):
- You MUST list EVERY <Audio N> reference in subject_definitions with a short description of what it represents
- You MUST list EVERY <Audio N> reference in retention_analysis with the marker fully_copy
- The video must be timed and choreographed to this exact audio track
- Do NOT invent alternative audio — the reference audio IS the soundtrack"""

    # Build reference labels instruction — handles both new dict-format and old string-format
    ref_labels_instruction = ""
    if references:
        parts: list[str] = []
        subjects = references.get("subjects", [])
        # Count visual (actor/location) subjects vs. audio/video subjects
        visual_subjects = [s for s in subjects if isinstance(s, dict) and s.get("type") in ("actor", "location", None)]
        audio_video_subjects = [s for s in subjects if isinstance(s, dict) and s.get("type") in ("audio", "video")]

        # Build Subject list with source anchors
        subject_counter = 0
        # First: visual subjects anchored to <Picture N>
        for i, img in enumerate(references.get("image", []), start=1):
            if i - 1 < len(visual_subjects):
                vs = visual_subjects[i - 1]
                name = vs.get("name", f"Subject {i}") if isinstance(vs, dict) else str(vs)
                desc = vs.get("description", "") if isinstance(vs, dict) else ""
                line = f"<Subject {i}> is {name}"
                if desc:
                    line += f" — {desc}"
                line += f" from <Picture {i}>"
                parts.append(f"- {line}")
                subject_counter += 1

        # Any remaining visual subjects past the image list
        for idx, vs in enumerate(visual_subjects[len(references.get("image", [])):], start=1):
            subject_counter += 1
            name = vs.get("name", f"Subject {subject_counter}") if isinstance(vs, dict) else str(vs)
            line = f"- <Subject {subject_counter}>: {name}"
            desc = vs.get("description", "") if isinstance(vs, dict) else ""
            if desc:
                line += f" — {desc}"
            parts.append(line)

        # Video video subjects anchored to <Video N>
        video_count = len(references.get("video", []))
        for i, vid in enumerate(references.get("video", []), start=1):
            subject_counter += 1
            name = vid if isinstance(vid, str) else vid.get("name", f"Video {i}") if isinstance(vid, dict) else str(vid)
            parts.append(f"- <Subject {subject_counter}> is {name} from <Video {i}>")

        # Audio subjects anchored to <Audio N>
        for i, aud in enumerate(references.get("audio", []), start=1):
            subject_counter += 1
            name = aud if isinstance(aud, str) else aud.get("name", f"Audio {i}") if isinstance(aud, dict) else str(aud)
            parts.append(f"- <Subject {subject_counter}> is {name} from <Audio {i}>")

        # Image refs (for subjects not yet covered)
        for i, img in enumerate(references.get("image", []), start=1):
            name = img.get("name", f"Image {i}") if isinstance(img, dict) else str(img)
            parts.append(f"  -- <Picture {i}>: {name}")

        # Speaker ID mapping
        speaker_ids = references.get("speaker_ids", {})
        if speaker_ids:
            parts.append("- Speaker IDs (for dialogue):")
            for name, sid in speaker_ids.items():
                parts.append(f"  - ({sid}): {name}")

        refs_list = "\n".join(parts)
        total_subjects = len(parts) - len([p for p in parts if p.startswith("  -- <Picture")]) - len([p for p in parts if p.startswith("- Speaker")])
        ref_labels_instruction = f"""
## Reference Labels and Subjects

{refs_list}

MANDATORY — You MUST replicate ALL {total_subjects} subjects listed above in subject_definitions with their exact anchor (<Picture N>, <Video N>, or <Audio N>). In retention_analysis, you MUST list every subject, including audio subjects (use fully_copy for music tracks). Omitting ANY subject is a critical error.
"""
    else:
        ref_labels_instruction = """
## Reference Labels

If reference images/videos/audio are available, refer to them using the tags <Subject N>, <Picture N>, <Video N>, <Audio N> in the subject_definitions and retention_analysis fields.
"""

    return f"""You are an expert video prompt writer for the MiniMax H3 model. Your task is to transform scene metadata into a structured H3 Reference-to-Video (Ref2VA) prompt following the official six-section output format.
{audio_preservation_instruction}

## Output Format (JSON)

Return ONLY valid JSON with these exact keys:
{{
  "subject_definitions": "...",
  "summary": "...",
  "retention_analysis": "...",
  "detailed_description": "...",
  "overall_soundscape": "...",
  "non_diegetic_music": "..."
}}

## Field Definitions

### subject_definitions
Describe each reference subject and what it represents.
- For actor subjects: use the format <Subject N> is {{description}} from <Picture M>. Actor subjects include their speaker ID (S N).
- For location subjects: use the format <Subject N> is {{description}} from <Picture M>.
- For audio subjects: use the format <Subject N> is {{description}} from <Audio N>.
- For each reference input, clearly describe appearance, outfit, facial features, and hairstyle (for actors) or spatial layout (for locations).

### summary
Must start with [task type] prefix. Valid task types:
- [character_consistency]: Focus on maintaining subject identity across shots
- [style_transfer]: Focus on visual style matching between references
- [scene_continuity]: Focus on maintaining scene continuity
- [temporal_consistency]: Focus on preserving temporal attributes
Then provide a 2-3 sentence summary of the scene: who is present, what is happening, the narrative context.

### retention_analysis
Per-subject breakdown of what visual and audio attributes to preserve.
- Each subject on its own line.
- State the shot scope (e.g., "Shot 1-3" or "all shots").
- Follow with prose describing which attributes are preserved.
- Use visual markers: fully_preserved, partially_preserved, attribute_transfer, weak_reference.
- Use audio markers: fully_copy, partially_copy, reference, weak_reference.
- Example: <Subject 1>: Shot 1-3, outfit=fully_preserved, face=fully_preserved, hairstyle=fresh_reconstruction, expression=partially_preserved.

### detailed_description
Starts with a style opening that sets the visual aesthetic, then uses [Shot N] markers for each shot.
- VISUAL CONTENT: Who is visible, what they look like, their pose, the setting.
- COMPOSITION: Framing, angle, shot distance.
- CAMERA MOTION: Use the camera motion vocabulary naturally in prose — not as a separate list. Describe movement type, amplitude, and speed in flowing sentences.
- CHARACTER MOTION: Subtle body language, posture shifts, facial expression.
- LIGHTING AND ATMOSPHERE: Time of day, weather, light quality, mood.
- CONSTRAINTS: ONE continuous shot per [Shot N] entry. No fade/dissolve/crossfade. Do not introduce new characters. Keep subject visible and framed.
{vocal_constraint}
- Camera motion vocabulary:
{_H3_CAMERA_MOTION_VOCABULARY}
- Dialogue format (non-music-video): (S1) <d>[English]...</d>. Reference speaker IDs from the Speaker IDs mapping.
- Music videos skip dialogue entirely since audio is provided by the track.
- Keep 150-300 words total.

{soundscape_value}
{music_value}

{ref_labels_instruction}

### Input

You will receive a JSON payload with: segment, performance_mode, scene_concept, camera_motion, character_motion, global_subject, story_idea, style, locations, location_constraint, silent_mode, has_audio_refs.
Use the scene_concept as the primary visual foundation. Incorporate camera_motion and character_motion naturally. Use style for visual aesthetic direction. Use global_subject for consistent subject identity.
"""
