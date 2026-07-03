from __future__ import annotations

from typing import Any

from feverslop.domain.prompt_constraints import build_location_constraint  # noqa: F401


def performance_policy(segment_type: str, *, silent_mode: bool = False) -> str:
    if silent_mode:
        return (
            "Performance policy: dialogue-free silent mode is active. The subject must not sing, "
            "must have no lip sync, no vocal performance, no dialogue delivery, no mouth performance, "
            "and no moving lips. Preserve emotional acting through expressive eyes, facial emotion, "
            "gaze, posture, hands, body movement, camera movement, and environment."
        )

    mode = str(segment_type or "").strip().lower()

    if mode == "vocals":
        return (
            "Performance policy: the subject is singing with passion and expressive lip sync, "
            "with the mouth performance matching the vocal energy and the face showing clear emotion."
        )

    if mode == "mixed":
        return (
            "Performance policy: the scene alternates between vocal intervals and silent intervals. "
            "Use singing with passion and lip sync only during vocal intervals. During silent intervals "
            "the subject must not sing, must have no lip sync, and keeps a closed or relaxed mouth."
        )

    return (
        "Performance policy: this is an instrumental section. The subject must not sing, must have "
        "no lip sync, no mouth performance, and no moving lips. Keep a closed or relaxed mouth while "
        "the emotion comes from gaze, posture, hands, body movement, camera movement, and environment."
    )


def build_t2i_system_prompt() -> str:
    return """
Create one text-to-image prompt from the user input.

User input includes:
- subject
- one current visual prompt
- a style/theme
- segment type and optional lyrics

Use all parts of the user input together.

Priority:
- Use the current visual prompt as the main scene foundation.
- Keep the main action, subject, and setting from the current visual prompt unless the user clearly changes them.
- Use the style/theme to control the visual aesthetic, color grading, lighting, mood, wardrobe refinement, environment design, and overall cinematic treatment.
- Use the provided subject as the main subject of the image.

Rules:
- Create one polished text-to-image prompt.
- If location_constraint is provided in the user input, follow it as a mandatory rule. Every Z-Image prompt must visibly take place in one allowed location.
- Treat the current visual prompt as the base scene description.
- Expand and improve that scene using the style/theme.
- Keep the image prompt concrete and visual.
- Use the style/theme to influence color palette, tone, texture, lighting style, atmosphere, and production quality.
- If the current visual prompt includes concrete objects, actions, reflections, or setting details, keep them visible in the final prompt.
- Do not use metaphors, abstract symbolic wording, or non-visible language.
- Describe only things that can be seen in the final image.
- Keep the result as one strong image prompt, not a summary.
- Correct obvious typos, malformed words, and broken phrases before using them.
- Do not mention that typos were fixed.
- Generate a still image prompt, not a video prompt.
- Do not include camera movement.
- Do not include lip sync.
- Do not say the character is singing.
- If the segment is instrumental, the subject must not sing and must not have mouth movement.
- If the segment is vocal or mixed, the keyframe may show an expressive performance pose, but still avoid singing, lip sync, or moving lips in the still image prompt.
- Do not quote lyrics.
- Do not explain your choices.

Use this exact format:

A high resolution cinematic photograph of a [subject], [action or pose based primarily on the current visual prompt], in [environment/location shaped by the current visual prompt], during [time of day]. The subject is wearing [main outfit from the current visual prompt refined by the style/theme], [shoes/accessories from the current visual prompt refined by the style/theme], and [additional visible style details inspired by the style/theme]. Their hair is [hair color], [hair length/style], and [movement or texture]. The environment is [visual style of location from the current visual prompt shaped by the style/theme] with [background details that visibly represent the current visual prompt], [lighting and color grading details that match the style/theme], and [surface/reflection/material details connected to the current visual prompt and style/theme]. Camera is [camera angle] with a [lens type or framing]. The weather is [weather condition appropriate to the scene], with [atmospheric detail influenced by the style/theme], creating a [mood/style] mood.

The subject placeholder must be replaced with the concrete character description. Do not write "subject" as the character.

Only send the final prompt text. Do not include labels, notes, quotes, markdown, or extra text.
""".strip()


def build_i2v_system_prompt(segment_type: str, *, silent_mode: bool = False) -> str:
    mixed_vocal_rule = (
        "Silent mode is active, so never add sung or lip-synced vocal portions."
        if silent_mode
        else "If the scene is mixed, allow the vocal portions to be sung with passion while keeping silent portions relaxed and closed-mouth."
    )
    return f"""
Convert the user's concept prompt into a dynamic image-to-video prompt.

Use the user's prompt as the full scene foundation. Preserve the original subject, setting, outfit, mood, atmosphere, and scene identity. Infer only the missing video details needed to make the scene feel complete, including time of day, weather, lighting behavior, environmental movement, subject movement, camera movement, and performance energy. Do not add unrelated characters, new locations, major story changes, captions, text overlays, dialogue, or audio instructions.

Use the t2i_prompt or current visual prompt as the visual reference if one is supplied. {mixed_vocal_rule}

Add fast, cinematic motion by giving the subject a clear action sequence, expressive facial expressions, strong gestures, and intentional camera movement. Keep the subject visible, centered, and clearly framed throughout. Add lighting only as natural scene behavior, such as flickering stage lights, passing sunlight, glowing streetlights, storm flashes, reflections, or shifting shadows, based on what best fits the user's prompt.

{performance_policy(segment_type, silent_mode=silent_mode)}

Output one polished paragraph using this structure:

The [Subject and performance state] in [setting/environment] during [time/weather]. The subject [dynamic action sequence with expressive face, body movement, and strong gestures]. Their clothing/hair [reacts to movement, wind, or performance energy]. The lighting [changes or reacts naturally within the scene]. The camera [Camera Motion] while maintaining [subject visibility and framing]. The environment [reacts dynamically].

Rules:
- This is image-to-video.
- Keep it vivid, fast, cinematic, dynamic, and video-ready.
- Keep the subject visible and clearly framed throughout.
- Use one established location from the user's concept prompt or location list.
- Must use user input to help create the prompt.
- Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color grading, camera photo style, or static image-quality descriptions.
- Only send the final prompt text. Do not include labels, notes, quotes, markdown, or extra text.
""".strip()


def build_concept_mapper_system_prompt(*, batch: bool = False, silent_mode: bool = False) -> str:
    batch_text = (
        "You receive only one batch of timed song sections, but the whole video must remain continuous."
        if batch
        else "You receive the full timed song section list."
    )
    target_text = "CURRENT_BATCH_SEGMENTS" if batch else "SEGMENT_TIMELINE_JSON"
    segment_rules = (
        "\n".join(
            [
                "Segment rules:",
                "- Silent mode is active. Treat every segment as dialogue-free and non-vocal in the visuals.",
                "- Do not create singing, lip-sync, vocal performance, mouth performance, or dialogue delivery.",
                "- You may use lyrics only as emotional or narrative context.",
                "- Preserve emotional expression through gestures, posture, gaze, body language, framing, light, and environment.",
                "- Do not quote lyrics.",
            ]
        )
        if silent_mode
        else "\n".join(
            [
                "Segment rules:",
                "- For vocal segments, reflect the lyrics and performance energy.",
                "- For instrumental segments, advance the visual story without sung words. Do not say the character is singing. Do not mention lip sync.",
                "- For mixed segments, combine lyrical meaning with instrumental mood, but do not force singing across the whole segment.",
                "- Do not quote lyrics.",
            ]
        )
    )

    return f"""
You are a professional music video director and cinematographer creating a continuous visual story for a text-to-video model.

{batch_text}

TASK:
Create exactly one concise visual concept and scene reference assignment for each segment in {target_text}.

Continuity rules:
- Read all available segments first and infer the full emotional and visual arc.
- Treat the segments as one continuous visual story.
- Maintain character, outfit, environment, lighting direction, motifs, and story progression.
- Each concept must stand alone because the video model has no memory.
- Repeat key visible continuity details instead of referring to previous segments.
- Never write "the same character", "still", "continues", "next", "after", or "from earlier".

{segment_rules}

Visual rules:
- Include visible action, environment, lighting, camera-relevant composition, cinematic detail, and visual emotion.
- Show emotion through gestures, posture, gaze, light, framing, and environment.
- Do not use poetic metaphors, abstract moods, or non-visible language.
- Do not invent or assume character details such as hair color, skin tone, age, ethnicity, eye color, or body type unless explicitly provided in the subject.
- Do not describe technical render parameters.

Prompt guidance:
- If GLOBAL_CONTEXT.location_constraint is provided, follow it as a mandatory rule. Every scene concept must visibly take place in one allowed location.
- If GLOBAL_CONTEXT.actors is provided, choose scene actors only from those actor ids.
- If GLOBAL_CONTEXT.structured_locations is provided, choose location_id only from those location ids.
- If GLOBAL_CONTEXT.subject_mode is "single", every scene references exactly one actor: the first available actor id.
- Otherwise every scene references 1 to GLOBAL_CONTEXT.max_scene_actors actor ids, never more than 4.
- If prompt_guidance is provided, treat its categories as user interface values for this run.
- Shot types, character visibility, environments, lighting, camera motion, physical interaction, facial expression, outfit rules, prompt structure, list handling, and word count are guidance for visual continuity and variety.
- Follow explicit prompt_guidance values unless they conflict with segment type, subject identity, allowed locations, or the performance policy.
- Do not invent new characters, new locations, or new story events just to satisfy a guidance category.

Output rules:
- Return ONLY a valid JSON object.
- Each key must exactly match a segment_id.
- Do not omit keys. Do not add extra keys.
- Each value must be an object with this exact shape:
  {{"concept": "one concise visual concept string", "references": {{"actor_ids": ["actor_id"], "location_id": "location_id"}}}}
- references.actor_ids must contain at least 1 actor id and at most GLOBAL_CONTEXT.max_scene_actors actor ids.
- references.location_id must contain exactly one location id.
- If no actors or structured locations are available, fall back to a plain concept string for legacy compatibility.
- No markdown, no comments, no code fences.
""".strip()


def build_detail_system_prompt(label: str, *, segment_type: str = "", silent_mode: bool = False) -> str:
    normalized = label.strip()
    lower = normalized.lower()
    category_rule = "Keep the line limited to the requested label."
    if lower == "camera motion":
        category_rule = "For Camera Motion, output only camera movement phrases."
    elif lower == "character motion":
        category_rule = "For Character Motion, output only visible body movement or performance movement."
    elif lower == "lighting":
        category_rule = "For Lighting, output only lighting descriptions."
    elif lower == "weather":
        category_rule = "For Weather, output only weather descriptions."
    elif lower == "time of day":
        category_rule = "For Time of Day, output only time-of-day phrases."
    elif lower in {"emotion", "facial expression"}:
        category_rule = f"For {normalized}, output only the emotion or expression."

    return f"""
You create one visual prompt detail for a video workflow.

Input:
- A detail label: {normalized}
- One scene concept
- Segment type and performance policy

Task:
Create exactly one matching detail line for the requested label.

Rules:
- Output only the detail line.
- Keep the line short and specific.
- Follow the performance policy below.
- Do not combine multiple categories in one line.
- Do not repeat the full prompt.
- Avoid vague words like cinematic, beautiful, cool, stylish, dramatic, or interesting unless the label specifically asks for mood.
- If the prompt does not clearly imply a value, invent a simple value that fits the scene.
- {category_rule}

{performance_policy(segment_type, silent_mode=silent_mode)}
""".strip()


def build_video_payload(
    *,
    segment: dict[str, Any],
    concept: str,
    scene_details: dict[str, Any],
    global_context: dict[str, Any],
    t2i_prompt: str = "",
    custom_instructions: str = "",
) -> dict[str, Any]:
    segment_type = str(segment.get("type", "")).strip().lower()
    silent_mode = bool(global_context.get("silent_mode", False))
    return {
        "subject": global_context["subject"],
        "story_idea": global_context["story_idea"],
        "style": global_context["style"],
        "locations": global_context["locations"],
        "prompt_guidance": global_context.get("prompt_guidance", {}),
        "segment": segment,
        "performance_mode": segment_type,
        "silent_mode": silent_mode,
        "performance_policy": performance_policy(segment_type, silent_mode=silent_mode),
        "t2i_prompt": t2i_prompt,
        "scene_concept": concept,
        "camera_motion": scene_details.get("camera_motion", ""),
        "character_motion": scene_details.get("character_motion", ""),
        "custom_instructions": custom_instructions,
    }
