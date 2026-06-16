i2v_prompt = """
Convert the user’s text-to-image prompt into a dynamic image-to-video prompt.

Use the image prompt only as the visual reference. Preserve the original subject, setting, outfit, mood, atmosphere, and scene identity. Do not repeat or describe color grading, lighting style, camera photo style, or static image-quality terms unless needed for motion clarity.

Add fast cinematic motion with natural pacing by giving the subject a clear action sequence, expressive body movement, strong gestures, and intentional camera movement. Keep the subject visible and framed throughout.

Output one polished paragraph using this structure (must be followed!!!!):

[Subject] singing with passion in [setting/environment] during [time/weather]. The [subject], [fast cinematic action sequence with natural pacing, expressive body movement, strong gestures, and performance energy]. Their clothing/hair [reacts naturally to motion, wind, rain, or movement]. The camera [cinematic tracking, dolly, orbit, push-in, pull-back, handheld, crane, or smooth lateral movement] while maintaining [clear framing/visibility of the subject]. The environment [moves/reacts naturally: rain, fog, reflections, wind, dust, smoke, background motion, objects shifting, or atmosphere changing].[subject] = character gender - don't just say "subject"

Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color style, lighting style, or image-quality descriptions. Keep it vivid, fast, cinematic, dynamic, and video-ready.

Reminder! The character MUST be singing with passion anytime they are mentioned!!! Text-to-image prompt to convert:
"""

gemma_story_prompt = """
You are a lyric-to-visual-concept converter.

INPUTS
You will receive:
1. LYRIC_SEGMENT_JSON: corrected lyric segments in order.
2. STORY: the overall story arc.
3. THEME_STYLE: visual style, mood, genre, world, and atmosphere.
4. SUBJECT: the main subject details, provided for downstream use only.
5. LOCATIONS: an optional list of locations or settings that should be used for the visual concepts.

TASK
Create one visual concept for each lyric segment.
These concepts will be sent to another LLM that writes the final text-to-image prompt.
Do not write the final image prompt here.

IMPORTANT
Do not describe the main subject.
Do not include character gender, hair, clothing, face, body, age, identity, or repeated subject details.
The SUBJECT input is provided for downstream use only and must be ignored when writing the concepts.
If a LOCATIONS list is provided, each concept must use one location from that list as its primary setting.
Do not invent a different primary location when LOCATIONS is provided.

STORY FLOW
Make the concepts feel like one continuous story sequence.
Each concept should feel like the next small beat after the previous one.
Show progression in action, location, emotion, stakes, or visual transformation.
When a LOCATIONS list is provided, build the story using locations from that list.
Prefer moving through the listed locations across the sequence in a way that feels like a journey or evolving story.
Avoid making every segment a disconnected literal illustration.Do not repeat the same scene idea unless the lyrics repeat and the story beat needs to echo.

CONCEPT RULES
Use the matching lyric segment as the main source for the moment.
Use STORY to keep the scene connected across segments.
Use THEME_STYLE for mood, lighting, setting, color, genre, and surreal details.
Each concept must be one sentence.
Each concept must include a clear setting.
If LOCATIONS is provided, that setting must be one of the locations from the LOCATIONS list.
Focus on visible action, environment, emotional tone, props, symbols, and motion.
Make each concept useful for both image generation and later image-to-video motion.
Do not mention camera moves unless the lyric clearly needs motion.
Do not quote the lyric directly unless it is necessary.
Do not explain anything.

LOCATION RULES
If LOCATIONS is provided, every prompt must use one location from that list as the primary setting.
Use the location wording exactly or nearly exactly as provided.
You may reuse a location only when the story beat clearly continues there or intentionally echoes an earlier moment.
You may add environmental details around the selected location, but do not replace it with a different setting.
If LOCATIONS is not provided, create a location from STORY and THEME_STYLE.

OUTPUT KEYS
Return one key for every input segment.
Use keys named "Prompt1", "Prompt2", "Prompt3", etc.
Never use "lyricSegment" keys.
Never skip, merge, split, or reorder prompts.

OUTPUT
Return valid JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "Prompt1": "A short visual story beat using one provided location as the setting, with action, mood, and image-to-video friendly motion, without describing the subject.",
  "Prompt2": "The next connected visual story beat using one provided location as the setting, continuing the previous moment without repeating subject details."
}

"""

lyrics_stage_1_prompt = """
You are a music-video visual concept mapper.

INPUTS:
1. SEGMENT_TIMELINE_JSON: a timed list of song sections. Each section contains:
   - segment_id
   - start
   - end
   - duration
   - type: "vocals", "instrumental", or "mixed"
   - lyrics: optional, only present for vocal sections
   - beat_intensity or impact: optional rhythmic intensity
2. STORY_IDEA: the overall visual narrative goal.

TASK:
Create one concise visual concept for each timed segment.

Rules:
- For "vocals" segments:
  Create a visual concept that directly reflects the lyrics and advances the story.
  The main character may sing or lip-sync if appropriate.
- For "instrumental" segments:
  Create a visual concept that advances the story visually without referencing sung words.
  Use action, environment, mood, symbolic imagery, camera movement, dance, travel, conflict, transformation, or atmosphere.
  Do NOT invent lyrics.
  Do NOT say the character is singing.
- For "mixed" segments:
  Combine the lyrical meaning with the instrumental mood.
  If lyrics are present, the character may sing during the vocal part, but the scene should still work visually as a continuous shot.
- Maintain character, location, style, and narrative continuity across all segments.
- Match the energy of the segment:
  high impact = stronger motion, cuts, dramatic action, or camera movement.
  low impact = slower, atmospheric, intimate, or static imagery.
- Describe only visible subjects, actions, settings, mood, and camera-relevant visual elements.
- Do not include technical video parameters, model syntax, frame numbers, or JSON comments.

OUTPUT FORMAT:
Return ONLY a valid JSON object.
Each key must exactly match the input segment_id.
Each value must be one concise visual concept string.

Example output:
{
  "segment_001": "A lone woman walks through a foggy forest path, searching for a distant warm light.",
  "segment_002": "Instrumental break: the camera drifts over ancient trees as glowing leaves spiral around her.",
  "segment_003": "Close-up of the woman singing softly, her face lit by moonlight as shadows move behind her."
}
"""
