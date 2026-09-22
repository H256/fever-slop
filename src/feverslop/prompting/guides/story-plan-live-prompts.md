# Story Plan Live Prompts

Write live image and video prompts for the supplied segments/shots. The
explicit user direction (`creative_direction`) is the highest-priority input:
when it conflicts with the bible or the brief creative direction, the user
direction wins.

Return:

- `prompts`: exactly ONE entry per supplied segment/shot id (`target`). Each
  entry carries:
  - `image_prompt`: a concrete, self-contained text-to-image prompt for the
    scene's keyframe. Describe the frame's composition, subjects, location,
    lighting, and mood in concrete visual terms.
  - `video_prompt`: a concrete, self-contained text-to-video prompt describing
    the motion and camera behavior for the scene. Describe what moves, how the
    camera moves, and the temporal arc of the shot.

Hard constraints:

- Return exactly the supplied `expected_targets`; never omit one or invent a
  new target.
- Reference ONLY the supplied canonical character, location, and prop ids;
  never invent a new id.
- Keep each prompt self-contained: a downstream model must be able to render it
  without reading any other field.
- No timestamps, frame counts, durations, or render settings (resolution, fps,
  aspect ratio, sampler, seed).
- No lyrics or audio bindings; the image and video prompts are visual only.
