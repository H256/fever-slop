# Continuous MiniMax H3 R2V scenes

A semantic scene can exceed one workflow invocation's duration limit. The
render-plan stage splits it into a serial continuation chain. The LLM can
describe a continuous action, but it does not choose technical clip lengths
or extend the scene's audio interval. Separate semantic scenes are not merged.

Rebuild the render plan after changing the scene timeline or workflow profile.
The existing render and concat stages then consume the technical entries.
Scene selection uses the original semantic scene number; selecting a technical
successor also includes its required predecessors.

## Duration limits

The selected final workflow profile's `duration_capability` takes precedence.
When it is absent, R2V uses the existing H3 backend contract: 24 fps, a 4-15
second generation window, and the `17N+5` frame raster. The largest valid frame
count within that window is 345, so a scene near the nominal 15-second limit
may already require splitting. Profiles must match the project's timeline FPS.

For a workflow with a different validated generation budget, configure its
`duration_capability` in `app_config.json`, for example:

```json
{
  "fps": 24,
  "min_seconds": 4,
  "max_seconds": 12,
  "preferred_seconds": 8,
  "frame_alignment": 17,
  "frame_offset": 5
}
```

Semantic scene generation retains the requested scene-duration range for R2V;
the backend limits constrain technical segments. Each scene produces one chain
covering its existing absolute interval, even if the LLM emits multiple action
intents or requests a different action duration.

## Frame and audio accounting

- `semantic_scene` and `semantic_segment_id` identify the original scene.
- Technical segment IDs are deterministic and scoped to that semantic identity.
- `frame_count` is the segment's contribution to the finished timeline.
- `anchor_frames` reserves one predecessor boundary frame for each successor.
- `render_frame_count` includes the anchor and rounds up to a valid model frame
  count. Postprocessing removes the excess tail padding, retaining the anchor.
- Audio windows use absolute source timestamps. A successor starts its model
  audio window one frame earlier for the anchor; its semantic end stays fixed.

Canonical prompt relays are clipped into segment-local frame ranges. Existing
manual prompt overrides and enriched actor/location references are carried
into the technical entries when regenerating a selected semantic scene.

## Execution, resume, and assembly

The renderer extracts and fingerprints the predecessor's last frame before
submitting a successor. R2V binds the verified image as its continuity reference.
Resume reuses a clip only when its frame count and boundary evidence are current.
A changed predecessor or damaged clip triggers the affected downstream work.

The base concat stage checks segment boundaries, removes a proven duplicate
successor frame, and verifies that each new continuation segment contributes
exactly its planned timeline frame count. It adds no crossfade. The original
song is muxed after video-only assembly.

H3 reference conditioning does not guarantee an identical generated first frame.
The default strict boundary policy rejects an unproven connection; it cannot
guarantee a visually seamless result from an arbitrary model response. Diagnostics
are written beside successful cutless assemblies. A real FFmpeg regression covers
frame extraction, cache reuse, duplicate removal, and exact final frame count;
ComfyUI/GPU visual continuity still requires a real render.
