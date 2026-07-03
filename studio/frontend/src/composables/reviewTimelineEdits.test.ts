import { describe, expect, test } from "bun:test";
import { computed, ref } from "vue";
import { useReviewTimelineEdits } from "./reviewTimelineEdits";
import type { ClipEdit } from "../lib/timelineTrim";
import type { RenderScene } from "../types";

function setup(scenesInput?: RenderScene[], editsInput?: ClipEdit[]) {
  const scenes = ref<RenderScene[]>(
    scenesInput ?? [
      { scene: 1, fps: 24, edit: { keep: true } },
      { scene: 2, fps: 30 }
    ]
  );
  const edits = ref<ClipEdit[]>(
    editsInput ?? [
      { scene: 1, rawInFrame: 24, rawOutFrame: 72, minRawInFrame: 0, maxRawOutFrame: 96 },
      { scene: 2, rawInFrame: 0, rawOutFrame: 60, minRawInFrame: 0, maxRawOutFrame: 90 }
    ]
  );
  return {
    scenes,
    edits,
    timeline: useReviewTimelineEdits(scenes, computed(() => edits.value))
  };
}

describe("useReviewTimelineEdits", () => {
  test("applies clip edit frames and derived seconds to matching scenes", () => {
    const { scenes, timeline } = setup();

    timeline.applyClipEdits([{ scene: 1, rawInFrame: 48, rawOutFrame: 96, minRawInFrame: 12, maxRawOutFrame: 120 }]);

    expect(scenes.value[0].edit).toEqual({
      keep: true,
      raw_in_frame: 48,
      raw_out_frame: 96,
      min_raw_in_frame: 12,
      max_raw_out_frame: 120,
      raw_in_seconds: 2,
      raw_out_seconds: 4
    });
    expect(scenes.value[1].edit).toBeUndefined();
  });

  test("returns edit seconds using scene fps and falls back to 24 fps", () => {
    const { timeline } = setup([{ scene: 1 }, { scene: 2, fps: 30 }]);

    expect(timeline.editSeconds(1)).toEqual({ in: 1, out: 3 });
    expect(timeline.editSeconds(2)).toEqual({ in: 0, out: 2 });
  });

  test("marks only scenes whose trim changed as stale", () => {
    const { scenes, timeline } = setup();
    const before: ClipEdit[] = [
      { scene: 1, rawInFrame: 24, rawOutFrame: 72, minRawInFrame: 0, maxRawOutFrame: 96 },
      { scene: 2, rawInFrame: 0, rawOutFrame: 60, minRawInFrame: 0, maxRawOutFrame: 90 }
    ];
    const after: ClipEdit[] = [
      { scene: 1, rawInFrame: 24, rawOutFrame: 72, minRawInFrame: 0, maxRawOutFrame: 96 },
      { scene: 2, rawInFrame: 6, rawOutFrame: 60, minRawInFrame: 0, maxRawOutFrame: 90 }
    ];

    timeline.markChangedScenesStale(before, after);

    expect(timeline.staleScenes.value).toEqual([2]);
    expect(timeline.isSceneStale(1)).toBe(false);
    expect(timeline.isSceneStale(2)).toBe(true);
    expect((scenes.value[1].edit as Record<string, unknown>).studio_stale_reason).toBe("clip trim changed");
  });

  test("clears stale flags without dropping unrelated edit fields", () => {
    const { scenes, timeline } = setup([
      { scene: 1, edit: { studio_stale: true, studio_stale_reason: "old", raw_in_frame: 10 } }
    ]);

    timeline.clearSceneStale(1);

    expect(scenes.value[0].edit).toEqual({ raw_in_frame: 10 });
    expect(timeline.isSceneStale(1)).toBe(false);
  });
});
