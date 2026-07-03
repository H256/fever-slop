import { describe, expect, test } from "bun:test";
import { choosePlaybackItem, previewStart } from "./reviewTimelinePlayback";
import type { TimelineItem } from "./reviewTimeline";

describe("review timeline playback helpers", () => {
  test("chooses the playable item under the scrubber first", () => {
    const items = [item({ scene: 1, start: 0, end: 4 }), item({ scene: 2, start: 4, end: 8 })];

    expect(choosePlaybackItem(items, 5, 1)?.scene).toBe(2);
  });

  test("falls back to selected then future then first playable item", () => {
    const items = [item({ scene: 1, start: 0, end: 4 }), item({ scene: 2, start: 10, end: 12 })];

    expect(choosePlaybackItem(items, 8, 1)?.scene).toBe(1);
    expect(choosePlaybackItem(items, 8, null)?.scene).toBe(2);
    expect(choosePlaybackItem(items, 20, null)?.scene).toBe(1);
  });

  test("uses raw start only when there is no final clip", () => {
    expect(previewStart(item({ finalClip: "final.mp4", rawStart: 7, start: 3 }))).toBe(3);
    expect(previewStart(item({ finalClip: "", rawStart: 7, start: 3 }))).toBe(7);
  });
});

function item(overrides: Partial<TimelineItem>): TimelineItem {
  return {
    scene: 0,
    start: 0,
    end: 0,
    duration: 0,
    rawStart: 0,
    rawEnd: 0,
    rawDuration: 0,
    finalClip: "",
    rawClip: "raw.mp4",
    clip: "raw.mp4",
    status: "raw",
    preview: "",
    hasManifestTiming: false,
    ...overrides
  };
}
