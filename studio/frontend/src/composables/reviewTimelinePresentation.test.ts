import { describe, expect, test } from "bun:test";
import { blockStyle, buildThumbnailRequests, formatTime, thumbnailFrameTimes, timelineTicks } from "./reviewTimelinePresentation";
import type { TimelineItem } from "./reviewTimeline";

describe("review timeline presentation helpers", () => {
  test("formats seconds as minute and two-digit second labels", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(65.9)).toBe("1:05");
  });

  test("builds timeline ticks with coarse steps for longer timelines", () => {
    expect(timelineTicks(20)).toEqual([0, 5, 10, 15, 20]);
    expect(timelineTicks(100)).toEqual([0, 15, 30, 45, 60, 75, 90, 100]);
    expect(timelineTicks(300)).toEqual([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]);
  });

  test("returns percentage block style relative to total duration", () => {
    expect(blockStyle(5, 10, 20)).toEqual({ left: "25%", width: "50%" });
    expect(blockStyle(5, -1, 0)).toEqual({ left: "500%", width: "0%" });
  });

  test("plans thumbnail frame times from duration and zoom", () => {
    expect(thumbnailFrameTimes(2, 1)).toEqual([0.4]);
    expect(thumbnailFrameTimes(20, 1)).toEqual([0, 4, 8, 12, 16, 20]);
    expect(thumbnailFrameTimes(40, 2)).toHaveLength(8);
  });

  test("deduplicates thumbnail requests across raw and final clips", () => {
    const items: TimelineItem[] = [
      item({ scene: 1, rawClip: "raw/a.mp4", finalClip: "final/a.mp4", duration: 20, rawDuration: 2 }),
      item({ scene: 2, rawClip: "raw/a.mp4", finalClip: "", duration: 5, rawDuration: 2 })
    ];

    expect(buildThumbnailRequests(items, 1)).toEqual([
      { path: "raw/a.mp4", times: [0.4] },
      { path: "final/a.mp4", times: [0, 4, 8, 12, 16, 20] }
    ]);
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
    rawClip: "",
    clip: "",
    status: "missing",
    preview: "",
    hasManifestTiming: false,
    ...overrides
  };
}
