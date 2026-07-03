import { describe, expect, test } from "bun:test";
import { isTimelineMedia, rawPreviewForEdit, renderManifestByScene } from "./reviewTimelineMedia";
import type { ClipEdit } from "../lib/timelineTrim";
import type { TimelineItem } from "./reviewTimeline";
import type { RenderScene } from "../types";

describe("review timeline media helpers", () => {
  test("indexes render manifest rows by numeric scene id and ignores malformed rows", () => {
    expect(
      renderManifestByScene([
        { scene: "1", render_frame_count: 48 },
        null,
        { no_scene: true },
        { scene: 2, audio_start_seconds: 10 }
      ])
    ).toEqual({
      1: { scene: "1", render_frame_count: 48 },
      2: { scene: 2, audio_start_seconds: 10 }
    });
  });

  test("detects media already represented by timeline raw or final clips", () => {
    const items = [item({ rawClip: "raw/a.mp4", finalClip: "final/a.mp4" }), item({ rawClip: "raw/b.mp4" })];

    expect(isTimelineMedia("raw/a.mp4", items)).toBe(true);
    expect(isTimelineMedia("final/a.mp4", items)).toBe(true);
    expect(isTimelineMedia("other.png", items)).toBe(false);
  });

  test("builds raw preview metadata from a clip edit and scene fps", () => {
    const preview = rawPreviewForEdit({
      sceneNumber: 1,
      mode: "right",
      items: [item({ scene: 1, rawClip: "raw/a.mp4" })],
      scenes: [{ scene: 1, fps: 25 }],
      edits: [edit({ scene: 1, rawInFrame: 25, rawOutFrame: 75 })]
    });

    expect(preview).toEqual({ scene: 1, clip: "raw/a.mp4", seconds: 3, edge: "OUT" });
  });

  test("returns null when raw preview data is incomplete", () => {
    expect(
      rawPreviewForEdit({
        sceneNumber: 1,
        mode: "left",
        items: [item({ scene: 1, rawClip: "" })],
        scenes: [{ scene: 1 }],
        edits: [edit({ scene: 1 })]
      })
    ).toBeNull();
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

function edit(overrides: Partial<ClipEdit>): ClipEdit {
  return {
    scene: 0,
    rawInFrame: 0,
    rawOutFrame: 0,
    minRawInFrame: 0,
    maxRawOutFrame: 0,
    ...overrides
  };
}
