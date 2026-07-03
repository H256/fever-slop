import { describe, expect, test } from "bun:test";
import { buildTimelineItems, derivedFinalClip, findSceneClip } from "./reviewTimeline";
import type { RenderScene } from "../types";

describe("review timeline helpers", () => {
  test("finds final clips before raw clips and ignores debug renders", () => {
    const videos = [
      "output/video/_debug/scene_0001.mp4",
      "output/video/raw/scene_0001_raw.mp4",
      "output/video/final/scene_0001.mp4"
    ];

    expect(findSceneClip(videos, 1, false)).toBe("output/video/final/scene_0001.mp4");
    expect(findSceneClip(videos, 1, true)).toBe("output/video/raw/scene_0001_raw.mp4");
  });

  test("derives final clip path from raw clip path", () => {
    expect(derivedFinalClip("output/video/raw/scene_0007_raw.mp4", 7)).toBe("output/video/final/scene_0007.mp4");
  });

  test("builds timeline items from scenes videos and manifest timing", () => {
    const scenes: RenderScene[] = [
      { scene: 1, duration_seconds: 2, frame_count: 48, fps: 24, ltx: { base_prompt: "first" } },
      { scene: 2, duration_seconds: 3, frame_count: 72, fps: 24, metadata: { lyrics: "second" } }
    ];
    const items = buildTimelineItems({
      scenes,
      videos: ["output/video/final/scene_0001.mp4", "output/video/raw/scene_0002_raw.mp4"],
      manifest: {
        2: { scene: 2, audio_start_seconds: 10, audio_duration_seconds: 4, scene_frame_count: 72, render_frame_count: 96 }
      }
    });

    expect(items.map((item) => ({ scene: item.scene, start: item.start, duration: item.duration, status: item.status }))).toEqual([
      { scene: 1, start: 0, duration: 2, status: "final" },
      { scene: 2, start: 2, duration: 3, status: "raw" }
    ]);
    expect(items[1].rawStart).toBe(10);
    expect(items[1].rawDuration).toBe(4);
    expect(items[0].preview).toBe("first");
    expect(items[1].preview).toBe("second");
  });
});
