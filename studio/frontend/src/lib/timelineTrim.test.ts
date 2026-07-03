import { describe, expect, test } from "bun:test";
import { applyBoundaryTrim, buildEditState, type ClipEdit } from "./timelineTrim";

const clips: ClipEdit[] = [
  { scene: 1, rawInFrame: 50, rawOutFrame: 150, minRawInFrame: 0, maxRawOutFrame: 175 },
  { scene: 2, rawInFrame: 50, rawOutFrame: 150, minRawInFrame: 0, maxRawOutFrame: 175 },
  { scene: 3, rawInFrame: 50, rawOutFrame: 150, minRawInFrame: 0, maxRawOutFrame: 175 }
];

describe("applyBoundaryTrim", () => {
  test("extending a clip right borrows frames from the next clip start", () => {
    const next = applyBoundaryTrim(clips, { scene: 1, edge: "right", deltaFrames: 10 });

    expect(next[0].rawOutFrame).toBe(160);
    expect(next[1].rawInFrame).toBe(60);
    expect(next.map((clip) => clip.rawOutFrame - clip.rawInFrame).reduce((a, b) => a + b, 0)).toBe(300);
  });

  test("extending a clip left borrows frames from the previous clip end", () => {
    const next = applyBoundaryTrim(clips, { scene: 2, edge: "left", deltaFrames: -12 });

    expect(next[1].rawInFrame).toBe(38);
    expect(next[0].rawOutFrame).toBe(138);
    expect(next.map((clip) => clip.rawOutFrame - clip.rawInFrame).reduce((a, b) => a + b, 0)).toBe(300);
  });

  test("shortening a clip right gives frames to the next clip when source is available", () => {
    const next = applyBoundaryTrim(clips, { scene: 1, edge: "right", deltaFrames: -8 });

    expect(next[0].rawOutFrame).toBe(142);
    expect(next[1].rawInFrame).toBe(42);
  });

  test("clamps to source bounds and never creates negative durations", () => {
    const next = applyBoundaryTrim(clips, { scene: 1, edge: "right", deltaFrames: 500 });

    expect(next[0].rawOutFrame).toBe(175);
    expect(next[1].rawInFrame).toBe(75);
    expect(next.every((clip) => clip.rawOutFrame > clip.rawInFrame)).toBe(true);
  });
});

describe("buildEditState", () => {
  test("derives source bounds from render manifest trim metadata", () => {
    const edit = buildEditState({ scene: 4, frameCount: 100, trimFrontFrames: 24, tailFrames: 12 });

    expect(edit).toEqual({ scene: 4, rawInFrame: 24, rawOutFrame: 124, minRawInFrame: 0, maxRawOutFrame: 136 });
  });
});
