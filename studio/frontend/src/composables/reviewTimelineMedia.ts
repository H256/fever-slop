import type { ClipEdit } from "../lib/timelineTrim";
import type { RenderScene } from "../types";
import type { RenderManifestEntry, TimelineItem } from "./reviewTimeline";

export interface RawPreview {
  scene: number;
  clip: string;
  seconds: number;
  edge: "IN" | "OUT";
}

export interface RawPreviewInput {
  sceneNumber: number;
  mode: "left" | "right";
  items: TimelineItem[];
  scenes: RenderScene[];
  edits: ClipEdit[];
}

export function renderManifestByScene(data: unknown): Record<number, RenderManifestEntry> {
  if (!Array.isArray(data)) return {};
  return Object.fromEntries(
    data
      .filter((entry): entry is RenderManifestEntry => Boolean(entry) && typeof entry === "object" && "scene" in entry)
      .map((entry) => [Number(entry.scene), entry])
  );
}

export function isTimelineMedia(path: string, items: TimelineItem[]): boolean {
  return items.some((item) => item.finalClip === path || item.rawClip === path);
}

export function rawPreviewForEdit(input: RawPreviewInput): RawPreview | null {
  const item = input.items.find((candidate) => candidate.scene === input.sceneNumber);
  const scene = input.scenes.find((candidate) => Number(candidate.scene) === input.sceneNumber);
  const edit = input.edits.find((candidate) => candidate.scene === input.sceneNumber);
  if (!item?.rawClip || !scene || !edit) return null;
  const seconds = (input.mode === "left" ? edit.rawInFrame : edit.rawOutFrame) / sceneFps(scene);
  return { scene: input.sceneNumber, clip: item.rawClip, seconds, edge: input.mode === "left" ? "IN" : "OUT" };
}

function sceneFps(scene: RenderScene): number {
  return Number(scene.fps ?? 24) || 24;
}
