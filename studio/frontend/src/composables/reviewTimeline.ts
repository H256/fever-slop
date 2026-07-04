import { buildEditState, type ClipEdit } from "../lib/timelineTrim";
import type { RenderScene } from "../types";

export interface TimelineItem {
  scene: number;
  start: number;
  end: number;
  duration: number;
  rawStart: number;
  rawEnd: number;
  rawDuration: number;
  finalClip: string;
  rawClip: string;
  clip: string;
  status: "final" | "raw" | "missing";
  preview: string;
  hasManifestTiming: boolean;
}

export interface RenderManifestEntry {
  scene: number;
  audio_start_seconds?: number;
  audio_duration_seconds?: number;
  trim_front_frames?: number;
  scene_frame_count?: number;
  render_frame_count?: number;
  tail_loss_frames?: number;
}

export interface BuildTimelineItemsInput {
  scenes: RenderScene[];
  videos: string[];
  manifest: Record<number, RenderManifestEntry>;
}

export function parseReviewRenderPlanScenes(data: unknown): RenderScene[] {
  const rawScenes = Array.isArray(data)
    ? data
    : data && typeof data === "object" && Array.isArray((data as Record<string, unknown>).shots)
      ? ((data as Record<string, unknown>).shots as unknown[])
      : data && typeof data === "object" && Array.isArray((data as Record<string, unknown>).scenes)
        ? ((data as Record<string, unknown>).scenes as unknown[])
        : [];
  return rawScenes.map((value, index) => {
    const scene = value && typeof value === "object" ? { ...(value as Record<string, unknown>) } : {};
    scene.scene = Number(scene.scene ?? index + 1);
    return scene as RenderScene;
  });
}

export function buildTimelineItems(input: BuildTimelineItemsInput): TimelineItem[] {
  const edits = input.scenes.map((scene) => buildClipEdit(scene, input.manifest[Number(scene.scene)]));
  return input.scenes.map((scene, index) => {
    const sceneNumber = Number(scene.scene);
    const edit = edits[index];
    const fps = sceneFps(scene);
    const start = edits.slice(0, index).reduce((total, clip, clipIndex) => total + (clip.rawOutFrame - clip.rawInFrame) / sceneFps(input.scenes[clipIndex]), 0);
    const duration = edit ? (edit.rawOutFrame - edit.rawInFrame) / fps : Number(scene.duration_seconds ?? 0);
    const manifest = input.manifest[sceneNumber];
    const fallbackRaw = fallbackRawTiming(scene, start, duration);
    const rawStart = Number(manifest?.audio_start_seconds ?? fallbackRaw.start);
    const rawDuration = Number(manifest?.audio_duration_seconds ?? fallbackRaw.duration);
    const finalClip = findSceneClip(input.videos, sceneNumber, false);
    const rawClip = findSceneClip(input.videos, sceneNumber, true);
    const clip = finalClip || rawClip;
    return {
      scene: sceneNumber,
      start,
      end: start + duration,
      duration,
      rawStart,
      rawEnd: rawStart + rawDuration,
      rawDuration,
      finalClip,
      rawClip,
      clip,
      status: finalClip ? "final" : rawClip ? "raw" : "missing",
      preview: scenePreview(scene),
      hasManifestTiming: Boolean(manifest)
    };
  });
}

export function buildClipEdit(scene: RenderScene, manifest?: RenderManifestEntry): ClipEdit {
  const fps = sceneFps(scene);
  const sceneStart = Number(scene.abs_start_seconds ?? 0);
  const sceneDuration = Number(scene.duration_seconds ?? 0);
  const rawTiming = fallbackRawTiming(scene, sceneStart, sceneDuration);
  const frameCount = Number(manifest?.scene_frame_count ?? scene.frame_count ?? Math.round(fps * sceneDuration));
  const trimFrontFrames = Number(manifest?.trim_front_frames ?? Math.max(0, Math.round((sceneStart - rawTiming.start) * fps)));
  const fallbackRenderFrameCount = Math.max(trimFrontFrames + frameCount, Math.round(rawTiming.duration * fps));
  const renderFrameCount = Number(manifest?.render_frame_count ?? fallbackRenderFrameCount);
  const explicitTailFrames = readPath(scene, ["rolling", "tail_loss_frames"]) ?? readPath(scene, ["ltx", "tail_loss_frames"]);
  const tailFrames = Math.max(0, Number(manifest?.tail_loss_frames ?? explicitTailFrames ?? renderFrameCount - trimFrontFrames - frameCount));
  const base = buildEditState({
    scene: Number(scene.scene),
    frameCount,
    trimFrontFrames,
    tailFrames
  });
  const edit = (scene.edit ?? {}) as Record<string, unknown>;
  return {
    ...base,
    rawInFrame: Number(edit.raw_in_frame ?? base.rawInFrame),
    rawOutFrame: Number(edit.raw_out_frame ?? base.rawOutFrame)
  };
}

export function findSceneClip(videos: string[], sceneNumber: number, raw: boolean): string {
  const padded = String(sceneNumber).padStart(4, "0");
  const candidates = videos.filter((path) => {
    if (!path.includes(`/scene_${padded}`)) return false;
    if (path.includes("_debug/")) return false;
    return raw ? path.includes("_raw") || path.includes("/raw/") : !path.includes("_raw") && !path.includes("/raw/");
  });
  return candidates.find((path) => path.includes("/final/")) ?? candidates[0] ?? "";
}

export function derivedFinalClip(rawClip: string, sceneNumber: number): string {
  const padded = String(sceneNumber).padStart(4, "0");
  if (rawClip.includes("/raw/")) return rawClip.replace("/raw/", "/final/").replace(`scene_${padded}_raw`, `scene_${padded}`);
  return rawClip.replace(`scene_${padded}_raw`, `scene_${padded}`);
}

export function fallbackRawTiming(scene: RenderScene, start: number, duration: number): { start: number; duration: number } {
  const fps = sceneFps(scene);
  const edit = (scene.edit ?? {}) as Record<string, unknown>;
  const editedOut = Number(edit.raw_out_seconds ?? 0);
  const frameCount = Number(scene.frame_count ?? 0);
  const renderFrameCount = Number(readPath(scene, ["rolling", "render_frame_count"]) ?? readPath(scene, ["ltx", "render_frame_count"]) ?? 0);
  const trimFrontFrames = Number(readPath(scene, ["rolling", "trim_front_frames"]) ?? readPath(scene, ["ltx", "trim_front_frames"]) ?? 0);
  if (renderFrameCount > frameCount) {
    return { start: Math.max(0, start - trimFrontFrames / fps), duration: renderFrameCount / fps };
  }
  if (editedOut > duration) return { start, duration: editedOut };
  return { start: Math.max(0, start - Math.min(2, start)), duration: duration + Math.min(2, start) + 1 };
}

function scenePreview(scene: RenderScene): string {
  return String(
    readPath(scene, ["ltx", "base_prompt"]) ??
      readPath(scene, ["ltx", "original_style_i2v_prompt"]) ??
      scene.description ??
      scene.action ??
      readPath(scene, ["z_image", "prompt"]) ??
      readPath(scene, ["metadata", "lyrics"]) ??
      ""
  );
}

function sceneFps(scene: RenderScene): number {
  return Number(scene.fps ?? 24) || 24;
}

function readPath(value: unknown, path: string[]): unknown {
  let current = value;
  for (const part of path) {
    if (!current || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}
