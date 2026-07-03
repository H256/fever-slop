import { computed, type ComputedRef, type Ref } from "vue";
import type { ClipEdit } from "../lib/timelineTrim";
import type { RenderScene } from "../types";

export function useReviewTimelineEdits(scenes: Ref<RenderScene[]>, clipEdits: ComputedRef<ClipEdit[]>) {
  const staleScenes = computed(() =>
    scenes.value
      .filter((scene) => Boolean((scene.edit as Record<string, unknown> | undefined)?.studio_stale))
      .map((scene) => Number(scene.scene))
  );

  function sceneFor(sceneNumber: number): RenderScene | undefined {
    return scenes.value.find((scene) => Number(scene.scene) === sceneNumber);
  }

  function sceneFps(scene: RenderScene): number {
    return Number(scene.fps ?? 24) || 24;
  }

  function applyClipEdits(edits: ClipEdit[]) {
    for (const edit of edits) {
      const scene = sceneFor(edit.scene);
      if (!scene) continue;
      const fps = sceneFps(scene);
      scene.edit = {
        ...((scene.edit ?? {}) as Record<string, unknown>),
        raw_in_frame: edit.rawInFrame,
        raw_out_frame: edit.rawOutFrame,
        min_raw_in_frame: edit.minRawInFrame,
        max_raw_out_frame: edit.maxRawOutFrame,
        raw_in_seconds: edit.rawInFrame / fps,
        raw_out_seconds: edit.rawOutFrame / fps
      };
    }
  }

  function markChangedScenesStale(before: ClipEdit[], after: ClipEdit[]) {
    for (const edit of after) {
      const previous = before.find((candidate) => candidate.scene === edit.scene);
      if (!previous || (previous.rawInFrame === edit.rawInFrame && previous.rawOutFrame === edit.rawOutFrame)) continue;
      markScenesStale([edit.scene], "clip trim changed");
    }
  }

  function editSeconds(sceneNumber: number): { in: number; out: number } {
    const scene = sceneFor(sceneNumber);
    const edit = clipEdits.value.find((candidate) => candidate.scene === sceneNumber);
    const fps = scene ? sceneFps(scene) : 24;
    return { in: Number(edit?.rawInFrame ?? 0) / fps, out: Number(edit?.rawOutFrame ?? 0) / fps };
  }

  function markScenesStale(sceneNumbers: number[], reason: string) {
    for (const scene of scenes.value) {
      if (!sceneNumbers.includes(Number(scene.scene))) continue;
      scene.edit = { ...((scene.edit ?? {}) as Record<string, unknown>), studio_stale: true, studio_stale_reason: reason };
    }
  }

  function clearSceneStale(sceneNumber: number) {
    const scene = sceneFor(sceneNumber);
    if (!scene?.edit || typeof scene.edit !== "object") return;
    const edit = { ...(scene.edit as Record<string, unknown>) };
    delete edit.studio_stale;
    delete edit.studio_stale_reason;
    scene.edit = edit;
  }

  function isSceneStale(sceneNumber?: number): boolean {
    if (!sceneNumber) return false;
    return Boolean((sceneFor(sceneNumber)?.edit as Record<string, unknown> | undefined)?.studio_stale);
  }

  return {
    applyClipEdits,
    clearSceneStale,
    editSeconds,
    isSceneStale,
    markChangedScenesStale,
    markScenesStale,
    sceneFor,
    sceneFps,
    staleScenes
  };
}
