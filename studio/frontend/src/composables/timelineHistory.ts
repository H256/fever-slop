import { ref, type Ref } from "vue";
import type { RenderScene } from "../types";

export function useTimelineHistory(scenes: Ref<RenderScene[]>, limit = 30) {
  const undoStack = ref<RenderScene[][]>([]);
  const redoStack = ref<RenderScene[][]>([]);

  function pushUndo() {
    undoStack.value.push(cloneScenes(scenes.value));
    if (undoStack.value.length > limit) undoStack.value.shift();
    redoStack.value = [];
  }

  function undoTimeline(): boolean {
    const previous = undoStack.value.pop();
    if (!previous) return false;
    redoStack.value.push(cloneScenes(scenes.value));
    scenes.value = cloneScenes(previous);
    return true;
  }

  function redoTimeline(): boolean {
    const next = redoStack.value.pop();
    if (!next) return false;
    undoStack.value.push(cloneScenes(scenes.value));
    scenes.value = cloneScenes(next);
    return true;
  }

  return {
    redoStack,
    redoTimeline,
    undoStack,
    undoTimeline,
    pushUndo
  };
}

function cloneScenes(scenes: RenderScene[]): RenderScene[] {
  return JSON.parse(JSON.stringify(scenes)) as RenderScene[];
}
