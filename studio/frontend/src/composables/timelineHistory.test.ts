import { describe, expect, test } from "bun:test";
import { ref } from "vue";
import { useTimelineHistory } from "./timelineHistory";
import type { RenderScene } from "../types";

describe("useTimelineHistory", () => {
  test("restores cloned scene snapshots for undo and redo", () => {
    const scenes = ref<RenderScene[]>([{ scene: 1, duration_seconds: 1 }]);
    const history = useTimelineHistory(scenes);

    history.pushUndo();
    scenes.value[0].duration_seconds = 2;

    expect(history.undoTimeline()).toBe(true);
    expect(scenes.value[0].duration_seconds).toBe(1);

    scenes.value[0].duration_seconds = 9;
    expect(history.redoTimeline()).toBe(true);
    expect(scenes.value[0].duration_seconds).toBe(2);
  });

  test("limits undo snapshots and clears redo after a new branch", () => {
    const scenes = ref<RenderScene[]>([{ scene: 1, duration_seconds: 0 }]);
    const history = useTimelineHistory(scenes, 2);

    for (let value = 1; value <= 3; value += 1) {
      history.pushUndo();
      scenes.value[0].duration_seconds = value;
    }

    expect(history.undoStack.value).toHaveLength(2);
    expect(history.undoTimeline()).toBe(true);
    expect(history.redoStack.value).toHaveLength(1);
    history.pushUndo();
    expect(history.redoStack.value).toHaveLength(0);
  });
});
