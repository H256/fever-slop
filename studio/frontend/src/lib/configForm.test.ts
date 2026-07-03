import { describe, expect, test } from "bun:test";
import { addArrayItem, ARRAY_TEMPLATES, mergeConfigDefaults, moveArrayItem, pruneConfigForSave, removeArrayItem, setPath } from "./configForm";

describe("mergeConfigDefaults", () => {
  test("adds supported config fields without overwriting existing values", () => {
    const config = mergeConfigDefaults({ project_name: "Demo", input_audio: "song.mp3", video: { fps: 30 } });

    expect(config.project_name).toBe("Demo");
    expect((config.video as Record<string, unknown>).fps).toBe(30);
    expect((config.video as Record<string, unknown>).width).toBe(1280);
    expect(config.subject_mode).toBe("multi");
    expect(config.max_scene_actors).toBe(4);
    expect(config.actors).toEqual([]);
    expect(config.loras).toEqual([]);
  });
});

describe("array helpers", () => {
  test("adds removes and moves object array items", () => {
    const config = mergeConfigDefaults({ project_name: "Demo", input_audio: "song.mp3" });

    addArrayItem(config, ["actors"], ARRAY_TEMPLATES.actors);
    addArrayItem(config, ["actors"], ARRAY_TEMPLATES.actors);
    setPath(config, ["actors", 0, "name"], "First");
    setPath(config, ["actors", 1, "name"], "Second");
    moveArrayItem(config, ["actors"], 1, -1);
    removeArrayItem(config, ["actors"], 1);

    expect(config.actors).toEqual([{ id: "", name: "Second", role: "", visual_description: "", image_prompt: "" }]);
  });
});

describe("pruneConfigForSave", () => {
  test("keeps runtime defaults and removes optional empty fields", () => {
    const config = mergeConfigDefaults({ project_name: "Demo", input_audio: "song.mp3" });

    const saved = pruneConfigForSave(config);

    expect(saved.project_name).toBe("Demo");
    expect(saved.input_audio).toBe("song.mp3");
    expect(saved.video).toEqual({ fps: 24, width: 1280, height: 704 });
    expect(saved.audio).toEqual({ demucs_model: "htdemucs_ft", whisper_model: "large", language: "de" });
    expect(saved.lyrics).toBeUndefined();
    expect(saved.story_idea).toBeUndefined();
    expect(saved.locations).toBeUndefined();
    expect(saved.actors).toBeUndefined();
    expect(saved.loras).toBeUndefined();
    expect(saved.lora_1).toBeUndefined();
  });

  test("keeps optional arrays and objects once they contain useful values", () => {
    const config = mergeConfigDefaults({ project_name: "Demo", input_audio: "song.mp3" });
    addArrayItem(config, ["actors"], ARRAY_TEMPLATES.actors);
    setPath(config, ["actors", 0, "name"], "Mara");
    setPath(config, ["story_idea"], "lost in neon");

    const saved = pruneConfigForSave(config);

    expect(saved.story_idea).toBe("lost in neon");
    expect(saved.actors).toEqual([{ name: "Mara" }]);
  });

  test("preserves unknown advanced fields even when empty", () => {
    const config = mergeConfigDefaults({
      project_name: "Demo",
      input_audio: "song.mp3",
      custom_plugin: { empty_but_intentional: "" }
    });

    const saved = pruneConfigForSave(config);

    expect(saved.custom_plugin).toEqual({ empty_but_intentional: "" });
  });
});
