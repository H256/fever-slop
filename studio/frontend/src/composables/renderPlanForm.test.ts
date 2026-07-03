import { describe, expect, test } from "bun:test";
import {
  collectRenderPlanFields,
  displayRenderPlanFieldValue,
  generationTargetsForField,
  groupRenderPlanFields,
  isAdvancedRenderPlanField,
  readPath,
  scenePreview,
  setPath,
  updateRenderPlanFieldValue
} from "./renderPlanForm";
import type { RenderScene } from "../types";

describe("render plan form helpers", () => {
  test("collects editable fields and excludes scene and reference selectors", () => {
    const fields = collectRenderPlanFields({
      scene: 1,
      references: { actor_ids: ["a"], location_id: "loc" },
      ltx: { base_prompt: "walk forward", fps: 24 },
      enabled: true,
      tags: ["hero", "wide"]
    });

    expect(fields.map((field) => field.path.join("."))).toEqual(["ltx.base_prompt", "ltx.fps", "enabled", "tags"]);
    expect(fields.find((field) => field.path.join(".") === "ltx.base_prompt")?.kind).toBe("shortText");
    expect(fields.find((field) => field.path.join(".") === "enabled")?.kind).toBe("boolean");
    expect(fields.find((field) => field.path.join(".") === "tags")?.kind).toBe("simpleArray");
  });

  test("groups fields by parent path and identifies advanced calculated fields", () => {
    const fields = collectRenderPlanFields({
      ltx: { base_prompt: "walk forward" },
      fps: 24,
      metadata: { lyrics: "line one\nline two" }
    });

    const groups = groupRenderPlanFields(fields);

    expect(groups.map((group) => ({ key: group.key, title: group.title, fields: group.fields.map((field) => field.path.join(".")) }))).toEqual([
      { key: "ltx", title: "ltx", fields: ["ltx.base_prompt"] },
      { key: "general", title: "General", fields: ["fps"] },
      { key: "metadata", title: "metadata", fields: ["metadata.lyrics"] }
    ]);
    expect(isAdvancedRenderPlanField(fields.find((field) => field.path.join(".") === "fps")!)).toBe(true);
    expect(isAdvancedRenderPlanField(fields.find((field) => field.path.join(".") === "ltx.base_prompt")!)).toBe(false);
  });

  test("updates draft values using typed field conversions", () => {
    const draft: Record<string, unknown> = { ltx: { base_prompt: "old" }, enabled: false, tags: [] };
    const fields = collectRenderPlanFields(draft);

    updateRenderPlanFieldValue(draft, fields.find((field) => field.path.join(".") === "ltx.base_prompt")!, "new prompt");
    updateRenderPlanFieldValue(draft, fields.find((field) => field.path.join(".") === "enabled")!, true);
    updateRenderPlanFieldValue(draft, fields.find((field) => field.path.join(".") === "tags")!, "wide, close, ");
    setPath(draft, ["references", "location_id"], "studio");

    expect(readPath(draft, ["ltx", "base_prompt"])).toBe("new prompt");
    expect(readPath(draft, ["enabled"])).toBe(true);
    expect(readPath(draft, ["tags"])).toEqual(["wide", "close"]);
    expect(readPath(draft, ["references", "location_id"])).toBe("studio");
  });

  test("reports display values generation targets and scene preview", () => {
    const scene: RenderScene = {
      scene: 3,
      z_image: { prompt: "portrait" },
      ltx: { base_prompt: "motion" },
      metadata: { base_concept: "concept" }
    };
    const fields = collectRenderPlanFields(scene);

    expect(displayRenderPlanFieldValue(fields.find((field) => field.path.join(".") === "z_image.prompt")!)).toBe("portrait");
    expect(generationTargetsForField(fields.find((field) => field.path.join(".") === "z_image.prompt")!)).toEqual(["image"]);
    expect(generationTargetsForField(fields.find((field) => field.path.join(".") === "metadata.base_concept")!)).toEqual(["image", "video"]);
    expect(scenePreview(scene)).toBe("motion");
  });
});
