import { computed, type Ref } from "vue";
import type { RenderScene } from "../types";

export type PathPart = string | number;
export type RenderPlanFieldKind = "boolean" | "number" | "shortText" | "longText" | "simpleArray";
export type GenerationTarget = "image" | "video";

export interface RenderPlanField {
  path: PathPart[];
  label: string;
  kind: RenderPlanFieldKind;
  value: unknown;
  help: string;
}

export interface RenderPlanFormGroup {
  key: string;
  title: string;
  path: PathPart[];
  fields: RenderPlanField[];
}

export function useRenderPlanForm(draft: Ref<Record<string, unknown>>) {
  const formFields = computed(() => collectRenderPlanFields(draft.value));
  const visibleFormGroups = computed(() => groupRenderPlanFields(formFields.value.filter((field) => !isAdvancedRenderPlanField(field))));
  const advancedFormGroups = computed(() => groupRenderPlanFields(formFields.value.filter(isAdvancedRenderPlanField)));

  return {
    advancedFormGroups,
    formFields,
    visibleFormGroups
  };
}

export function collectRenderPlanFields(value: unknown, path: PathPart[] = []): RenderPlanField[] {
  if (path.length === 1 && path[0] === "scene") return [];
  if (path.join(".") === "references.actor_ids" || path.join(".") === "references.location_id") return [];
  if (typeof value === "boolean") return [field(path, "boolean", value)];
  if (typeof value === "number") return [field(path, "number", value)];
  if (typeof value === "string") return [field(path, value.length > 90 || value.includes("\n") ? "longText" : "shortText", value)];
  if (Array.isArray(value)) {
    if (value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
      return [field(path, "simpleArray", value)];
    }
    return value.flatMap((item, index) => collectRenderPlanFields(item, [...path, index]));
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => collectRenderPlanFields(child, [...path, key]));
  }
  return [];
}

export function groupRenderPlanFields(fields: RenderPlanField[]): RenderPlanFormGroup[] {
  const groups = new Map<string, RenderPlanFormGroup>();
  for (const field of fields) {
    const groupPath = field.path.length > 1 ? field.path.slice(0, -1) : [];
    const key = groupPath.join(".") || "general";
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        path: groupPath,
        title: groupPath.length ? labelForPath(groupPath) : "General",
        fields: []
      });
    }
    groups.get(key)?.fields.push(field);
  }
  return [...groups.values()];
}

export function fieldLabel(field: RenderPlanField, group: RenderPlanFormGroup): string {
  return labelForPath(field.path.slice(group.path.length));
}

export function updateRenderPlanFieldValue(target: Record<string, unknown>, field: RenderPlanField, rawValue: string | number | boolean) {
  let value: unknown = rawValue;
  if (field.kind === "number") value = Number(rawValue);
  if (field.kind === "simpleArray") {
    value = String(rawValue)
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  setPath(target, field.path, value);
}

export function setPath(target: Record<string, unknown>, path: PathPart[], value: unknown) {
  let current: unknown = target;
  for (const [index, part] of path.slice(0, -1).entries()) {
    const parent = current as Record<string, unknown> & Record<number, unknown>;
    if (parent[part] === undefined) parent[part] = typeof path[index + 1] === "number" ? [] : {};
    current = parent[part];
  }
  (current as Record<string, unknown> & Record<number, unknown>)[path[path.length - 1]] = value;
}

export function readPath(value: unknown, path: string[]): unknown {
  let current = value;
  for (const part of path) {
    if (!current || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

export function displayRenderPlanFieldValue(field: RenderPlanField): string {
  return Array.isArray(field.value) ? field.value.join(", ") : String(field.value ?? "");
}

export function isAdvancedRenderPlanField(field: RenderPlanField): boolean {
  const name = String(field.path.at(-1));
  return new Set([
    "abs_start_seconds",
    "abs_end_seconds",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "frame_count",
    "frame",
    "frame_start",
    "frame_end",
    "cut"
  ]).has(name);
}

export function generationTargetsForField(field: RenderPlanField): GenerationTarget[] {
  const key = field.path.join(".");
  const name = String(field.path.at(-1));
  if (key === "z_image.prompt" || key === "ltx.t2i_prompt" || name === "frame") return ["image"];
  if (key.startsWith("ltx.") || name === "frame_start" || name === "frame_end" || name === "frame_count") return ["video"];
  if (key === "metadata.base_concept" || key === "metadata.camera_motion" || key === "metadata.character_motion") return ["image", "video"];
  if (["fps", "width", "height", "duration_seconds"].includes(name)) return ["image", "video"];
  return [];
}

export function optionsForRenderPlanField(field: RenderPlanField): string[] {
  const key = field.path.join(".");
  const name = String(field.path.at(-1));
  const options: Record<string, string[]> = {
    "ltx.render_mode_hint": ["single_prompt", "relay", "auto"],
    "metadata.type": ["vocals", "instrumental"],
    state: ["vocals", "instrumental"]
  };
  return options[key] ?? options[name] ?? [];
}

export function scenePreview(scene: RenderScene): string {
  return String(readPath(scene, ["ltx", "base_prompt"]) ?? readPath(scene, ["z_image", "prompt"]) ?? readPath(scene, ["metadata", "lyrics"]) ?? "");
}

function field(path: PathPart[], kind: RenderPlanFieldKind, value: unknown): RenderPlanField {
  return { path, kind, value, label: labelForPath(path), help: helpForRenderField(path) };
}

function labelForPath(path: PathPart[]): string {
  return path
    .map((part) => (typeof part === "number" ? `#${part + 1}` : part.replaceAll("_", " ")))
    .join(" / ");
}

function helpForRenderField(path: PathPart[]): string {
  const key = path.join(".");
  const name = String(path.at(-1));
  const descriptions: Record<string, string> = {
    "z_image.prompt": "Required. Affects image generation. Prompt used to create the scene start frame.",
    "ltx.base_prompt": "Required. Affects video generation. Main video prompt for this scene.",
    "ltx.t2i_prompt": "Optional. Affects image/video generation if this plan is reused. Image prompt passed forward into video generation.",
    "ltx.i2v_prompt_from_t2i": "Optional. Affects video generation. Motion prompt derived from the image prompt.",
    "ltx.original_style_i2v_prompt": "Optional. Does not affect generation unless copied into the active prompt. Original motion/style prompt kept for comparison.",
    "ltx.render_mode_hint": "Optional. Does not directly affect generation. Records which video render mode produced this scene.",
    "metadata.lyrics": "Optional. Affects regeneration context. Lyrics or transcript text associated with this scene.",
    "metadata.base_concept": "Required for regeneration. Affects future prompt generation. Scene concept used by prompt builders.",
    "metadata.camera_motion": "Optional. Affects future prompt generation. Camera movement guidance for the scene.",
    "metadata.character_motion": "Optional. Affects future prompt generation. Character movement guidance for the scene.",
    "metadata.segment_id": "Calculated. Does not affect generation directly. Links the scene back to the timeline segment.",
    "metadata.type": "Calculated. Affects regeneration context. Timeline segment type such as vocals or instrumental."
  };
  const byName: Record<string, string> = {
    prompt: "Required. Affects generation for the frame range or stage that consumes it.",
    state: "Calculated. Does not affect generation directly. Prompt relay state for this frame range.",
    frame_start: "Calculated. Affects video generation timing if changed. First frame covered by this relay prompt.",
    frame_end: "Calculated. Affects video generation timing if changed. Last frame covered by this relay prompt.",
    abs_start_seconds: "Calculated. Affects final timing if changed. Scene start time in the source audio.",
    abs_end_seconds: "Calculated. Affects final timing if changed. Scene end time in the source audio.",
    duration_seconds: "Calculated. Affects generated clip length if changed. Scene duration.",
    fps: "Calculated from project video settings. Affects frame counts and render timing if changed.",
    width: "Calculated from project video settings. Affects generation resolution if changed.",
    height: "Calculated from project video settings. Affects generation resolution if changed.",
    frame_count: "Calculated from duration and FPS. Affects generated clip length if changed.",
    frame: "Calculated. Affects image generation frame selection if changed.",
    cut: "Optional. Affects whether this render-plan entry is treated as a cut."
  };
  return descriptions[key] ?? byName[name] ?? "Optional. May affect later regeneration if a pipeline step consumes this value.";
}
