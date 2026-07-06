export type PathPart = string | number;

export const DEFAULT_CONFIG = {
  project_name: "",
  input_audio: "",
  silent_mode: false,
  lyrics: "",
  video: {
    fps: 24,
    width: 1280,
    height: 704
  },
  audio: {
    demucs_model: "htdemucs_ft",
    whisper_model: "large",
    language: "en"
  },
  video_pipeline: "ltx_msr",
  scene_generation: {
    min_duration: 2.0,
    max_duration: 10.0,
    bias: 0.7,
    duration_preset: "impact_weighted",
    seed: -1
  },
  vocal_detection: {
    merge_gap: 0.5,
    min_vocal_duration: 0.4,
    min_silence_duration: 0.8,
    rms_low_percentile: 20,
    rms_high_percentile: 85,
    rms_ratio: 0.35,
    smooth_frames: 10
  },
  story_idea: "",
  style: "",
  subject: "",
  subject_mode: "multi",
  max_scene_actors: 4,
  locations: [],
  actors: [],
  steering: {
    global: "",
    story_idea: "",
    style: "",
    subject: "",
    locations: "",
    concepts: "",
    zimage: "",
    ltx: "",
    final_prompts: ""
  },
  prompt_guidance: {
    character_visibility: "",
    shot_types: "",
    environments: "",
    lighting: "",
    camera_motion: "",
    physical_interaction: "",
    facial_expression: "",
    outfit_rules: "",
    prompt_structure: "",
    list_handling: "",
    word_count_min: 40,
    word_count_max: 50
  },
  lora_1: {
    enabled: false,
    name: "",
    strength_model: 1.0,
    strength_clip: 1.0
  },
  lora_split_enabled: false,
  loras: []
};

export const ARRAY_TEMPLATES: Record<string, unknown> = {
  locations: {
    id: "",
    name: "",
    visual_description: "",
    image_prompt: ""
  },
  actors: {
    id: "",
    name: "",
    role: "",
    visual_description: "",
    image_prompt: ""
  },
  loras: {
    enabled: false,
    name: "",
    strength_model: 1.0,
    strength_clip: 1.0
  }
};

export function mergeConfigDefaults(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return clone(DEFAULT_CONFIG) as Record<string, unknown>;
  return mergeDefaults(clone(DEFAULT_CONFIG), value) as Record<string, unknown>;
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

export function getPath(target: unknown, path: PathPart[]): unknown {
  return path.reduce((current, part) => (current as Record<string, unknown> | unknown[] | undefined)?.[part as never], target);
}

export function addArrayItem(target: Record<string, unknown>, path: PathPart[], template: unknown) {
  arrayAt(target, path).push(clone(template));
}

export function removeArrayItem(target: Record<string, unknown>, path: PathPart[], index: number) {
  arrayAt(target, path).splice(index, 1);
}

export function moveArrayItem(target: Record<string, unknown>, path: PathPart[], index: number, direction: -1 | 1) {
  const array = arrayAt(target, path);
  const nextIndex = index + direction;
  if (nextIndex < 0 || nextIndex >= array.length) return;
  const [item] = array.splice(index, 1);
  array.splice(nextIndex, 0, item);
}

export function pruneConfigForSave(config: Record<string, unknown>): Record<string, unknown> {
  const knownPaths = knownConfigPaths();
  const keepPaths = new Set([
    "project_name",
    "input_audio",
    "silent_mode",
    "video.fps",
    "video.width",
    "video.height",
    "audio.demucs_model",
    "audio.whisper_model",
    "audio.language",
    "scene_generation.min_duration",
    "scene_generation.max_duration",
    "scene_generation.bias",
    "scene_generation.duration_preset",
    "scene_generation.seed",
    "vocal_detection.merge_gap",
    "vocal_detection.min_vocal_duration",
    "vocal_detection.min_silence_duration",
    "vocal_detection.rms_low_percentile",
    "vocal_detection.rms_high_percentile",
    "vocal_detection.rms_ratio",
    "vocal_detection.smooth_frames",
    "video_pipeline"
  ]);
  return pruneValue(config, [], keepPaths, knownPaths) as Record<string, unknown>;
}

function mergeDefaults(defaultValue: unknown, value: unknown): unknown {
  if (Array.isArray(defaultValue)) return Array.isArray(value) ? value : defaultValue;
  if (isRecord(defaultValue) && isRecord(value)) {
    return Object.fromEntries(
      [
        ...Object.keys(defaultValue),
        ...Object.keys(value).filter((key) => !(key in defaultValue))
      ].map((key) => [key, mergeDefaults(defaultValue[key], value[key])])
    );
  }
  return value ?? defaultValue;
}

function arrayAt(target: Record<string, unknown>, path: PathPart[]): unknown[] {
  const value = getPath(target, path);
  if (Array.isArray(value)) return value;
  setPath(target, path, []);
  return getPath(target, path) as unknown[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function clone<T>(value: T): T {
  return structuredClone(value);
}

function pruneValue(value: unknown, path: PathPart[], keepPaths: Set<string>, knownPaths: Set<string>): unknown {
  const key = normalizedPath(path);
  if (path.length && !knownPaths.has(key)) return value;
  if (keepPaths.has(key)) return value;
  if (typeof value === "string") return value.trim() ? value : undefined;
  if (Array.isArray(value)) {
    const items = value.map((item, index) => pruneValue(item, [...path, index], keepPaths, knownPaths)).filter((item) => item !== undefined);
    return items.length ? items : undefined;
  }
  if (isRecord(value)) {
    if (isEmptyLora(value)) return undefined;
    const entries = Object.entries(value)
      .map(([childKey, child]) => [childKey, pruneValue(child, [...path, childKey], keepPaths, knownPaths)] as const)
      .filter(([, child]) => child !== undefined);
    return entries.length ? Object.fromEntries(entries) : undefined;
  }
  if (value === false || value === null || value === undefined) return undefined;
  return value;
}

function isEmptyLora(value: Record<string, unknown>): boolean {
  return value.enabled === false && !String(value.name ?? "").trim() && Number(value.strength_model ?? 1) === 1 && Number(value.strength_clip ?? 1) === 1;
}

function knownConfigPaths(): Set<string> {
  const paths = new Set<string>();
  collectKnownPaths(DEFAULT_CONFIG, [], paths);
  for (const [key, template] of Object.entries(ARRAY_TEMPLATES)) {
    collectKnownPaths(template, [key, 0], paths);
  }
  return paths;
}

function collectKnownPaths(value: unknown, path: PathPart[], paths: Set<string>) {
  if (path.length) paths.add(normalizedPath(path));
  if (isRecord(value)) {
    for (const [key, child] of Object.entries(value)) collectKnownPaths(child, [...path, key], paths);
  }
}

function normalizedPath(path: PathPart[]): string {
  return path.map((part) => (typeof part === "number" ? "*" : part)).join(".");
}
