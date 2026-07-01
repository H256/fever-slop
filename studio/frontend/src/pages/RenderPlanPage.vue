<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Image, Play, Save, Video } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api, mediaUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import type { RenderScene } from "../types";
import JsonEditor from "../components/JsonEditor.vue";
import ConfirmDialog from "../components/ConfirmDialog.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const planPath = ref("");
const scenes = ref<RenderScene[]>([]);
const selectedSceneNumber = ref<number | null>(null);
const selectedRenderScenes = ref<Set<number>>(new Set());
const pendingRerender = ref<"selected" | "all" | null>(null);
const draft = ref<Record<string, unknown>>({});
const selected = computed(() => scenes.value.find((scene) => scene.scene === selectedSceneNumber.value));
const selectedRenderSceneNumbers = computed(() => [...selectedRenderScenes.value].sort((a, b) => a - b));

type PathPart = string | number;
type FieldKind = "boolean" | "number" | "shortText" | "longText" | "simpleArray";
interface FormField {
  path: PathPart[];
  label: string;
  kind: FieldKind;
  value: unknown;
  help: string;
}
interface FormGroup {
  key: string;
  title: string;
  path: PathPart[];
  fields: FormField[];
}
interface ReferenceOption {
  id: string;
  name: string;
  sheetPath: string;
}

const formFields = computed(() => collectFields(draft.value));
const visibleFormGroups = computed(() => groupFields(formFields.value.filter((field) => !isAdvancedRenderField(field))));
const advancedFormGroups = computed(() => groupFields(formFields.value.filter(isAdvancedRenderField)));
const actors = ref<ReferenceOption[]>([]);
const locations = ref<ReferenceOption[]>([]);
const selectedActorIds = computed(() => readPath(draft.value, ["references", "actor_ids"]) as string[] | undefined);
const selectedLocationId = computed(() => readPath(draft.value, ["references", "location_id"]) as string | undefined);

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await loadReferences();
  planPath.value = studio.currentProject?.artifacts.render_plans[0] ?? "";
  if (planPath.value) {
    const artifact = await api.artifact(projectId.value, planPath.value);
    scenes.value = artifact.data as RenderScene[];
    selectedSceneNumber.value = scenes.value[0]?.scene ?? null;
  }
});

watch(selected, (scene) => {
  draft.value = scene ? cloneJson(scene) : {};
});

async function saveScene() {
  if (!selectedSceneNumber.value) return;
  await api.patchScene(projectId.value, planPath.value, selectedSceneNumber.value, draft.value);
  const index = scenes.value.findIndex((scene) => scene.scene === selectedSceneNumber.value);
  if (index >= 0) scenes.value[index] = { ...(draft.value as RenderScene) };
}

function toggleScene(scene: RenderScene) {
  selectedSceneNumber.value = scene.scene;
}

function toggleRenderScene(sceneNumber: number, checked: boolean) {
  const next = new Set(selectedRenderScenes.value);
  if (checked) next.add(sceneNumber);
  else next.delete(sceneNumber);
  selectedRenderScenes.value = next;
}

function askRerender(mode: "selected" | "all") {
  pendingRerender.value = mode;
}

async function runRerender() {
  const mode = pendingRerender.value;
  pendingRerender.value = null;
  if (mode === "selected" && selectedRenderSceneNumbers.value.length) {
    await studio.startJob(projectId.value, "ltx-render-scenes", selectedRenderSceneNumbers.value);
  }
  if (mode === "all") {
    await studio.startJob(projectId.value, "ltx-render-scenes");
  }
}

async function loadReferences() {
  const references = studio.currentProject?.artifacts.references ?? [];
  actors.value = await loadReferenceGroup(references, "actors");
  locations.value = await loadReferenceGroup(references, "locations");
}

async function loadReferenceGroup(paths: string[], group: "actors" | "locations"): Promise<ReferenceOption[]> {
  const manifestPaths = paths.filter((path) => path.includes(`/references/${group}/`) && path.endsWith("/manifest.json"));
  const options = await Promise.all(
    manifestPaths.map(async (path) => {
      const data = (await api.artifact(projectId.value, path)).data as Record<string, unknown>;
      const id = String(data.id ?? path.split("/").at(-2) ?? "");
      return {
        id,
        name: String(data.name ?? id),
        sheetPath: path.replace("manifest.json", group === "actors" ? "msr_sheet.png" : "sheet.png")
      };
    })
  );
  return options.sort((a, b) => a.name.localeCompare(b.name));
}

function collectFields(value: unknown, path: PathPart[] = []): FormField[] {
  if (path.length === 1 && path[0] === "scene") return [];
  if (path.join(".") === "references.actor_ids" || path.join(".") === "references.location_id") return [];
  if (typeof value === "boolean") return [field(path, "boolean", value)];
  if (typeof value === "number") return [field(path, "number", value)];
  if (typeof value === "string") return [field(path, value.length > 90 || value.includes("\n") ? "longText" : "shortText", value)];
  if (Array.isArray(value)) {
    if (value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
      return [field(path, "simpleArray", value)];
    }
    return value.flatMap((item, index) => collectFields(item, [...path, index]));
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => collectFields(child, [...path, key]));
  }
  return [];
}

function field(path: PathPart[], kind: FieldKind, value: unknown): FormField {
  return { path, kind, value, label: labelFor(path), help: helpForRenderField(path) };
}

function labelFor(path: PathPart[]): string {
  return path
    .map((part) => (typeof part === "number" ? `#${part + 1}` : part.replaceAll("_", " ")))
    .join(" / ");
}

function updateField(field: FormField, event: Event) {
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  let value: unknown = target.value;
  if (field.kind === "boolean") value = (target as HTMLInputElement).checked;
  if (field.kind === "number") value = Number(target.value);
  if (field.kind === "simpleArray") {
    value = target.value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  setPath(draft.value, field.path, value);
}

function toggleActor(actorId: string, checked: boolean) {
  const current = new Set(selectedActorIds.value ?? []);
  if (checked) current.add(actorId);
  else current.delete(actorId);
  setPath(draft.value, ["references", "actor_ids"], [...current]);
}

function setLocation(locationId: string) {
  setPath(draft.value, ["references", "location_id"], locationId);
}

function setPath(target: Record<string, unknown>, path: PathPart[], value: unknown) {
  let current: unknown = target;
  for (const part of path.slice(0, -1)) {
    if (!(part in (current as Record<string, unknown>))) {
      (current as Record<string, unknown>)[part] = {};
    }
    current = (current as Record<string, unknown> | unknown[])[part as never];
  }
  (current as Record<string, unknown> | unknown[])[path[path.length - 1] as never] = value as never;
}

function displayValue(field: FormField): string {
  return Array.isArray(field.value) ? field.value.join(", ") : String(field.value ?? "");
}

function groupFields(fields: FormField[]): FormGroup[] {
  const groups = new Map<string, FormGroup>();
  for (const field of fields) {
    const groupPath = field.path.length > 1 ? field.path.slice(0, -1) : [];
    const key = groupPath.join(".") || "general";
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        path: groupPath,
        title: groupPath.length ? labelFor(groupPath) : "General",
        fields: []
      });
    }
    groups.get(key)?.fields.push(field);
  }
  return [...groups.values()];
}

function fieldLabel(field: FormField, group: FormGroup): string {
  return labelFor(field.path.slice(group.path.length));
}

function isAdvancedRenderField(field: FormField): boolean {
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

function generationTargets(field: FormField): ("image" | "video")[] {
  const key = field.path.join(".");
  const name = String(field.path.at(-1));
  if (key === "z_image.prompt" || key === "ltx.t2i_prompt" || name === "frame") return ["image"];
  if (key.startsWith("ltx.") || name === "frame_start" || name === "frame_end" || name === "frame_count") return ["video"];
  if (key === "metadata.base_concept" || key === "metadata.camera_motion" || key === "metadata.character_motion") return ["image", "video"];
  if (["fps", "width", "height", "duration_seconds"].includes(name)) return ["image", "video"];
  return [];
}

function optionsForField(field: FormField): string[] {
  const key = field.path.join(".");
  const name = String(field.path.at(-1));
  const options: Record<string, string[]> = {
    "ltx.render_mode_hint": ["single_prompt", "relay", "auto"],
    "metadata.type": ["vocals", "instrumental"],
    state: ["vocals", "instrumental"]
  };
  return options[key] ?? options[name] ?? [];
}

function scenePreview(scene: RenderScene): string {
  return String(readPath(scene, ["ltx", "base_prompt"]) ?? readPath(scene, ["z_image", "prompt"]) ?? readPath(scene, ["metadata", "lyrics"]) ?? "");
}

function readPath(value: unknown, path: string[]): unknown {
  let current = value;
  for (const part of path) {
    if (!current || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Render Plan</h1>
        <p>{{ planPath || "No render plan found." }}</p>
      </div>
      <div class="button-row">
        <button class="button secondary" :disabled="selectedRenderScenes.size === 0" @click="askRerender('selected')">
          <Play :size="18" /> Rerender selected
        </button>
        <button class="button" @click="askRerender('all')"><Play :size="18" /> Rerender all</button>
      </div>
    </header>
    <div v-if="scenes.length" class="editor-layout">
      <section class="panel scene-table">
        <div v-for="scene in scenes" :key="scene.scene" class="scene-row" :class="{ active: scene.scene === selectedSceneNumber }">
          <input
            type="checkbox"
            :checked="selectedRenderScenes.has(scene.scene)"
            :aria-label="`Select scene ${scene.scene} for rerender`"
            @change="toggleRenderScene(scene.scene, ($event.target as HTMLInputElement).checked)"
          />
          <button @click="toggleScene(scene)">
            <strong>Scene {{ scene.scene }}</strong>
            <small>{{ scenePreview(scene).slice(0, 120) }}</small>
          </button>
        </div>
      </section>
      <section class="document-panel">
        <header>
          <h2>Scene {{ selectedSceneNumber }}</h2>
          <button class="button" @click="saveScene"><Save :size="18" /> Save</button>
        </header>
        <section class="reference-picker" v-if="actors.length || locations.length">
          <h3>References</h3>
          <div v-if="actors.length" class="reference-group">
            <h4>Actors</h4>
            <label v-for="actor in actors" :key="actor.id" class="reference-card">
              <input
                type="checkbox"
                :checked="selectedActorIds?.includes(actor.id)"
                @change="toggleActor(actor.id, ($event.target as HTMLInputElement).checked)"
              />
              <img :src="mediaUrl(projectId, actor.sheetPath)" :alt="actor.name" />
              <span>{{ actor.name }}</span>
            </label>
          </div>
          <div v-if="locations.length" class="reference-group">
            <h4>Location</h4>
            <label v-for="location in locations" :key="location.id" class="reference-card">
              <input
                type="radio"
                name="scene-location"
                :checked="selectedLocationId === location.id"
                @change="setLocation(location.id)"
              />
              <img :src="mediaUrl(projectId, location.sheetPath)" :alt="location.name" />
              <span>{{ location.name }}</span>
            </label>
          </div>
        </section>
        <div class="generated-form">
          <fieldset v-for="group in visibleFormGroups" :key="group.key" class="form-block">
            <legend>{{ group.title }}</legend>
            <label v-for="field in group.fields" :key="field.path.join('.')">
              <span class="field-heading">
                <span class="field-title">{{ fieldLabel(field, group) }}</span>
                <span v-if="generationTargets(field).length" class="generation-icons" aria-label="Used in generation">
                  <Image v-if="generationTargets(field).includes('image')" :size="15" />
                  <Video v-if="generationTargets(field).includes('video')" :size="15" />
                </span>
              </span>
              <span class="field-help">{{ field.help }}</span>
              <select v-if="optionsForField(field).length" :value="displayValue(field)" @change="updateField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
              </select>
              <input
                v-else-if="field.kind === 'boolean'"
                type="checkbox"
                :checked="Boolean(field.value)"
                @change="updateField(field, $event)"
              />
              <input
                v-else-if="field.kind === 'number'"
                type="number"
                :value="field.value"
                @input="updateField(field, $event)"
              />
              <textarea
                v-else-if="field.kind === 'longText'"
                class="transcript-area"
                :value="displayValue(field)"
                @input="updateField(field, $event)"
              />
              <input
                v-else
                type="text"
                :value="displayValue(field)"
                @input="updateField(field, $event)"
              />
            </label>
          </fieldset>
        </div>
        <details v-if="advancedFormGroups.length" class="advanced-json advanced-settings">
          <summary>Advanced calculated settings</summary>
          <div class="generated-form">
            <fieldset v-for="group in advancedFormGroups" :key="group.key" class="form-block">
              <legend>{{ group.title }}</legend>
              <label v-for="field in group.fields" :key="field.path.join('.')">
                <span class="field-heading">
                  <span class="field-title">{{ fieldLabel(field, group) }}</span>
                  <span v-if="generationTargets(field).length" class="generation-icons" aria-label="Used in generation">
                    <Image v-if="generationTargets(field).includes('image')" :size="15" />
                    <Video v-if="generationTargets(field).includes('video')" :size="15" />
                  </span>
                </span>
                <span class="field-help">{{ field.help }}</span>
                <select v-if="optionsForField(field).length" :value="displayValue(field)" @change="updateField(field, $event)">
                  <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
                </select>
                <input
                  v-else-if="field.kind === 'boolean'"
                  type="checkbox"
                  :checked="Boolean(field.value)"
                  @change="updateField(field, $event)"
                />
                <input
                  v-else-if="field.kind === 'number'"
                  type="number"
                  :value="field.value"
                  @input="updateField(field, $event)"
                />
                <textarea
                  v-else-if="field.kind === 'longText'"
                  class="transcript-area"
                  :value="displayValue(field)"
                  @input="updateField(field, $event)"
                />
                <input
                  v-else
                  type="text"
                  :value="displayValue(field)"
                  @input="updateField(field, $event)"
                />
              </label>
            </fieldset>
          </div>
        </details>
        <details class="advanced-json">
          <summary>Advanced JSON</summary>
          <JsonEditor v-model="draft" />
        </details>
      </section>
    </div>
    <ConfirmDialog
      :open="Boolean(pendingRerender)"
      title="Start rerender?"
      :message="
        pendingRerender === 'all'
          ? `This will rerender all ${scenes.length} scenes for ${projectId}. Existing scene clips may be overwritten and jobs cannot be cancelled yet.`
          : `This will rerender ${selectedRenderSceneNumbers.length} selected scene(s): ${selectedRenderSceneNumbers.join(', ')}. Existing scene clips may be overwritten and jobs cannot be cancelled yet.`
      "
      confirm-label="Start rerender"
      @cancel="pendingRerender = null"
      @confirm="runRerender"
    />
  </section>
</template>
