<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { onBeforeRouteLeave, useRoute } from "vue-router";
import { Save, Undo2 } from "lucide-vue-next";
import { api } from "../api";
import {
  addArrayItem,
  ARRAY_TEMPLATES,
  getPath,
  mergeConfigDefaults,
  moveArrayItem,
  pruneConfigForSave,
  removeArrayItem,
  setPath,
  type PathPart
} from "../lib/configForm";
import JsonEditor from "../components/JsonEditor.vue";

type FieldKind = "boolean" | "number" | "shortText" | "longText";
interface FormField {
  path: PathPart[];
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
interface ArrayGroup {
  key: string;
  title: string;
  path: PathPart[];
  items: unknown[];
  template: unknown;
}

const route = useRoute();
const projectId = computed(() => String(route.params.projectId));
const config = ref<Record<string, unknown> | null>(null);
const savedSnapshot = ref("");
const loading = ref(true);
const saving = ref(false);
const error = ref("");
const success = ref("");
const validationErrors = computed(() => validateConfig(config.value));
const dirty = computed(() => Boolean(config.value && JSON.stringify(config.value) !== savedSnapshot.value));
const fields = computed(() => (config.value ? collectFields(config.value) : []));
const groups = computed(() => groupFields(fields.value));
const arrayGroups = computed(() => (config.value ? collectArrayGroups(config.value) : []));

onMounted(async () => {
  window.addEventListener("beforeunload", beforeUnload);
  await loadConfig();
});

onUnmounted(() => {
  window.removeEventListener("beforeunload", beforeUnload);
});

onBeforeRouteLeave(() => {
  if (!dirty.value) return true;
  return window.confirm("Discard unsaved project settings changes?");
});

async function loadConfig() {
  loading.value = true;
  error.value = "";
  success.value = "";
  try {
    const artifact = await api.artifact(projectId.value, "config.json");
    config.value = mergeConfigDefaults(artifact.data);
    savedSnapshot.value = JSON.stringify(config.value);
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught);
  } finally {
    loading.value = false;
  }
}

async function save() {
  error.value = "";
  success.value = "";
  if (!config.value) return;
  if (validationErrors.value.length) return;
  saving.value = true;
  try {
    const payload = pruneConfigForSave(config.value);
    await api.saveArtifact(projectId.value, "config.json", payload);
    config.value = mergeConfigDefaults(payload);
    savedSnapshot.value = JSON.stringify(config.value);
    success.value = "Settings saved";
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught);
  } finally {
    saving.value = false;
  }
}

function reset() {
  if (!savedSnapshot.value) return;
  config.value = JSON.parse(savedSnapshot.value) as Record<string, unknown>;
  error.value = "";
  success.value = "";
}

function beforeUnload(event: BeforeUnloadEvent) {
  if (!dirty.value) return;
  event.preventDefault();
  event.returnValue = "";
}

function validateConfig(value: Record<string, unknown> | null): string[] {
  if (!value) return [];
  const errors: string[] = [];
  if (!String(value.project_name ?? "").trim()) errors.push("Project name is required.");
  if (!String(value.input_audio ?? "").trim()) errors.push("Input audio is required.");
  if (!["single", "multi"].includes(String(value.subject_mode ?? "multi"))) errors.push("Subject mode must be single or multi.");
  const maxSceneActors = Number(value.max_scene_actors ?? 4);
  if (!Number.isFinite(maxSceneActors) || maxSceneActors < 1 || maxSceneActors > 4) errors.push("Max scene actors must be between 1 and 4.");
  return errors;
}

function collectFields(value: unknown, path: PathPart[] = []): FormField[] {
  if (typeof value === "boolean") return [field(path, "boolean", value)];
  if (typeof value === "number") return [field(path, "number", value)];
  if (typeof value === "string") return [field(path, value.length > 90 || value.includes("\n") ? "longText" : "shortText", value)];
  if (Array.isArray(value)) return value.flatMap((item, index) => collectFields(item, [...path, index]));
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .filter(([key]) => path.length > 0 || !(key in ARRAY_TEMPLATES))
      .flatMap(([key, child]) => collectFields(child, [...path, key]));
  }
  return [];
}

function collectArrayGroups(value: Record<string, unknown>): ArrayGroup[] {
  return Object.entries(ARRAY_TEMPLATES).map(([key, template]) => ({
    key,
    title: labelFor([key]),
    path: [key],
    items: Array.isArray(getPath(value, [key])) ? (getPath(value, [key]) as unknown[]) : [],
    template
  }));
}

function field(path: PathPart[], kind: FieldKind, value: unknown): FormField {
  return { path, kind, value, help: helpForConfigField(path) };
}

function updateField(field: FormField, event: Event) {
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  let value: unknown = target.value;
  if (field.kind === "boolean") value = (target as HTMLInputElement).checked;
  if (field.kind === "number") value = Number(target.value);
  if (config.value) setPath(config.value, field.path, value);
}

function addConfigArrayItem(group: ArrayGroup) {
  if (config.value) addArrayItem(config.value, group.path, group.template);
}

function removeConfigArrayItem(group: ArrayGroup, index: number) {
  if (config.value) removeArrayItem(config.value, group.path, index);
}

function moveConfigArrayItem(group: ArrayGroup, index: number, direction: -1 | 1) {
  if (config.value) moveArrayItem(config.value, group.path, index, direction);
}

function groupFields(formFields: FormField[]): FormGroup[] {
  const map = new Map<string, FormGroup>();
  for (const field of formFields) {
    const groupPath = field.path.length > 1 ? field.path.slice(0, -1) : [];
    const key = groupPath.join(".") || "general";
    if (!map.has(key)) map.set(key, { key, path: groupPath, title: groupPath.length ? labelFor(groupPath) : "General", fields: [] });
    map.get(key)?.fields.push(field);
  }
  return [...map.values()];
}

function labelFor(path: PathPart[]): string {
  return path.map((part) => (typeof part === "number" ? `#${part + 1}` : part.replaceAll("_", " "))).join(" / ");
}

function fieldLabel(field: FormField, group: FormGroup): string {
  return labelFor(field.path.slice(group.path.length));
}

function displayValue(field: FormField): string {
  return String(field.value ?? "");
}

function optionsForField(field: FormField): string[] {
  const key = field.path.join(".");
  const name = String(field.path.at(-1));
  const options: Record<string, string[]> = {
    "audio.language": ["en", "de", "fr", "es", "it"],
    "audio.demucs_model": ["htdemucs_ft", "htdemucs", "mdx_extra_q"],
    "audio.whisper_model": ["tiny", "base", "small", "medium", "large"],
    "scene_generation.duration_preset": ["impact_weighted", "uniform", "vocal_weighted"],
    subject_mode: ["multi", "single"],
    video_pipeline: ["ltx_i2v", "ltx_msr"],
    render_mode: ["single_prompt", "relay", "auto"]
  };
  return options[key] ?? options[name] ?? [];
}

function helpForConfigField(path: PathPart[]): string {
  const key = path.join(".");
  const descriptions: Record<string, string> = {
    project_name: "Required. Main project display name used by generation outputs.",
    input_audio: "Required. Project-relative path to the source audio file used by the pipeline.",
    lyrics: "Optional. Lyrics text used for timeline alignment and prompt context.",
    story_idea: "Optional. Narrative direction for generated scenes.",
    style: "Optional. Global visual style for prompt generation.",
    subject: "Optional. Main subject anchor for consistency.",
    subject_mode: "Required. Controls whether scenes use one subject or multiple actors.",
    max_scene_actors: "Required. Maximum actors per generated scene, from 1 to 4.",
    "video.fps": "Required. Video frame rate used for timing and rendering.",
    "video.width": "Required. Generated video width.",
    "video.height": "Required. Generated video height.",
    "audio.demucs_model": "Required. Audio stem separation model.",
    "audio.whisper_model": "Required. Transcription model.",
    "audio.language": "Required. Language hint for transcription.",
    lora_split_enabled: "Optional. Enables split LoRA handling where supported."
  };
  return descriptions[key] ?? "Optional. Used by generation when the relevant pipeline step consumes it.";
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Project Settings</h1>
        <p>Edit <code>config.json</code>, the main configuration used by generation and pipeline jobs.</p>
      </div>
      <div class="button-row">
        <span v-if="dirty" class="status-badge warning">Unsaved changes</span>
        <button class="button secondary" :disabled="!dirty || loading || saving" @click="reset"><Undo2 :size="18" /> Discard changes</button>
        <button class="button" :disabled="!dirty || loading || saving || Boolean(validationErrors.length)" @click="save"><Save :size="18" /> Save</button>
      </div>
    </header>

    <section v-if="loading" class="panel empty">Loading project settings...</section>
    <section v-else-if="error" class="panel error-text">{{ error }}</section>
    <section v-else-if="!config" class="panel empty">No config.json found for this project.</section>
    <div v-else class="settings-layout">
      <aside class="panel settings-summary">
        <h2>Main configuration</h2>
        <p>Changes saved here are written back to <code>config.json</code> and used by future generation runs.</p>
        <p v-if="success" class="success-text">{{ success }}</p>
        <ul v-if="validationErrors.length" class="validation-list">
          <li v-for="message in validationErrors" :key="message">{{ message }}</li>
        </ul>
      </aside>

      <section class="document-panel settings-form">
        <fieldset v-for="group in arrayGroups" :key="group.key" class="form-block array-form-block">
          <legend>{{ group.title }}</legend>
          <article v-for="(_item, index) in group.items" :key="index" class="array-item-block">
            <header>
              <strong>{{ group.title }} {{ index + 1 }}</strong>
              <div class="array-controls">
                <button type="button" class="button secondary compact-button" :disabled="index === 0" @click="moveConfigArrayItem(group, index, -1)">Up</button>
                <button type="button" class="button secondary compact-button" :disabled="index === group.items.length - 1" @click="moveConfigArrayItem(group, index, 1)">Down</button>
                <button type="button" class="button danger compact-button" @click="removeConfigArrayItem(group, index)">Remove</button>
              </div>
            </header>
            <label v-for="field in collectFields(_item, [...group.path, index])" :key="field.path.join('.')">
              <span class="field-title">{{ labelFor(field.path.slice(group.path.length + 1)) }}</span>
              <span class="field-help">{{ field.help }}</span>
              <select v-if="optionsForField(field).length" :value="displayValue(field)" @change="updateField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayValue(field)" @input="updateField(field, $event)" />
              <input v-else type="text" :value="displayValue(field)" @input="updateField(field, $event)" />
            </label>
          </article>
          <button type="button" class="button secondary" @click="addConfigArrayItem(group)">Add {{ group.title }}</button>
        </fieldset>

        <details v-for="group in groups" :key="group.key" class="settings-section" open>
          <summary>{{ group.title }}</summary>
          <fieldset class="form-block">
            <label v-for="field in group.fields" :key="field.path.join('.')">
              <span class="field-title">{{ fieldLabel(field, group) }}</span>
              <span class="field-help">{{ field.help }}</span>
              <select v-if="optionsForField(field).length" :value="displayValue(field)" @change="updateField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayValue(field)" @input="updateField(field, $event)" />
              <input v-else type="text" :value="displayValue(field)" @input="updateField(field, $event)" />
            </label>
          </fieldset>
        </details>

        <details class="advanced-json">
          <summary>Advanced JSON</summary>
          <JsonEditor v-model="config" />
        </details>
      </section>
    </div>
  </section>
</template>
