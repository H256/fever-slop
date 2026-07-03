<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { onBeforeRouteLeave, useRoute } from "vue-router";
import { Save, Undo2, Upload } from "lucide-vue-next";
import { api } from "../api";
import {
  addArrayItem,
  ARRAY_TEMPLATES,
  mergeConfigDefaults,
  moveArrayItem,
  pruneConfigForSave,
  removeArrayItem,
  type PathPart
} from "../lib/configForm";
import {
  collectArrayGroups,
  collectObjectFields,
  displayObjectFieldValue,
  fieldLabel,
  groupObjectFields,
  labelForPath,
  updateObjectField,
  type ObjectArrayGroup,
  type ObjectFormField
} from "../lib/objectForm";
import JsonEditor from "../components/JsonEditor.vue";
import type { ProjectSummary } from "../types";

const route = useRoute();
const projectId = computed(() => String(route.params.projectId));
const project = ref<ProjectSummary | null>(null);
const config = ref<Record<string, unknown> | null>(null);
const savedSnapshot = ref("");
const loading = ref(true);
const saving = ref(false);
const uploadingAudio = ref(false);
const error = ref("");
const success = ref("");
const uploadError = ref("");
const validationErrors = computed(() => validateConfig(config.value, project.value));
const dirty = computed(() => Boolean(config.value && JSON.stringify(config.value) !== savedSnapshot.value));
const fields = computed(() =>
  config.value
    ? collectObjectFields(config.value, {
        excludeRootKeys: [...Object.keys(ARRAY_TEMPLATES), "input_audio", "silent_mode"],
        helpForField: helpForConfigField,
        primitiveArrayMode: "expand"
      })
    : []
);
const groups = computed(() => groupObjectFields(fields.value));
const arrayGroups = computed(() => (config.value ? collectArrayGroups(config.value, ARRAY_TEMPLATES) : []));
const audioPath = computed(() => String(config.value?.input_audio ?? ""));

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
  uploadError.value = "";
  try {
    const [projectSummary, artifact] = await Promise.all([api.project(projectId.value), api.artifact(projectId.value, "config.json")]);
    project.value = projectSummary;
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
  uploadError.value = "";
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
  uploadError.value = "";
}

function beforeUnload(event: BeforeUnloadEvent) {
  if (!dirty.value) return;
  event.preventDefault();
  event.returnValue = "";
}

function validateConfig(value: Record<string, unknown> | null, projectSummary: ProjectSummary | null): string[] {
  if (!value) return [];
  const errors: string[] = [];
  const projectType = projectSummary?.project_type ?? projectSummary?.metadata?.project_type ?? "standard_music_video";
  if (!String(value.project_name ?? "").trim()) errors.push("Project name is required.");
  if (projectType !== "full_auto" && !String(value.input_audio ?? "").trim()) errors.push("Input audio is required.");
  if (typeof (value.silent_mode ?? false) !== "boolean") errors.push("Silent Mode must be true or false.");
  if (!["single", "multi"].includes(String(value.subject_mode ?? "multi"))) errors.push("Subject mode must be single or multi.");
  const maxSceneActors = Number(value.max_scene_actors ?? 4);
  if (!Number.isFinite(maxSceneActors) || maxSceneActors < 1 || maxSceneActors > 4) errors.push("Max scene actors must be between 1 and 4.");
  if (!["ltx_i2v", "ltx_msr"].includes(String(value.video_pipeline ?? "ltx_msr"))) errors.push("Video pipeline must be ltx_i2v or ltx_msr.");
  return errors;
}

function updateField(field: ObjectFormField, event: Event) {
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  if (config.value) updateObjectField(config.value, field, field.kind === "boolean" ? (target as HTMLInputElement).checked : target.value);
}

function updateInputAudio(event: Event) {
  const target = event.target as HTMLInputElement;
  if (config.value) config.value.input_audio = target.value;
}

function updateSilentMode(event: Event) {
  const target = event.target as HTMLInputElement;
  if (config.value) config.value.silent_mode = target.checked;
}

async function uploadAudio(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file || !config.value) return;
  uploadingAudio.value = true;
  error.value = "";
  uploadError.value = "";
  success.value = "";
  try {
    const result = await api.uploadAudio(projectId.value, file);
    config.value.input_audio = result.path;
    updateSavedAudioPath(result.path);
    success.value = "Audio uploaded";
  } catch (caught) {
    uploadError.value = apiErrorMessage(caught);
  } finally {
    uploadingAudio.value = false;
    target.value = "";
  }
}

function updateSavedAudioPath(path: string) {
  if (!savedSnapshot.value) return;
  const snapshot = JSON.parse(savedSnapshot.value) as Record<string, unknown>;
  snapshot.input_audio = path;
  savedSnapshot.value = JSON.stringify(snapshot);
}

function apiErrorMessage(caught: unknown): string {
  if (!(caught instanceof Error)) return String(caught);
  const body = "body" in caught ? String((caught as { body?: string }).body ?? "") : "";
  if (body) {
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (parsed.detail) return String(parsed.detail);
    } catch {
      return body;
    }
  }
  return caught.message;
}

function addConfigArrayItem(group: ObjectArrayGroup) {
  if (config.value) addArrayItem(config.value, group.path, group.template);
}

function removeConfigArrayItem(group: ObjectArrayGroup, index: number) {
  if (config.value) removeArrayItem(config.value, group.path, index);
}

function moveConfigArrayItem(group: ObjectArrayGroup, index: number, direction: -1 | 1) {
  if (config.value) moveArrayItem(config.value, group.path, index, direction);
}

function optionsForField(field: ObjectFormField): string[] {
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
    video_pipeline: "Required. MSR uses Scene/Actor Bible references; ltx_i2v is the classic path without those references.",
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
        <section class="form-block audio-source-section" aria-labelledby="audio-source-title">
          <div class="section-heading">
            <div>
              <h2 id="audio-source-title">Audio Source</h2>
              <p>Upload audio into this project's <code>input/</code> folder or enter a project-relative path.</p>
            </div>
            <label class="button secondary audio-upload-button">
              <Upload :size="18" />
              <span>{{ uploadingAudio ? "Uploading..." : "Upload" }}</span>
              <input
                aria-label="Upload audio file"
                type="file"
                accept=".mp3,.wav,.flac,.m4a,.ogg,audio/*"
                :disabled="uploadingAudio"
                @change="uploadAudio"
              />
            </label>
          </div>
          <label>
            <span class="field-title">Input audio</span>
            <span class="field-help">Required for standard projects. Stored as a path relative to the project root.</span>
            <input type="text" :value="audioPath" placeholder="input/song.mp3" @input="updateInputAudio" />
          </label>
          <p v-if="audioPath" class="relative-path-display"><strong>Project path:</strong> <code>{{ audioPath }}</code></p>
          <p v-if="uploadError" class="error-text">{{ uploadError }}</p>
        </section>

        <section class="form-block" aria-labelledby="generation-preferences-title">
          <div class="section-heading">
            <div>
              <h2 id="generation-preferences-title">Generation Preferences</h2>
              <p>Project-level prompt behavior used by future generation runs.</p>
            </div>
          </div>
          <label class="switch-row">
            <input
              type="checkbox"
              role="switch"
              aria-label="Silent Mode"
              :checked="Boolean(config.silent_mode)"
              :disabled="saving"
              @change="updateSilentMode"
            />
            <span>
              <span class="field-title">Silent Mode</span>
              <span class="field-help">Disables singing and lip-sync prompts while preserving emotional acting. Ideal for instrumental music videos.</span>
            </span>
          </label>
        </section>

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
            <label v-for="field in collectObjectFields(_item, { helpForField: helpForConfigField, primitiveArrayMode: 'expand' }, [...group.path, index])" :key="field.path.join('.')">
              <span class="field-title">{{ labelForPath(field.path.slice(group.path.length + 1)) }}</span>
              <span class="field-help">{{ field.help }}</span>
              <select v-if="optionsForField(field).length" :value="displayObjectFieldValue(field)" @change="updateField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayObjectFieldValue(field)" @input="updateField(field, $event)" />
              <input v-else type="text" :value="displayObjectFieldValue(field)" @input="updateField(field, $event)" />
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
              <select v-if="optionsForField(field).length" :value="displayObjectFieldValue(field)" @change="updateField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayObjectFieldValue(field)" @input="updateField(field, $event)" />
              <input v-else type="text" :value="displayObjectFieldValue(field)" @input="updateField(field, $event)" />
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
