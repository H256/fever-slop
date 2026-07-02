<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Play, Save } from "lucide-vue-next";
import { useRoute, useRouter } from "vue-router";
import { api, mediaUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import JsonEditor from "../components/JsonEditor.vue";

const route = useRoute();
const router = useRouter();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const selectedPath = ref("");
const data = ref<unknown>(null);
const confirmPipelineStart = ref(false);
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

const artifactPaths = computed(() =>
  [
    ...(studio.currentProject?.artifacts.configs ?? []),
    ...(studio.currentProject?.artifacts.render_plans ?? []),
    ...(studio.currentProject?.artifacts.generated_json ?? []),
    ...(studio.currentProject?.artifacts.references ?? []),
    ...(studio.currentProject?.artifacts.images ?? []),
    ...(studio.currentProject?.artifacts.videos ?? [])
  ].filter((path, index, paths) => paths.indexOf(path) === index)
);
const isConfig = computed(() => selectedPath.value === "config.json");
const isJson = computed(() => selectedPath.value.endsWith(".json"));
const isImage = computed(() => /\.(png|jpe?g|webp|gif)$/i.test(selectedPath.value));
const isVideo = computed(() => /\.(mp4|mov|webm)$/i.test(selectedPath.value));
const configFields = computed(() => (isConfig.value ? collectFields(data.value) : []));
const configGroups = computed(() => groupFields(configFields.value));

onMounted(async () => {
  await studio.loadProject(projectId.value);
  selectedPath.value = String(route.query.path ?? artifactPaths.value[0] ?? "");
});

watch(selectedPath, async (path) => {
  if (!path) return;
  data.value = isJson.value ? (await api.artifact(projectId.value, path)).data : null;
});

watch(
  () => route.query.path,
  (path) => {
    if (typeof path === "string") selectedPath.value = path;
  }
);

async function save() {
  if (selectedPath.value && isJson.value) await api.saveArtifact(projectId.value, selectedPath.value, data.value);
}

async function startStandardPipeline() {
  confirmPipelineStart.value = false;
  if (isConfig.value) await save();
  await studio.startJob(projectId.value, "full-pipeline");
  await router.push(`/projects/${projectId.value}/pipeline`);
}

function collectFields(value: unknown, path: PathPart[] = []): FormField[] {
  if (typeof value === "boolean") return [field(path, "boolean", value)];
  if (typeof value === "number") return [field(path, "number", value)];
  if (typeof value === "string") return [field(path, value.length > 90 || value.includes("\n") ? "longText" : "shortText", value)];
  if (Array.isArray(value)) {
    if (value.every((item) => ["string", "number", "boolean"].includes(typeof item))) return [field(path, "simpleArray", value)];
    return value.flatMap((item, index) => collectFields(item, [...path, index]));
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => collectFields(child, [...path, key]));
  }
  return [];
}

function field(path: PathPart[], kind: FieldKind, value: unknown): FormField {
  return { path, kind, value, label: labelFor(path), help: helpForConfigField(path) };
}

function updateConfigField(field: FormField, event: Event) {
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  let value: unknown = target.value;
  if (field.kind === "boolean") value = (target as HTMLInputElement).checked;
  if (field.kind === "number") value = Number(target.value);
  if (field.kind === "simpleArray") value = target.value.split(",").map((item) => item.trim()).filter(Boolean);
  if (data.value && typeof data.value === "object") setPath(data.value as Record<string, unknown>, field.path, value);
}

function setPath(target: Record<string, unknown>, path: PathPart[], value: unknown) {
  let current: unknown = target;
  for (const part of path.slice(0, -1)) {
    if (!(part in (current as Record<string, unknown>))) (current as Record<string, unknown>)[part] = {};
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

function labelFor(path: PathPart[]): string {
  return path.map((part) => (typeof part === "number" ? `#${part + 1}` : part.replaceAll("_", " "))).join(" / ");
}

function fieldLabel(field: FormField, group: FormGroup): string {
  return labelFor(field.path.slice(group.path.length));
}

function optionsForField(field: FormField): string[] {
  const key = field.path.join(".");
  const name = String(field.path.at(-1));
  const options: Record<string, string[]> = {
    "audio.language": ["en", "de", "fr", "es", "it"],
    "audio.demucs_model": ["htdemucs_ft", "htdemucs", "mdx_extra_q"],
    "audio.whisper_model": ["tiny", "base", "small", "medium", "large"],
    "scene_generation.duration_preset": ["impact_weighted", "uniform", "vocal_weighted"],
    video_pipeline: ["ltx_i2v", "ltx_msr"],
    render_mode: ["single_prompt", "relay", "auto"]
  };
  return options[key] ?? options[name] ?? [];
}

function helpForConfigField(path: PathPart[]): string {
  const key = path.join(".");
  const name = String(path.at(-1));
  const descriptions: Record<string, string> = {
    project_name: "Required. Does not affect generation prompts directly. Display name and default output filename stem.",
    input_audio: "Required. Affects the whole pipeline. Project-relative path to the source audio file.",
    lyrics: "Optional. Affects timeline alignment and prompt context when available.",
    story_idea: "Optional. Affects prompt generation. High-level narrative direction for generated scenes.",
    style: "Optional. Affects prompt generation. Global visual style applied to generated prompts.",
    subject: "Optional. Affects prompt generation and anchor fixing. Main subject anchor for consistency.",
    locations: "Optional. Affects prompt generation. Comma-separated location ideas available to planning.",
    "video.fps": "Required. Affects video generation timing and frame counts.",
    "video.width": "Required. Affects image/video generation resolution.",
    "video.height": "Required. Affects image/video generation resolution.",
    "audio.demucs_model": "Optional. Affects audio analysis. Demucs model name for stem separation.",
    "audio.whisper_model": "Optional. Affects lyric/timeline analysis. Whisper model name.",
    "audio.language": "Optional. Affects transcription and prompt context. Language hint.",
    "scene_generation.min_duration": "Optional. Affects scene segmentation and later clip lengths.",
    "scene_generation.max_duration": "Optional. Affects scene segmentation and later clip lengths.",
    "scene_generation.bias": "Optional. Affects scene segmentation balance.",
    "scene_generation.duration_preset": "Optional. Affects scene duration strategy.",
    "scene_generation.seed": "Optional. Affects deterministic planning where supported.",
    "lora_1.enabled": "Optional. Affects image/video generation when the workflow uses this LoRA slot.",
    "lora_1.name": "Optional. Affects image/video generation when enabled. ComfyUI LoRA name/path.",
    "lora_1.strength_model": "Optional. Affects generation style/identity strength for the first LoRA slot.",
    "lora_1.strength_clip": "Optional. Affects CLIP/text conditioning strength for the first LoRA slot.",
    lora_split_enabled: "Optional. Affects LoRA application where split handling is supported."
  };
  const byName: Record<string, string> = {
    global: "Optional. Affects prompt generation across stages.",
    concepts: "Optional. Affects concept generation.",
    zimage: "Optional. Affects image prompt generation.",
    ltx: "Optional. Affects video prompt generation.",
    prompt: "Optional. Affects whichever prompt stage consumes this guidance.",
    character_visibility: "Optional. Affects prompt generation. Rules for character visibility.",
    shot_types: "Optional. Affects prompt generation. Preferred shot composition guidance.",
    environments: "Optional. Affects prompt generation. Environment guidance.",
    lighting: "Optional. Affects prompt generation. Lighting guidance.",
    camera_motion: "Optional. Affects prompt generation. Camera movement guidance.",
    physical_interaction: "Optional. Affects prompt generation. Physical action or interaction guidance.",
    outfit_rules: "Optional. Affects prompt generation. Costume/wardrobe consistency guidance."
  };
  return descriptions[key] ?? byName[name] ?? "Optional. May affect later generation if a pipeline step consumes this setting.";
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Artifacts</h1>
        <p>Edit JSON artifacts and preview generated media.</p>
      </div>
      <div class="button-row">
        <button v-if="isConfig" class="button secondary" @click="confirmPipelineStart = true"><Play :size="18" /> Save and run pipeline</button>
        <button class="button" :disabled="!selectedPath || !isJson" @click="save"><Save :size="18" /> Save</button>
      </div>
    </header>
    <div class="editor-layout">
      <section class="panel path-list">
        <button v-for="path in artifactPaths" :key="path" :class="{ active: path === selectedPath }" @click="selectedPath = path">{{ path }}</button>
      </section>
      <section class="document-panel">
        <h2>{{ selectedPath }}</h2>
        <div v-if="isConfig" class="generated-form">
          <fieldset v-for="group in configGroups" :key="group.key" class="form-block">
            <legend>{{ group.title }}</legend>
            <label v-for="field in group.fields" :key="field.path.join('.')">
              <span class="field-title">{{ fieldLabel(field, group) }}</span>
              <span class="field-help">{{ field.help }}</span>
              <select v-if="optionsForField(field).length" :value="displayValue(field)" @change="updateConfigField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateConfigField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateConfigField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayValue(field)" @input="updateConfigField(field, $event)" />
              <input v-else type="text" :value="displayValue(field)" @input="updateConfigField(field, $event)" />
            </label>
          </fieldset>
          <details class="advanced-json">
            <summary>Advanced JSON</summary>
            <JsonEditor v-model="data" />
          </details>
        </div>
        <JsonEditor v-else-if="selectedPath && isJson" v-model="data" />
        <figure v-else-if="selectedPath && isImage" class="artifact-preview">
          <img :src="mediaUrl(projectId, selectedPath)" :alt="selectedPath" />
          <figcaption>{{ selectedPath }}</figcaption>
        </figure>
        <figure v-else-if="selectedPath && isVideo" class="artifact-preview">
          <video :src="mediaUrl(projectId, selectedPath)" controls />
          <figcaption>{{ selectedPath }}</figcaption>
        </figure>
        <section v-else-if="selectedPath" class="panel">
          <h3>Preview unavailable</h3>
          <p>This artifact is not JSON, image, or video media.</p>
        </section>
      </section>
    </div>
    <ConfirmDialog
      :open="confirmPipelineStart"
      title="Run full pipeline?"
      message="This saves config.json and starts the normal generation pipeline. Generated artifacts may be overwritten."
      confirm-label="Save and run"
      @cancel="confirmPipelineStart = false"
      @confirm="startStandardPipeline"
    />
  </section>
</template>
