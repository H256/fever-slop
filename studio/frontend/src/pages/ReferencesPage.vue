<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { Plus, RefreshCw, Save, Trash2, WandSparkles, X } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import { api, mediaUrl } from "../api";
import type { RenderScene } from "../types";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const renderPlanPath = ref("");
const scenes = ref<RenderScene[]>([]);
const usedActorIds = ref<Set<string>>(new Set());
const usedLocationIds = ref<Set<string>>(new Set());
const selectedManifestPath = ref("");
const manifestData = ref<Record<string, unknown> | null>(null);
const newKind = ref<"actors" | "locations">("actors");
const newId = ref("");
const newName = ref("");
const newDescription = ref("");
const newImage = ref<File | null>(null);
const showAddForm = ref(false);
const lightboxImage = ref("");
const rerenderingReference = ref(false);
const startingProcess = ref(false);
const allReferenceImages = computed(() => studio.currentProject?.artifacts.images.filter((path) => path.includes("/references/")) ?? []);
const allManifests = computed(() => studio.currentProject?.artifacts.references.filter((path) => path.endsWith("/manifest.json")) ?? []);
const manifests = computed(() => allManifests.value.filter(isUsedReferencePath));
const otherReferences = computed(() => [...allManifests.value, ...allReferenceImages.value].filter((path) => !isUsedReferencePath(path)));
const actorIds = computed(() => new Set(allManifests.value.map((path) => referenceId(path, "actors")).filter(Boolean) as string[]));
const locationIds = computed(() => new Set(allManifests.value.map((path) => referenceId(path, "locations")).filter(Boolean) as string[]));
const missingActors = computed(() => [...usedActorIds.value].filter((id) => !actorIds.value.has(id)));
const missingLocations = computed(() => [...usedLocationIds.value].filter((id) => id && !locationIds.value.has(id)));
const manifestFields = computed(() => collectFields(manifestData.value));
const selectedReferenceId = computed(() => referenceId(selectedManifestPath.value, "actors") ?? referenceId(selectedManifestPath.value, "locations") ?? "");
const selectedReferenceKind = computed(() => (selectedManifestPath.value.includes("/actors/") ? "actor" : "location"));
const selectedReferenceImages = computed(() => referenceImagesForManifest(selectedManifestPath.value));
const hasActiveProcess = computed(() => studio.jobs.some((job) => ["queued", "running"].includes(job.status)) || rerenderingReference.value || startingProcess.value);

type PathPart = string | number;
type FieldKind = "boolean" | "number" | "shortText" | "longText" | "simpleArray";
interface FormField {
  path: PathPart[];
  label: string;
  kind: FieldKind;
  value: unknown;
  help: string;
}

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await studio.loadJobs(projectId.value);
  await loadUsedReferences();
  if (manifests.value[0]) await selectManifest(manifests.value[0]);
});

async function loadUsedReferences() {
  renderPlanPath.value = studio.currentProject?.artifacts.render_plans[0] ?? "";
  if (!renderPlanPath.value) return;
  scenes.value = (await api.artifact(projectId.value, renderPlanPath.value)).data as RenderScene[];
  const actors = new Set<string>();
  const locations = new Set<string>();
  for (const scene of scenes.value) {
    const references = scene.references as Record<string, unknown> | undefined;
    for (const id of (references?.actor_ids as string[] | undefined) ?? []) actors.add(id);
    if (typeof references?.location_id === "string") locations.add(references.location_id);
  }
  usedActorIds.value = actors;
  usedLocationIds.value = locations;
}

function isUsedReferencePath(path: string): boolean {
  const actor = path.match(/\/references\/actors\/([^/]+)\//)?.[1];
  const location = path.match(/\/references\/locations\/([^/]+)\//)?.[1];
  return Boolean((actor && usedActorIds.value.has(actor)) || (location && usedLocationIds.value.has(location)));
}

async function selectManifest(path: string) {
  selectedManifestPath.value = path;
  manifestData.value = (await api.artifact(projectId.value, path)).data as Record<string, unknown>;
}

async function saveManifest() {
  if (!selectedManifestPath.value || !manifestData.value) return;
  await api.saveArtifact(projectId.value, selectedManifestPath.value, manifestData.value);
  await studio.loadProject(projectId.value);
}

async function rerenderSelectedReference() {
  if (!selectedReferenceId.value || hasActiveProcess.value) return;
  rerenderingReference.value = true;
  const job = await studio.startJob(projectId.value, "reference-rerender", undefined, {
    reference_kind: selectedReferenceKind.value,
    reference_id: selectedReferenceId.value
  });
  const timer = window.setInterval(async () => {
    const jobs = await api.jobs(projectId.value);
    const current = jobs.find((candidate) => candidate.id === job.id);
    if (!current || current.status === "queued" || current.status === "running") return;
    window.clearInterval(timer);
    rerenderingReference.value = false;
    await studio.loadJobs(projectId.value);
    await studio.loadProject(projectId.value);
    await selectManifest(selectedManifestPath.value);
  }, 1500);
}

async function startProcess(action: string) {
  if (hasActiveProcess.value) return;
  startingProcess.value = true;
  try {
    await studio.startJob(projectId.value, action);
    await studio.loadJobs(projectId.value);
  } finally {
    startingProcess.value = false;
  }
}

async function addReference() {
  const id = slugify(newId.value || newName.value);
  if (!id) return;
  const path = `output/references/${newKind.value}/${id}/manifest.json`;
  const data = {
    id,
    name: newName.value || id,
    ...(newKind.value === "locations" ? { kind: "location" } : { role: "" }),
    visual_description: newDescription.value,
    image_prompt: newDescription.value
  };
  await api.saveArtifact(projectId.value, path, data);
  if (newImage.value) {
    const extension = imageExtension(newImage.value);
    await api.uploadMedia(projectId.value, `output/references/${newKind.value}/${id}/sheet.${extension}`, await readFileAsDataUrl(newImage.value));
  }
  newId.value = "";
  newName.value = "";
  newDescription.value = "";
  newImage.value = null;
  showAddForm.value = false;
  await studio.loadProject(projectId.value);
  await selectManifest(path);
}

async function removeFromRenderPlan() {
  if (!renderPlanPath.value || !selectedReferenceId.value) return;
  for (const scene of scenes.value) {
    const references = (scene.references ?? {}) as Record<string, unknown>;
    if (selectedReferenceKind.value === "actor") {
      references.actor_ids = ((references.actor_ids as string[] | undefined) ?? []).filter((id) => id !== selectedReferenceId.value);
    } else if (references.location_id === selectedReferenceId.value) {
      references.location_id = "";
    }
    scene.references = references;
  }
  await api.saveArtifact(projectId.value, renderPlanPath.value, scenes.value);
  await loadUsedReferences();
}

function referenceId(path: string, kind: "actors" | "locations"): string | undefined {
  return path.match(new RegExp(`/references/${kind}/([^/]+)/`))?.[1];
}

function referenceImagesForManifest(path: string): string[] {
  if (!path) return [];
  const prefix = path.replace(/manifest\.json$/, "");
  return allReferenceImages.value.filter((imagePath) => imagePath.startsWith(prefix));
}

function referenceLabel(path: string): string {
  const id = referenceId(path, "actors") ?? referenceId(path, "locations") ?? path;
  return id.replaceAll("_", " ");
}

function referenceKind(path: string): string {
  return path.includes("/actors/") ? "Actor" : "Location";
}

function slugify(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
}

function setNewImage(event: Event) {
  newImage.value = ((event.target as HTMLInputElement).files ?? [])[0] ?? null;
}

function imageExtension(file: File): string {
  if (file.type === "image/jpeg") return "jpg";
  if (file.type === "image/webp") return "webp";
  return "png";
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
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
  return { path, kind, value, label: labelFor(path), help: helpForManifestField(path) };
}

function labelFor(path: PathPart[]): string {
  return path.map((part) => (typeof part === "number" ? `#${part + 1}` : part.replaceAll("_", " "))).join(" / ");
}

function helpForManifestField(path: PathPart[]): string {
  const key = path.join(".");
  const descriptions: Record<string, string> = {
    id: "Required. Stable reference id used by render-plan scenes.",
    name: "Required. Human-readable reference name shown in Studio.",
    role: "Optional. Actor role or narrative function. Affects prompt/reference regeneration.",
    kind: "Optional. Location category. Affects reference organization.",
    visual_description: "Required for rerendering. Describes the actor or location visually.",
    image_prompt: "Required for rerendering. Prompt used to generate the reference sheet."
  };
  return descriptions[key] ?? "Optional. May affect reference rendering if the reference pipeline consumes this value.";
}

function updateManifestField(field: FormField, event: Event) {
  if (!manifestData.value) return;
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  let value: unknown = target.value;
  if (field.kind === "boolean") value = (target as HTMLInputElement).checked;
  if (field.kind === "number") value = Number(target.value);
  if (field.kind === "simpleArray") value = target.value.split(",").map((item) => item.trim()).filter(Boolean);
  setPath(manifestData.value, field.path, value);
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

function optionsForField(field: FormField): string[] {
  const name = String(field.path.at(-1));
  const options: Record<string, string[]> = {
    kind: ["actor", "location"],
    role: ["", "hero", "villain", "supporting", "creature", "background"]
  };
  return options[name] ?? [];
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>References</h1>
        <p>Actor and location references used by the active render plan.</p>
      </div>
      <div class="button-row">
        <button class="button" :disabled="hasActiveProcess" @click="startProcess('msr-references')"><RefreshCw :size="18" /> Render refs</button>
        <button class="button secondary" :disabled="hasActiveProcess" @click="startProcess('rebuild-plan')"><WandSparkles :size="18" /> Rebuild plan</button>
      </div>
    </header>
    <section v-if="hasActiveProcess" class="panel notice-panel">
      <h2>A backend process is running</h2>
      <p>Reference controls are disabled until the current job finishes.</p>
    </section>
    <section v-if="missingActors.length || missingLocations.length" class="panel notice-panel">
      <h2>Missing References</h2>
      <p>These ids are used by the render plan but do not have a manifest.</p>
      <div class="badge-row">
        <span v-for="id in missingActors" :key="`actor-${id}`" class="status-badge missing">actor: {{ id }}</span>
        <span v-for="id in missingLocations" :key="`location-${id}`" class="status-badge missing">location: {{ id }}</span>
      </div>
    </section>
    <div class="reference-workspace">
      <section class="panel reference-list-panel">
        <header class="panel-header">
          <h2>Used References</h2>
          <button class="button secondary" @click="showAddForm = !showAddForm"><Plus :size="18" /> Add</button>
        </header>
        <div class="reference-manifest-list">
          <button
            v-for="path in manifests"
            :key="path"
            class="reference-manifest-card"
            :class="{ active: path === selectedManifestPath }"
            @click="selectManifest(path)"
          >
            <div class="reference-card-thumbs">
              <img v-for="imagePath in referenceImagesForManifest(path).slice(0, 3)" :key="imagePath" :src="mediaUrl(projectId, imagePath)" :alt="referenceLabel(path)" />
              <span v-if="referenceImagesForManifest(path).length === 0" class="reference-empty-thumb">No image</span>
            </div>
            <strong>{{ referenceLabel(path) }}</strong>
            <small>{{ referenceKind(path) }} · {{ path }}</small>
          </button>
        </div>
      </section>
      <section class="document-panel">
        <header>
          <h2>{{ selectedManifestPath || "Select a manifest" }}</h2>
          <div class="button-row">
            <button class="button secondary" :disabled="!selectedManifestPath || hasActiveProcess" @click="removeFromRenderPlan"><Trash2 :size="18" /> Remove from plan</button>
            <button class="button secondary" :disabled="!selectedManifestPath || hasActiveProcess" @click="rerenderSelectedReference"><RefreshCw :size="18" /> Rerender</button>
            <button class="button" :disabled="!selectedManifestPath || hasActiveProcess" @click="saveManifest"><Save :size="18" /> Save manifest</button>
          </div>
        </header>
        <p class="job-note">After changing prompt or visual description fields, rerender references and rebuild the plan to propagate changes.</p>
        <div v-if="selectedReferenceImages.length" class="reference-selected-images">
          <button v-for="imagePath in selectedReferenceImages" :key="imagePath" class="reference-image-button" @click="lightboxImage = imagePath">
            <img :src="mediaUrl(projectId, imagePath)" :alt="selectedReferenceId" />
          </button>
        </div>
        <div v-if="manifestData" class="generated-form">
          <fieldset class="form-block">
            <legend>Manifest</legend>
            <label v-for="field in manifestFields" :key="field.path.join('.')">
              <span class="field-title">{{ field.label }}</span>
              <span class="field-help">{{ field.help }}</span>
              <select v-if="optionsForField(field).length" :value="displayValue(field)" @change="updateManifestField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option || "none" }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateManifestField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateManifestField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayValue(field)" @input="updateManifestField(field, $event)" />
              <input v-else type="text" :value="displayValue(field)" @input="updateManifestField(field, $event)" />
            </label>
          </fieldset>
        </div>
      </section>
      <section v-if="showAddForm" class="panel add-reference-panel">
        <h2>Add Reference</h2>
        <label>
          Type
          <select v-model="newKind">
            <option value="actors">Actor</option>
            <option value="locations">Location</option>
          </select>
        </label>
        <label>
          Name
          <input v-model="newName" type="text" />
        </label>
        <label>
          ID
          <input v-model="newId" type="text" placeholder="auto from name" />
        </label>
        <label>
          Visual description
          <textarea v-model="newDescription" class="transcript-area" />
        </label>
        <label>
          Existing image
          <input type="file" accept="image/png,image/jpeg,image/webp" @change="setNewImage" />
          <span class="field-help">Optional. Saved as the reference sheet image and shown in reference pickers.</span>
        </label>
        <button class="button" @click="addReference"><Plus :size="18" /> Add manifest</button>
      </section>
    </div>
    <details v-if="otherReferences.length" class="panel collapsed-list">
      <summary>Other reference artifacts</summary>
      <div class="path-list">
        <RouterLink v-for="path in otherReferences" :key="path" :to="`/projects/${projectId}/artifacts?path=${encodeURIComponent(path)}`">{{ path }}</RouterLink>
      </div>
    </details>
    <div v-if="lightboxImage" class="modal-backdrop" role="presentation" @click.self="lightboxImage = ''">
      <section class="lightbox" role="dialog" aria-modal="true" aria-label="Reference image preview">
        <button class="icon-button lightbox-close" @click="lightboxImage = ''"><X :size="20" /></button>
        <img :src="mediaUrl(projectId, lightboxImage)" alt="Reference preview" />
      </section>
    </div>
  </section>
</template>
