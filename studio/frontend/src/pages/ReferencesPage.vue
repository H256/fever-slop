<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { Plus, RefreshCw, Save, Trash2, WandSparkles, X } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import { api, mediaUrl } from "../api";
import type { RenderScene } from "../types";
import type { PathPart } from "../lib/configForm";
import { collectObjectFields, displayObjectFieldValue, updateObjectField, type ObjectFormField } from "../lib/objectForm";

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
const processError = ref("");
const allReferenceImages = computed(() => studio.currentProject?.artifacts.images.filter((path) => path.includes("/references/")) ?? []);
const allManifests = computed(() => studio.currentProject?.artifacts.references.filter((path) => path.endsWith("/manifest.json")) ?? []);
const manifests = computed(() => allManifests.value.filter(isUsedReferencePath));
const otherReferences = computed(() => [...allManifests.value, ...allReferenceImages.value].filter((path) => !isUsedReferencePath(path)));
const hasMovieManifest = computed(() => allManifests.value.includes("movie/references/manifest.json"));
const isMovieProject = computed(() => studio.currentProject?.project_type === "movie");
const actorIds = computed(() => new Set([
  ...allManifests.value.map((path) => referenceId(path, "actors")).filter(Boolean) as string[],
  ...(isMovieProject.value && hasMovieManifest.value ? [...usedActorIds.value] : [])
]));
const locationIds = computed(() => new Set([
  ...allManifests.value.map((path) => referenceId(path, "locations")).filter(Boolean) as string[],
  ...(isMovieProject.value && hasMovieManifest.value ? [...usedLocationIds.value] : [])
]));
const missingActors = computed(() => [...usedActorIds.value].filter((id) => !actorIds.value.has(id)));
const missingLocations = computed(() => [...usedLocationIds.value].filter((id) => id && !locationIds.value.has(id)));
const manifestFields = computed(() => collectObjectFields(manifestData.value, { helpForField: helpForManifestField, primitiveArrayMode: "field" }));
const selectedReferenceId = computed(() => referenceId(selectedManifestPath.value, "actors") ?? referenceId(selectedManifestPath.value, "locations") ?? "");
const selectedReferenceKind = computed(() => (selectedManifestPath.value.includes("/actors/") ? "actor" : "location"));
const selectedReferenceImages = computed(() => referenceImagesForManifest(selectedManifestPath.value));
const hasActiveProcess = computed(() => studio.jobs.some((job) => ["queued", "running"].includes(job.status)) || rerenderingReference.value || startingProcess.value);

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await studio.loadJobs(projectId.value);
  await loadUsedReferences();
  if (manifests.value[0]) await selectManifest(manifests.value[0]);
});

async function loadUsedReferences() {
  renderPlanPath.value = studio.currentProject?.artifacts.render_plans[0] ?? "";
  if (!renderPlanPath.value) return;
  scenes.value = renderPlanScenes((await api.artifact(projectId.value, renderPlanPath.value)).data);
  const actors = new Set<string>();
  const locations = new Set<string>();
  for (const scene of scenes.value) {
    const references = scene.references as Record<string, unknown> | undefined;
    const referenceIds = scene.reference_ids as Record<string, unknown> | undefined;
    for (const id of (references?.actor_ids as string[] | undefined) ?? []) actors.add(id);
    for (const id of (referenceIds?.actors as string[] | undefined) ?? []) actors.add(id);
    if (typeof references?.location_id === "string") locations.add(references.location_id);
    if (typeof referenceIds?.location === "string") locations.add(referenceIds.location);
  }
  usedActorIds.value = actors;
  usedLocationIds.value = locations;
}

function renderPlanScenes(data: unknown): RenderScene[] {
  if (Array.isArray(data)) return data as RenderScene[];
  if (data && typeof data === "object") {
    const plan = data as Record<string, unknown>;
    if (Array.isArray(plan.shots)) return plan.shots as RenderScene[];
    if (Array.isArray(plan.scenes)) return plan.scenes as RenderScene[];
  }
  return [];
}

function isUsedReferencePath(path: string): boolean {
  if (isMovieProject.value && path === "movie/references/manifest.json") return true;
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
  processError.value = "";
  rerenderingReference.value = true;
  let job;
  try {
    job = await studio.startJob(projectId.value, "reference-rerender", undefined, {
      reference_kind: selectedReferenceKind.value,
      reference_id: selectedReferenceId.value
    });
  } catch (caught) {
    processError.value = apiErrorMessage(caught);
    rerenderingReference.value = false;
    return;
  }
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
  processError.value = "";
  try {
    await studio.startJob(projectId.value, isMovieProject.value && action === "msr-references" ? "movie-references" : action);
    await studio.loadJobs(projectId.value);
  } catch (caught) {
    processError.value = apiErrorMessage(caught);
  } finally {
    startingProcess.value = false;
  }
}

function apiErrorMessage(caught: unknown): string {
  if (!(caught instanceof Error)) return String(caught);
  const body = "body" in caught ? String((caught as { body?: string }).body ?? "") : "";
  if (!body) return caught.message;
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    return parsed.detail ? String(parsed.detail) : body;
  } catch {
    return body;
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

function updateManifestField(field: ObjectFormField, event: Event) {
  if (!manifestData.value) return;
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  updateObjectField(manifestData.value, field, field.kind === "boolean" ? (target as HTMLInputElement).checked : target.value);
}

function optionsForField(field: ObjectFormField): string[] {
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
    <section v-if="processError" class="panel pipeline-error">
      <strong>Could not start reference job</strong>
      <p>{{ processError }}</p>
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
              <select v-if="optionsForField(field).length" :value="displayObjectFieldValue(field)" @change="updateManifestField(field, $event)">
                <option v-for="option in optionsForField(field)" :key="option" :value="option">{{ option || "none" }}</option>
              </select>
              <input v-else-if="field.kind === 'boolean'" type="checkbox" :checked="Boolean(field.value)" @change="updateManifestField(field, $event)" />
              <input v-else-if="field.kind === 'number'" type="number" :value="field.value" @input="updateManifestField(field, $event)" />
              <textarea v-else-if="field.kind === 'longText'" class="transcript-area" :value="displayObjectFieldValue(field)" @input="updateManifestField(field, $event)" />
              <input v-else type="text" :value="displayObjectFieldValue(field)" @input="updateManifestField(field, $event)" />
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
