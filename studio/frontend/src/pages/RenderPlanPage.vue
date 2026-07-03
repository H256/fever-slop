<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Play, Save } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api, mediaUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import type { RenderScene } from "../types";
import JsonEditor from "../components/JsonEditor.vue";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import RenderPlanFieldGroups from "../components/RenderPlanFieldGroups.vue";
import {
  readPath,
  scenePreview,
  setPath,
  updateRenderPlanFieldValue,
  useRenderPlanForm,
  type RenderPlanField
} from "../composables/renderPlanForm";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const planPath = ref("");
const scenes = ref<RenderScene[]>([]);
const selectedSceneNumber = ref<number | null>(null);
const selectedRenderScenes = ref<Set<number>>(new Set());
const pendingRerender = ref<"selected" | "all" | null>(null);
const startingRerender = ref(false);
const draft = ref<Record<string, unknown>>({});
const selected = computed(() => scenes.value.find((scene) => scene.scene === selectedSceneNumber.value));
const selectedRenderSceneNumbers = computed(() => [...selectedRenderScenes.value].sort((a, b) => a - b));
interface ReferenceOption {
  id: string;
  name: string;
  sheetPath: string;
}

const { visibleFormGroups, advancedFormGroups } = useRenderPlanForm(draft);
const actors = ref<ReferenceOption[]>([]);
const locations = ref<ReferenceOption[]>([]);
const selectedActorIds = computed(() => readPath(draft.value, ["references", "actor_ids"]) as string[] | undefined);
const selectedLocationId = computed(() => readPath(draft.value, ["references", "location_id"]) as string | undefined);
const hasActiveProcess = computed(() => studio.jobs.some((job) => ["queued", "running"].includes(job.status)));

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await studio.loadJobs(projectId.value);
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
  startingRerender.value = true;
  try {
    if (mode === "selected" && selectedRenderSceneNumbers.value.length) {
      await studio.startJob(projectId.value, "ltx-render-scenes", selectedRenderSceneNumbers.value);
    }
    if (mode === "all") {
      await studio.startJob(projectId.value, "ltx-render-scenes");
    }
    await studio.loadJobs(projectId.value);
  } finally {
    startingRerender.value = false;
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

function updateField(field: RenderPlanField, event: Event) {
  const target = event.target as HTMLInputElement | HTMLTextAreaElement;
  updateRenderPlanFieldValue(draft.value, field, field.kind === "boolean" ? (target as HTMLInputElement).checked : target.value);
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
        <button class="button secondary" :disabled="selectedRenderScenes.size === 0 || hasActiveProcess || startingRerender" @click="askRerender('selected')">
          <Play :size="18" /> Rerender selected
        </button>
        <button class="button" :disabled="hasActiveProcess || startingRerender" @click="askRerender('all')"><Play :size="18" /> Rerender all</button>
      </div>
    </header>
    <section v-if="hasActiveProcess || startingRerender" class="panel notice-panel">
      <h2>A backend process is running</h2>
      <p>Render controls are disabled until the current job finishes.</p>
    </section>
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
        <RenderPlanFieldGroups :groups="visibleFormGroups" @update-field="updateField" />
        <details v-if="advancedFormGroups.length" class="advanced-json advanced-settings">
          <summary>Advanced calculated settings</summary>
          <RenderPlanFieldGroups :groups="advancedFormGroups" @update-field="updateField" />
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
