<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Play, Save } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api } from "../api";
import { useStudioStore } from "../stores/studio";
import type { RenderScene } from "../types";
import JsonEditor from "../components/JsonEditor.vue";
import ProjectNav from "../components/ProjectNav.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const planPath = ref("");
const scenes = ref<RenderScene[]>([]);
const selectedSceneNumber = ref<number | null>(null);
const draft = ref<Record<string, unknown>>({});
const selected = computed(() => scenes.value.find((scene) => scene.scene === selectedSceneNumber.value));

onMounted(async () => {
  await studio.loadProject(projectId.value);
  planPath.value = studio.currentProject?.artifacts.render_plans[0] ?? "";
  if (planPath.value) {
    const artifact = await api.artifact(projectId.value, planPath.value);
    scenes.value = artifact.data as RenderScene[];
    selectedSceneNumber.value = scenes.value[0]?.scene ?? null;
  }
});

watch(selected, (scene) => {
  draft.value = scene ? { ...scene } : {};
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

async function rerenderSelected() {
  if (selectedSceneNumber.value) await studio.startJob(projectId.value, "ltx-render-scenes", [selectedSceneNumber.value]);
}
</script>

<template>
  <section class="page">
    <ProjectNav :project-id="projectId" />
    <header class="page-header toolbar-header">
      <div>
        <h1>Render Plan</h1>
        <p>{{ planPath || "No render plan found." }}</p>
      </div>
      <button class="button" :disabled="!selectedSceneNumber" @click="rerenderSelected"><Play :size="18" /> Rerender</button>
    </header>
    <div v-if="scenes.length" class="editor-layout">
      <section class="panel scene-table">
        <button v-for="scene in scenes" :key="scene.scene" :class="{ active: scene.scene === selectedSceneNumber }" @click="toggleScene(scene)">
          <strong>Scene {{ scene.scene }}</strong>
          <span>{{ String(scene.prompt ?? scene.msr_prompt ?? scene.lyrics ?? "").slice(0, 120) }}</span>
        </button>
      </section>
      <section class="document-panel">
        <header>
          <h2>Scene {{ selectedSceneNumber }}</h2>
          <button class="button" @click="saveScene"><Save :size="18" /> Save</button>
        </header>
        <label>
          Prompt
          <textarea v-model="(draft.prompt as string)" class="transcript-area" />
        </label>
        <label>
          MSR Prompt
          <textarea v-model="(draft.msr_prompt as string)" class="transcript-area" />
        </label>
        <JsonEditor v-model="draft" />
      </section>
    </div>
  </section>
</template>
