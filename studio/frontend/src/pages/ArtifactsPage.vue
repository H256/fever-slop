<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Save } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api } from "../api";
import { useStudioStore } from "../stores/studio";
import JsonEditor from "../components/JsonEditor.vue";
import ProjectNav from "../components/ProjectNav.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const selectedPath = ref("");
const data = ref<unknown>(null);
const jsonPaths = computed(() => [
  ...(studio.currentProject?.artifacts.configs ?? []),
  ...(studio.currentProject?.artifacts.render_plans ?? []),
  ...(studio.currentProject?.artifacts.generated_json ?? [])
]);

onMounted(async () => {
  await studio.loadProject(projectId.value);
  selectedPath.value = String(route.query.path ?? jsonPaths.value[0] ?? "");
});

watch(selectedPath, async (path) => {
  if (!path) return;
  data.value = (await api.artifact(projectId.value, path)).data;
});

async function save() {
  if (selectedPath.value) await api.saveArtifact(projectId.value, selectedPath.value, data.value);
}
</script>

<template>
  <section class="page">
    <ProjectNav :project-id="projectId" />
    <header class="page-header toolbar-header">
      <div>
        <h1>Artifacts</h1>
        <p>Edit project JSON files directly.</p>
      </div>
      <button class="button" :disabled="!selectedPath" @click="save"><Save :size="18" /> Save</button>
    </header>
    <div class="editor-layout">
      <section class="panel path-list">
        <button v-for="path in jsonPaths" :key="path" :class="{ active: path === selectedPath }" @click="selectedPath = path">{{ path }}</button>
      </section>
      <section class="document-panel">
        <h2>{{ selectedPath }}</h2>
        <JsonEditor v-if="selectedPath" v-model="data" />
      </section>
    </div>
  </section>
</template>
