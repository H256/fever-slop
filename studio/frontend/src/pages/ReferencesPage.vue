<script setup lang="ts">
import { computed, onMounted } from "vue";
import { RefreshCw, WandSparkles } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import MediaPreview from "../components/MediaPreview.vue";
import ProjectNav from "../components/ProjectNav.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const referenceImages = computed(() => studio.currentProject?.artifacts.images.filter((path) => path.includes("reference")) ?? []);
const manifests = computed(() => studio.currentProject?.artifacts.references.filter((path) => path.endsWith(".json")) ?? []);

onMounted(() => studio.loadProject(projectId.value));
</script>

<template>
  <section class="page">
    <ProjectNav :project-id="projectId" />
    <header class="page-header toolbar-header">
      <div>
        <h1>References</h1>
        <p>Actor and location reference manifests and generated stills.</p>
      </div>
      <div class="button-row">
        <button class="button" @click="studio.startJob(projectId, 'msr-references')"><RefreshCw :size="18" /> Render refs</button>
        <button class="button secondary" @click="studio.startJob(projectId, 'msr-enrich')"><WandSparkles :size="18" /> Rebuild plan</button>
      </div>
    </header>
    <section class="panel">
      <h2>Manifests</h2>
      <div class="path-list">
        <RouterLink v-for="path in manifests" :key="path" :to="`/projects/${projectId}/artifacts?path=${encodeURIComponent(path)}`">{{ path }}</RouterLink>
      </div>
    </section>
    <div class="media-grid">
      <MediaPreview v-for="path in referenceImages" :key="path" :project-id="projectId" :path="path" />
    </div>
  </section>
</template>
