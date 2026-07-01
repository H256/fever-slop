<script setup lang="ts">
import { computed, onMounted } from "vue";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import MediaPreview from "../components/MediaPreview.vue";
import ProjectNav from "../components/ProjectNav.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const videos = computed(() => studio.currentProject?.artifacts.videos ?? []);
const images = computed(() => studio.currentProject?.artifacts.images.slice(0, 24) ?? []);
onMounted(() => studio.loadProject(projectId.value));
</script>

<template>
  <section class="page">
    <ProjectNav :project-id="projectId" />
    <header class="page-header">
      <h1>Review</h1>
      <p>Inspect generated clips, videos, and stills.</p>
    </header>
    <div class="media-grid">
      <MediaPreview v-for="path in videos" :key="path" :project-id="projectId" :path="path" />
      <MediaPreview v-for="path in images" :key="path" :project-id="projectId" :path="path" />
    </div>
  </section>
</template>
