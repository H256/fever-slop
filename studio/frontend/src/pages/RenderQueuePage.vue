<script setup lang="ts">
import { computed, onMounted } from "vue";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import JobDrawer from "../components/JobDrawer.vue";
import ProjectNav from "../components/ProjectNav.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
onMounted(() => studio.loadJobs(projectId.value));
</script>

<template>
  <section class="page">
    <ProjectNav :project-id="projectId" />
    <header class="page-header">
      <h1>Render Queue</h1>
      <p>In-memory Studio jobs for this server session.</p>
    </header>
    <JobDrawer :jobs="studio.jobs" />
  </section>
</template>
