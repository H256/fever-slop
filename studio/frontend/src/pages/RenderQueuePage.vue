<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import JobDrawer from "../components/JobDrawer.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
let pollTimer: number | undefined;

onMounted(() => {
  studio.loadJobs(projectId.value);
  pollTimer = window.setInterval(() => studio.loadJobs(projectId.value), 2000);
});

onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer);
});
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h1>Render Queue</h1>
      <p>In-memory Studio jobs for this server session. The list refreshes every 2 seconds.</p>
    </header>
    <JobDrawer :jobs="studio.jobs" />
  </section>
</template>
