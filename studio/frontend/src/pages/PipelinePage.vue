<script setup lang="ts">
import { computed, onMounted } from "vue";
import { Play } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import ProjectNav from "../components/ProjectNav.vue";
import JobDrawer from "../components/JobDrawer.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const actions = [
  ["main-pipeline", "Main pipeline"],
  ["msr-references", "MSR references"],
  ["msr-enrich", "MSR enrichment"],
  ["storyboard", "Storyboard"],
  ["ltx-render-scenes", "Render selected scenes"],
  ["final-concat", "Final concat"],
  ["full-pipeline", "Full pipeline"]
] as const;

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await studio.loadJobs(projectId.value);
});

async function run(action: string) {
  await studio.startJob(projectId.value, action);
  await studio.loadJobs(projectId.value);
}
</script>

<template>
  <section class="page">
    <ProjectNav :project-id="projectId" />
    <header class="page-header">
      <h1>Pipeline</h1>
      <p>Run existing FeverSlop pipeline stages as background jobs.</p>
    </header>
    <div class="split">
      <section class="panel action-list">
        <button v-for="[action, label] in actions" :key="action" class="action-button" @click="run(action)">
          <Play :size="18" />
          <span>{{ label }}</span>
        </button>
      </section>
      <JobDrawer :jobs="studio.jobs" />
    </div>
  </section>
</template>
