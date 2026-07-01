<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { Play } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import JobDrawer from "../components/JobDrawer.vue";
import ConfirmDialog from "../components/ConfirmDialog.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const pendingAction = ref<(typeof actions)[number] | null>(null);
let pollTimer: number | undefined;
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
  pollTimer = window.setInterval(() => studio.loadJobs(projectId.value), 2000);
});

onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer);
});

function askToRun(action: (typeof actions)[number]) {
  pendingAction.value = action;
}

async function runConfirmed() {
  if (!pendingAction.value) return;
  const [action] = pendingAction.value;
  pendingAction.value = null;
  await studio.startJob(projectId.value, action);
  await studio.loadJobs(projectId.value);
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h1>Pipeline</h1>
      <p>Run existing FeverSlop pipeline stages as background jobs. Jobs can overwrite generated artifacts and cannot be cancelled yet.</p>
    </header>
    <div class="split">
      <section class="panel action-list">
        <button v-for="action in actions" :key="action[0]" class="action-button" @click="askToRun(action)">
          <Play :size="18" />
          <span>{{ action[1] }}</span>
        </button>
      </section>
      <JobDrawer :jobs="studio.jobs" />
    </div>
    <ConfirmDialog
      :open="Boolean(pendingAction)"
      title="Run pipeline job?"
      :message="`This will start '${pendingAction?.[1]}' for ${projectId}. It may overwrite generated project artifacts, can trigger external tools, and cannot be cancelled from Studio yet.`"
      confirm-label="Run job"
      @cancel="pendingAction = null"
      @confirm="runConfirmed"
    />
  </section>
</template>
