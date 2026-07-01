<script setup lang="ts">
import { computed, onMounted } from "vue";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import StatusBadge from "../components/StatusBadge.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const sizeEntries = computed(() => {
  const sizes = studio.currentProject?.artifact_sizes?.by_type ?? {};
  return Object.entries(sizes)
    .filter(([, bytes]) => bytes > 0)
    .sort((a, b) => b[1] - a[1]);
});
const totalBytes = computed(() => studio.currentProject?.artifact_sizes?.total_bytes ?? 0);
const chartColors = ["#5b5ce2", "#10a37f", "#f2b705", "#e85d75", "#3388dd", "#8f95a3", "#9b59b6", "#555"];
onMounted(() => studio.loadProject(projectId.value));

const pieStyle = computed(() => {
  if (!totalBytes.value || sizeEntries.value.length === 0) return { background: "#eef0f4" };
  let cursor = 0;
  const stops = sizeEntries.value.map(([, bytes], index) => {
    const start = cursor;
    cursor += (bytes / totalBytes.value) * 100;
    return `${chartColors[index % chartColors.length]} ${start}% ${cursor}%`;
  });
  return { background: `conic-gradient(${stops.join(", ")})` };
});

function legendColor(index: number): Record<string, string> {
  return { background: chartColors[index % chartColors.length] };
}

function percent(bytes: number): string {
  return totalBytes.value ? `${((bytes / totalBytes.value) * 100).toFixed(1)}%` : "0%";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unit]}`;
}
</script>

<template>
  <section v-if="studio.currentProject" class="page">    <header class="page-header">
      <h1>{{ studio.currentProject.name }}</h1>
      <p>{{ studio.currentProject.path }}</p>
    </header>
    <div class="dashboard-grid">
      <section class="panel">
        <h2>Pipeline Status</h2>
        <div class="status-list">
          <div v-for="(status, name) in studio.currentProject.status" :key="name">
            <span>{{ name }}</span>
            <StatusBadge :status="status" />
          </div>
        </div>
      </section>
      <section class="panel">
        <h2>Artifacts</h2>
        <div class="artifact-counts">
          <div v-for="(items, name) in studio.currentProject.artifacts" :key="name">
            <strong>{{ items.length }}</strong>
            <span>{{ name }}</span>
          </div>
        </div>
      </section>
      <section class="panel project-size-panel">
        <div class="panel-header">
          <h2>Project Size</h2>
          <strong>{{ formatBytes(totalBytes) }}</strong>
        </div>
        <div class="size-breakdown">
          <div class="size-pie" :style="pieStyle" />
          <div class="size-legend">
          <div v-for="([name, bytes], index) in sizeEntries" :key="name" class="size-chart-row">
            <span class="size-swatch" :style="legendColor(index)" />
            <div>
              <strong>{{ name }}</strong>
              <span>{{ formatBytes(bytes) }} · {{ percent(bytes) }}</span>
            </div>
          </div>
          </div>
        </div>
      </section>
    </div>
  </section>
</template>
