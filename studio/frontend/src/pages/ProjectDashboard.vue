<script setup lang="ts">
import { computed, onMounted } from "vue";
import { useRoute } from "vue-router";
import { useStudioStore } from "../stores/studio";
import StatusBadge from "../components/StatusBadge.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
onMounted(() => studio.loadProject(projectId.value));
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
    </div>
  </section>
</template>
