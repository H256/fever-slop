<script setup lang="ts">
import { onMounted } from "vue";
import { useStudioStore } from "../stores/studio";
import StatusBadge from "../components/StatusBadge.vue";

const studio = useStudioStore();
onMounted(() => studio.loadProjects());
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h1>Projects</h1>
      <p>Local FeverSlop project folders under <code>projects/</code>.</p>
    </header>
    <div class="project-grid">
      <RouterLink v-for="project in studio.projects" :key="project.id" class="project-card" :to="`/projects/${project.id}`">
        <h2>{{ project.name }}</h2>
        <p>{{ project.id }}</p>
        <div class="badge-row">
          <StatusBadge v-for="(status, name) in project.status" :key="name" :status="`${name}:${status}`" />
        </div>
      </RouterLink>
    </div>
  </section>
</template>
