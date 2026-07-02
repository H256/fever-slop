<script setup lang="ts">
import { onMounted } from "vue";
import { Folder } from "lucide-vue-next";
import { useStudioStore } from "../stores/studio";
import type { ProjectSummary } from "../types";

const studio = useStudioStore();
onMounted(() => studio.loadProjects());

const badgeLabels: Record<string, string> = {
  config: "Config",
  render_plan: "Render plan",
  references: "References"
};

function presentBadges(project: ProjectSummary): string[] {
  return Object.entries(project.status)
    .filter(([, status]) => status === "present")
    .map(([name]) => name);
}
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
        <p class="project-folder"><Folder :size="15" /> {{ project.id }}</p>
        <div class="badge-row">
          <span
            v-for="name in presentBadges(project)"
            :key="name"
            class="status-badge project-artifact-badge"
            :class="`artifact-${name}`"
          >
            {{ badgeLabels[name] ?? name }}
          </span>
        </div>
      </RouterLink>
    </div>
  </section>
</template>
