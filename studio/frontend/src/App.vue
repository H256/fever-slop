<script setup lang="ts">
import { computed } from "vue";
import { Boxes, Clapperboard, FileJson, FolderKanban, Gauge, Images, LayoutDashboard, ListVideo, Settings } from "lucide-vue-next";
import { useRoute } from "vue-router";

const route = useRoute();
const projectId = computed(() => (typeof route.params.projectId === "string" ? route.params.projectId : ""));
</script>

<template>
  <div class="studio-shell">
    <aside class="sidebar">
      <RouterLink class="brand" to="/">
        <span class="brand-mark">FS</span>
        <span>FeverSlop Studio</span>
      </RouterLink>
      <nav>
        <RouterLink to="/"><FolderKanban :size="18" /> Projects</RouterLink>
        <RouterLink to="/settings"><Settings :size="18" /> Settings</RouterLink>
      </nav>
      <nav v-if="projectId" class="sidebar-project-nav">
        <span class="sidebar-label">{{ projectId }}</span>
        <RouterLink :to="`/projects/${projectId}`"><LayoutDashboard :size="18" /> Dashboard</RouterLink>
        <RouterLink :to="`/projects/${projectId}/pipeline`"><Gauge :size="18" /> Pipeline</RouterLink>
        <RouterLink :to="`/projects/${projectId}/render-plan`"><ListVideo :size="18" /> Render Plan</RouterLink>
        <RouterLink :to="`/projects/${projectId}/references`"><Images :size="18" /> References</RouterLink>
        <RouterLink :to="`/projects/${projectId}/settings`"><Settings :size="18" /> Project Settings</RouterLink>
        <RouterLink :to="`/projects/${projectId}/artifacts`"><FileJson :size="18" /> Artifacts</RouterLink>
        <RouterLink :to="`/projects/${projectId}/queue`"><Boxes :size="18" /> Queue</RouterLink>
        <RouterLink :to="`/projects/${projectId}/review`"><Clapperboard :size="18" /> Review</RouterLink>
      </nav>
    </aside>
    <main class="main-pane">
      <RouterView />
    </main>
  </div>
</template>
