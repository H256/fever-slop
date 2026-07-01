<script setup lang="ts">
import { Boxes, Clapperboard, FileJson, Gauge, Images, LayoutDashboard, ListVideo } from "lucide-vue-next";

defineProps<{ projectId: string }>();
</script>

<template>
  <div class="project-tabs">
    <RouterLink :to="`/projects/${projectId}`"><LayoutDashboard :size="16" /> Dashboard</RouterLink>
    <RouterLink :to="`/projects/${projectId}/pipeline`"><Gauge :size="16" /> Pipeline</RouterLink>
    <RouterLink :to="`/projects/${projectId}/render-plan`"><ListVideo :size="16" /> Render Plan</RouterLink>
    <RouterLink :to="`/projects/${projectId}/references`"><Images :size="16" /> References</RouterLink>
    <RouterLink :to="`/projects/${projectId}/artifacts`"><FileJson :size="16" /> Artifacts</RouterLink>
    <RouterLink :to="`/projects/${projectId}/queue`"><Boxes :size="16" /> Queue</RouterLink>
    <RouterLink :to="`/projects/${projectId}/review`"><Clapperboard :size="16" /> Review</RouterLink>
  </div>
</template>
