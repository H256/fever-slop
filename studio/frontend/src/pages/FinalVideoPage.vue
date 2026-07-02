<script setup lang="ts">
import { computed, onMounted } from "vue";
import { Download, Film } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { mediaUrl } from "../api";
import { useStudioStore } from "../stores/studio";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const videos = computed(() => studio.currentProject?.artifacts.videos ?? []);
const finalVideo = computed(() => pickFinalVideo(videos.value));
const finalVideoUrl = computed(() => (finalVideo.value ? mediaUrl(projectId.value, finalVideo.value) : ""));

onMounted(() => studio.loadProject(projectId.value));

function pickFinalVideo(paths: string[]): string {
  const candidates = paths.filter((path) => !path.includes("_video_only") && !/scene_\d+\.mp4$/i.test(path) && !path.includes("/raw/"));
  return [...candidates, ...paths].sort((a, b) => scoreVideo(b) - scoreVideo(a))[0] ?? "";
}

function scoreVideo(path: string): number {
  let score = 0;
  if (path.includes("/ltx_msr/")) score += 20;
  if (path.includes("/ltx_single_prompt/")) score += 10;
  if (!path.includes("_video_only")) score += 10;
  if (!/scene_\d+\.mp4$/i.test(path)) score += 10;
  return score;
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Final Video</h1>
        <p>Preview and download the final rendered video for this project.</p>
      </div>
      <a v-if="finalVideo" class="button" :href="finalVideoUrl" download>
        <Download :size="18" /> Download
      </a>
    </header>

    <section v-if="!finalVideo" class="panel empty">
      <Film :size="22" />
      <p>No final video was found yet. Run final concat or the full pipeline first.</p>
    </section>

    <section v-else class="panel final-video-panel">
      <video :src="finalVideoUrl" controls playsinline />
      <div class="artifact-path">{{ finalVideo }}</div>
    </section>
  </section>
</template>
