<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { Download, Film, Wrench } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { mediaUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import type { Job } from "../types";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const videos = computed(() => studio.currentProject?.artifacts.videos ?? []);
const finalVideo = computed(() => pickFinalVideo(videos.value));
const mediaVersion = ref(0);
const finalVideoUrl = computed(() => (finalVideo.value ? `${mediaUrl(projectId.value, finalVideo.value)}&v=${mediaVersion.value}` : ""));
const isMovieProject = computed(() => studio.currentProject?.project_type === "movie");
const startingConcat = ref(false);
const concatError = ref("");
const hasActiveProcess = computed(() => studio.jobs.some((job) => ["queued", "running"].includes(job.status)));
const buildDisabledReason = computed(() => {
  if (hasActiveProcess.value) return "Build final movie is disabled until the current job finishes.";
  return "";
});

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await studio.loadJobs(projectId.value);
  mediaVersion.value = Date.now();
});

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

async function buildFinalMovie() {
  if (hasActiveProcess.value) {
    concatError.value = buildDisabledReason.value;
    return;
  }
  startingConcat.value = true;
  concatError.value = "";
  try {
    const job = await studio.startJob(projectId.value, "movie-final-concat");
    await waitForJob(job.id);
    await studio.loadJobs(projectId.value);
    await studio.loadProject(projectId.value);
    mediaVersion.value = Date.now();
  } catch (caught) {
    concatError.value = apiErrorMessage(caught);
  } finally {
    startingConcat.value = false;
  }
}

async function waitForJob(jobId: string): Promise<Job> {
  for (;;) {
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
    await studio.loadJobs(projectId.value);
    const job = studio.jobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status === "queued" || job.status === "running") continue;
    if (job.status === "failed") throw new Error(job.error || "Build final movie failed");
    return job;
  }
}

function apiErrorMessage(caught: unknown): string {
  if (!(caught instanceof Error)) return String(caught);
  const body = "body" in caught ? String((caught as { body?: string }).body ?? "") : "";
  if (!body) return caught.message;
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    return parsed.detail ? String(parsed.detail) : body;
  } catch {
    return body;
  }
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Final Video</h1>
        <p>Preview and download the final rendered video for this project.</p>
      </div>
      <div class="button-row">
        <button v-if="isMovieProject" class="button secondary" :disabled="startingConcat || hasActiveProcess" :title="buildDisabledReason" @click="buildFinalMovie">
          <Wrench :size="18" /> {{ startingConcat ? "Starting..." : "Build final movie" }}
        </button>
        <a v-if="finalVideo" class="button" :href="finalVideoUrl" download>
          <Download :size="18" /> Download
        </a>
      </div>
    </header>

    <section v-if="!finalVideo" class="panel empty">
      <Film :size="22" />
      <p>No final video was found yet. Run final concat or the full pipeline first.</p>
    </section>
    <section v-if="hasActiveProcess" class="panel notice-panel">
      <h2>A backend process is running</h2>
      <p>{{ buildDisabledReason }}</p>
    </section>
    <section v-if="concatError" class="panel notice-panel">
      <h2>Build failed</h2>
      <p>{{ concatError }}</p>
    </section>

    <section v-else class="panel final-video-panel">
      <video :key="finalVideoUrl" :src="finalVideoUrl" controls playsinline />
      <div class="artifact-path">{{ finalVideo }}</div>
    </section>
  </section>
</template>
