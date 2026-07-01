<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { Clapperboard, Play, Scissors } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api, mediaUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import type { RenderScene } from "../types";

interface TimelineItem {
  scene: number;
  start: number;
  end: number;
  duration: number;
  finalClip: string;
  rawClip: string;
  clip: string;
  status: "final" | "raw" | "missing";
  preview: string;
}

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const planPath = ref("");
const scenes = ref<RenderScene[]>([]);
const selectedScene = ref<number | null>(null);
const scrubSeconds = ref(0);
const pendingRetake = ref(false);
const pendingRecut = ref(false);
const playingTimeline = ref(false);
const cutIn = ref(0);
const cutOut = ref(0);
const videoRef = ref<HTMLVideoElement | null>(null);
const allVideos = computed(() => studio.currentProject?.artifacts.videos ?? []);
const allImages = computed(() => studio.currentProject?.artifacts.images ?? []);
const totalDuration = computed(() => Math.max(...scenes.value.map((scene) => Number(scene.abs_end_seconds ?? 0)), 0));
const timelineItems = computed<TimelineItem[]>(() =>
  scenes.value.map((scene) => {
    const sceneNumber = Number(scene.scene);
    const finalClip = findSceneClip(sceneNumber, false);
    const rawClip = findSceneClip(sceneNumber, true);
    const clip = finalClip || rawClip;
    return {
      scene: sceneNumber,
      start: Number(scene.abs_start_seconds ?? 0),
      end: Number(scene.abs_end_seconds ?? 0),
      duration: Number(scene.duration_seconds ?? 0),
      finalClip,
      rawClip,
      clip,
      status: finalClip ? "final" : rawClip ? "raw" : "missing",
      preview: scenePreview(scene)
    };
  })
);
const selectedItem = computed(() => timelineItems.value.find((item) => item.scene === selectedScene.value) ?? timelineItems.value[0]);
const selectedClipUrl = computed(() => (selectedItem.value?.clip ? mediaUrl(projectId.value, selectedItem.value.clip) : ""));
const playableItems = computed(() => timelineItems.value.filter((item) => item.clip));
const otherMedia = computed(() => [...allVideos.value, ...allImages.value].filter((path) => !isTimelineMedia(path)));

onMounted(async () => {
  await studio.loadProject(projectId.value);
  planPath.value = studio.currentProject?.artifacts.render_plans[0] ?? "";
  if (planPath.value) {
    scenes.value = (await api.artifact(projectId.value, planPath.value)).data as RenderScene[];
    selectedScene.value = Number(scenes.value[0]?.scene ?? 0) || null;
  }
});

watch(selectedItem, (item) => {
  const scene = scenes.value.find((candidate) => Number(candidate.scene) === item?.scene);
  const edit = (scene?.edit ?? {}) as Record<string, unknown>;
  cutIn.value = Number(edit.raw_in_seconds ?? 0);
  cutOut.value = Number(edit.raw_out_seconds ?? item?.duration ?? 0);
});

function selectItem(item: TimelineItem) {
  selectedScene.value = item.scene;
  scrubSeconds.value = item.start;
  seekPreview();
}

async function scrub() {
  playingTimeline.value = false;
  const item = timelineItems.value.find((candidate) => scrubSeconds.value >= candidate.start && scrubSeconds.value <= candidate.end);
  if (item) selectedScene.value = item.scene;
  await seekPreview();
}

async function seekPreview() {
  await nextTick();
  const video = videoRef.value;
  const item = selectedItem.value;
  if (!video || !item?.clip) return;
  video.currentTime = Math.max(0, scrubSeconds.value - item.start);
}

async function runRetake() {
  pendingRetake.value = false;
  if (selectedScene.value) await studio.startJob(projectId.value, "ltx-render-scenes", [selectedScene.value]);
}

async function playTimeline() {
  const first = playableItems.value[0];
  if (!first) return;
  playingTimeline.value = true;
  selectItem(first);
  await nextTick();
  await videoRef.value?.play();
}

async function playNextClip() {
  if (!playingTimeline.value || !selectedItem.value) return;
  const index = playableItems.value.findIndex((item) => item.scene === selectedItem.value?.scene);
  const next = playableItems.value[index + 1];
  if (!next) {
    playingTimeline.value = false;
    return;
  }
  selectItem(next);
  await nextTick();
  await videoRef.value?.play();
}

function syncScrubber() {
  if (!selectedItem.value || !videoRef.value) return;
  scrubSeconds.value = selectedItem.value.start + videoRef.value.currentTime;
}

async function runRecut() {
  pendingRecut.value = false;
  const item = selectedItem.value;
  if (!item?.rawClip || !planPath.value) return;
  const outputClip = item.finalClip || derivedFinalClip(item.rawClip, item.scene);
  const scene = scenes.value.find((candidate) => Number(candidate.scene) === item.scene);
  if (scene) {
    scene.edit = { ...((scene.edit ?? {}) as Record<string, unknown>), raw_in_seconds: cutIn.value, raw_out_seconds: cutOut.value };
    await api.saveArtifact(projectId.value, planPath.value, scenes.value);
  }
  const job = await studio.startJob(projectId.value, "recut-scene", undefined, {
    raw_clip: item.rawClip,
    output_clip: outputClip,
    raw_in_seconds: cutIn.value,
    raw_out_seconds: cutOut.value
  });
  const timer = window.setInterval(async () => {
    const jobs = await api.jobs(projectId.value);
    const current = jobs.find((candidate) => candidate.id === job.id);
    if (!current || current.status === "queued" || current.status === "running") return;
    window.clearInterval(timer);
    await studio.loadProject(projectId.value);
  }, 1500);
}

function findSceneClip(sceneNumber: number, raw: boolean): string {
  const padded = String(sceneNumber).padStart(4, "0");
  const candidates = allVideos.value.filter((path) => {
    if (!path.includes(`/scene_${padded}`)) return false;
    if (path.includes("_debug/")) return false;
    return raw ? path.includes("_raw") || path.includes("/raw/") : !path.includes("_raw") && !path.includes("/raw/");
  });
  return candidates.find((path) => path.includes("/final/")) ?? candidates[0] ?? "";
}

function isTimelineMedia(path: string): boolean {
  return timelineItems.value.some((item) => item.finalClip === path || item.rawClip === path);
}

function derivedFinalClip(rawClip: string, sceneNumber: number): string {
  const padded = String(sceneNumber).padStart(4, "0");
  if (rawClip.includes("/raw/")) return rawClip.replace("/raw/", "/final/").replace(`scene_${padded}_raw`, `scene_${padded}`);
  return rawClip.replace(`scene_${padded}_raw`, `scene_${padded}`);
}

function blockStyle(item: TimelineItem): Record<string, string> {
  const total = totalDuration.value || 1;
  return {
    left: `${(item.start / total) * 100}%`,
    width: `${Math.max((item.duration / total) * 100, 1)}%`
  };
}

function scenePreview(scene: RenderScene): string {
  return String(readPath(scene, ["ltx", "base_prompt"]) ?? readPath(scene, ["z_image", "prompt"]) ?? readPath(scene, ["metadata", "lyrics"]) ?? "");
}

function readPath(value: unknown, path: string[]): unknown {
  let current = value;
  for (const part of path) {
    if (!current || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function formatTime(value: number): string {
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Review</h1>
        <p>Timeline preview from the active render plan and available raw/final scene clips.</p>
      </div>
      <div class="button-row">
        <button class="button secondary" :disabled="playableItems.length === 0" @click="playTimeline"><Play :size="18" /> Play timeline</button>
        <button class="button" :disabled="!selectedScene" @click="pendingRetake = true"><Clapperboard :size="18" /> Render retake</button>
      </div>
    </header>

    <section class="timeline-editor">
      <div class="timeline-preview panel">
        <video
          v-if="selectedClipUrl"
          ref="videoRef"
          :key="selectedClipUrl"
          :src="selectedClipUrl"
          controls
          @ended="playNextClip"
          @timeupdate="syncScrubber"
        />
        <div v-else class="timeline-missing-preview">No clip exists for selected scene.</div>
        <aside v-if="selectedItem">
          <h2>Scene {{ selectedItem.scene }}</h2>
          <span class="status-badge" :class="selectedItem.status">{{ selectedItem.status }}</span>
          <p>{{ selectedItem.preview }}</p>
          <dl>
            <div><dt>Start</dt><dd>{{ formatTime(selectedItem.start) }}</dd></div>
            <div><dt>End</dt><dd>{{ formatTime(selectedItem.end) }}</dd></div>
            <div><dt>Duration</dt><dd>{{ selectedItem.duration.toFixed(2) }}s</dd></div>
          </dl>
          <div class="cut-controls">
            <label>
              Raw in
              <input v-model.number="cutIn" type="number" min="0" step="0.01" />
            </label>
            <label>
              Raw out
              <input v-model.number="cutOut" type="number" min="0" step="0.01" />
            </label>
            <button class="button secondary" :disabled="!selectedItem.rawClip || cutOut <= cutIn" @click="pendingRecut = true">
              <Scissors :size="18" /> Recut
            </button>
          </div>
        </aside>
      </div>

      <section class="panel timeline-panel">
        <div class="timeline-scrubber">
          <span>{{ formatTime(scrubSeconds) }}</span>
          <input v-model.number="scrubSeconds" type="range" min="0" :max="totalDuration" step="0.01" @input="scrub" />
          <span>{{ formatTime(totalDuration) }}</span>
        </div>
        <div class="timeline-track">
          <button
            v-for="item in timelineItems"
            :key="item.scene"
            class="timeline-clip"
            :class="[item.status, { active: item.scene === selectedScene }]"
            :style="blockStyle(item)"
            @click="selectItem(item)"
          >
            <strong>Scene {{ item.scene }}</strong>
            <small>{{ item.status === "missing" ? "Gap: no clip" : item.status === "raw" ? "Raw only" : "Final clip" }}</small>
          </button>
        </div>
      </section>
    </section>

    <details v-if="otherMedia.length" class="panel collapsed-list">
      <summary>Other media artifacts</summary>
      <div class="path-list">
        <RouterLink v-for="path in otherMedia" :key="path" :to="`/projects/${projectId}/artifacts?path=${encodeURIComponent(path)}`">{{ path }}</RouterLink>
      </div>
    </details>

    <ConfirmDialog
      :open="pendingRetake"
      title="Render retake?"
      :message="`This will rerender scene ${selectedScene}. Existing generated scene clips may be overwritten and jobs cannot be cancelled yet.`"
      confirm-label="Render retake"
      @cancel="pendingRetake = false"
      @confirm="runRetake"
    />
    <ConfirmDialog
      :open="pendingRecut"
      title="Recut scene?"
      :message="`This will trim the raw clip for scene ${selectedScene} and overwrite the derived scene clip. The raw clip remains unchanged.`"
      confirm-label="Recut scene"
      @cancel="pendingRecut = false"
      @confirm="runRecut"
    />
  </section>
</template>
