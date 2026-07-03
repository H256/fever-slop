<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { Scissors } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api, mediaUrl, thumbnailUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import ReviewConfirmDialogs from "../components/ReviewConfirmDialogs.vue";
import ReviewTimelineClipLanes from "../components/ReviewTimelineClipLanes.vue";
import ReviewToolbar from "../components/ReviewToolbar.vue";
import ReviewTransport from "../components/ReviewTransport.vue";
import { applyBoundaryTrim, type ClipEdit } from "../lib/timelineTrim";
import { buildClipEdit, buildTimelineItems, derivedFinalClip, type RenderManifestEntry, type TimelineItem } from "../composables/reviewTimeline";
import { useReviewTimelineEdits } from "../composables/reviewTimelineEdits";
import { isTimelineMedia as pathIsTimelineMedia, rawPreviewForEdit, renderManifestByScene, type RawPreview } from "../composables/reviewTimelineMedia";
import { previewStart } from "../composables/reviewTimelinePlayback";
import { useReviewTimelinePlayback } from "../composables/reviewTimelinePlaybackState";
import { blockStyle as timelineBlockStyle, buildThumbnailRequests, formatTime, thumbnailFrameTimes, timelineTicks as buildTimelineTicks } from "../composables/reviewTimelinePresentation";
import { useTimelineHistory } from "../composables/timelineHistory";
import type { RenderScene } from "../types";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const planPath = ref("");
const scenes = ref<RenderScene[]>([]);
const selectedScene = ref<number | null>(null);
const scrubSeconds = ref(0);
const pendingRetake = ref(false);
const pendingRecut = ref(false);
const pendingTimelineSave = ref(false);
const playingTimeline = ref(false);
const cutIn = ref(0);
const cutOut = ref(0);
const exactRecut = ref(true);
const renderManifest = ref<Record<number, RenderManifestEntry>>({});
const timelineDirty = ref(false);
const timelineZoom = ref(1);
const rawPreview = ref<RawPreview | null>(null);
const videoRef = ref<HTMLVideoElement | null>(null);
const audioRef = ref<HTMLAudioElement | null>(null);
const waveformCanvas = ref<HTMLCanvasElement | null>(null);
let waveformRun = 0;
const { undoStack, redoStack, pushUndo, undoTimeline: restoreUndoSnapshot, redoTimeline: restoreRedoSnapshot } = useTimelineHistory(scenes);
const allVideos = computed(() => studio.currentProject?.artifacts.videos ?? []);
const allImages = computed(() => studio.currentProject?.artifacts.images ?? []);
const audioSource = computed(() => studio.currentProject?.artifacts.audio?.[0] ?? "");
const timelineItems = computed<TimelineItem[]>(() => buildTimelineItems({ scenes: scenes.value, videos: allVideos.value, manifest: renderManifest.value }));
const clipEdits = computed(() => scenes.value.map((scene) => buildClipEdit(scene, renderManifest.value[Number(scene.scene)])));
const totalDuration = computed(() => Math.max(0, ...timelineItems.value.map((item) => item.end)));
const selectedItem = computed(() => timelineItems.value.find((item) => item.scene === selectedScene.value) ?? timelineItems.value[0]);
const selectedClipUrl = computed(() => {
  const clip = rawPreview.value?.clip || selectedItem.value?.clip;
  return clip ? mediaUrl(projectId.value, clip) : "";
});
const audioUrl = computed(() => (audioSource.value ? mediaUrl(projectId.value, audioSource.value) : ""));
const playableItems = computed(() => timelineItems.value.filter((item) => item.clip));
const {
  applyClipEdits,
  clearSceneStale,
  editSeconds,
  isSceneStale,
  markChangedScenesStale,
  sceneFor: sceneDurationScene,
  sceneFps,
  staleScenes
} = useReviewTimelineEdits(scenes, clipEdits);
const otherMedia = computed(() => [...allVideos.value, ...allImages.value].filter((path) => !pathIsTimelineMedia(path, timelineItems.value)));
const timelineScaleStyle = computed(() => ({ width: `${timelineZoom.value * 100}%`, minWidth: "100%", "--timeline-zoom": String(timelineZoom.value) }));
const playheadStyle = computed(() => ({ left: `${((scrubSeconds.value || 0) / (totalDuration.value || 1)) * 100}%` }));
const timelineTicks = computed(() => buildTimelineTicks(totalDuration.value));
const { pauseAudio, playAudio, playNextClip, playTimeline, scrub, seekPreview, selectItem, stopTimeline, syncScrubber } = useReviewTimelinePlayback({
  timelineItems,
  playableItems,
  selectedItem,
  rawPreview,
  selectedScene,
  scrubSeconds,
  playingTimeline,
  videoRef,
  audioRef
});

onMounted(async () => {
  await studio.loadProject(projectId.value);
  planPath.value = studio.currentProject?.artifacts.render_plans[0] ?? "";
  await loadRenderManifest();
  if (planPath.value) {
    scenes.value = (await api.artifact(projectId.value, planPath.value)).data as RenderScene[];
    selectedScene.value = Number(scenes.value[0]?.scene ?? 0) || null;
  }
  await drawWaveform();
});

watch(audioUrl, () => drawWaveform());

watch(selectedItem, (item) => {
  const scene = scenes.value.find((candidate) => Number(candidate.scene) === item?.scene);
  const edit = (scene?.edit ?? {}) as Record<string, unknown>;
  const seconds = item ? editSeconds(item.scene) : { in: 0, out: 0 };
  cutIn.value = Number(edit.raw_in_seconds ?? seconds.in);
  cutOut.value = Number(edit.raw_out_seconds ?? seconds.out);
});

async function runRetake() {
  pendingRetake.value = false;
  if (selectedScene.value) await studio.startJob(projectId.value, "ltx-render-scenes", [selectedScene.value]);
}

async function saveTimeline() {
  pendingTimelineSave.value = false;
  if (!planPath.value) return;
  await api.saveArtifact(projectId.value, planPath.value, scenes.value);
  timelineDirty.value = false;
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
    raw_out_seconds: cutOut.value,
    exact: exactRecut.value
  });
  const timer = window.setInterval(async () => {
    const jobs = await api.jobs(projectId.value);
    const current = jobs.find((candidate) => candidate.id === job.id);
    if (!current || current.status === "queued" || current.status === "running") return;
    window.clearInterval(timer);
    clearSceneStale(item.scene);
    await api.saveArtifact(projectId.value, planPath.value, scenes.value);
    await studio.loadProject(projectId.value);
  }, 1500);
}

async function rebuildStaleScenes() {
  if (!staleScenes.value.length) return;
  const jobs = timelineItems.value
    .filter((item) => staleScenes.value.includes(item.scene) && item.rawClip)
    .map((item) =>
      studio.startJob(projectId.value, "recut-scene", undefined, {
        raw_clip: item.rawClip,
        output_clip: item.finalClip || derivedFinalClip(item.rawClip, item.scene),
        raw_in_seconds: editSeconds(item.scene).in,
        raw_out_seconds: editSeconds(item.scene).out,
        exact: exactRecut.value
      })
    );
  await Promise.all(jobs);
}

async function prebuildThumbnails() {
  await studio.startJob(projectId.value, "thumbnail-prebuild", undefined, { thumbnails: thumbnailRequests() });
}

async function cleanupThumbnails() {
  await studio.startJob(projectId.value, "thumbnail-cleanup");
}

async function loadRenderManifest() {
  const manifestPath = studio.currentProject?.artifacts.generated_json.find((path) => path.endsWith("render_manifest.json"));
  if (!manifestPath) return;
  const data = (await api.artifact(projectId.value, manifestPath)).data;
  renderManifest.value = renderManifestByScene(data);
}

function startFinalDrag(event: PointerEvent, item: TimelineItem, mode: "left" | "right") {
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  selectItem(item);
  const track = (event.currentTarget as HTMLElement).closest(".timeline-track");
  const rect = track?.getBoundingClientRect();
  if (!rect?.width) return;
  pushUndo();
  const original = scenes.value.map((scene) => cloneScene(scene));
  const originalEdits = clipEdits.value.map((edit) => ({ ...edit }));
  const startX = event.clientX;
  const secondsPerPixel = (totalDuration.value || 1) / rect.width;
  updateRawPreview(item.scene, mode, originalEdits);
  const move = (moveEvent: PointerEvent) => {
    scenes.value = original.map((scene) => cloneScene(scene));
    const deltaFrames = Math.round(((moveEvent.clientX - startX) * secondsPerPixel) / frameStep(item));
    const nextEdits = applyBoundaryTrim(originalEdits, { scene: item.scene, edge: mode, deltaFrames });
    applyClipEdits(nextEdits);
    updateRawPreview(item.scene, mode, nextEdits);
    markChangedScenesStale(originalEdits, nextEdits);
    timelineDirty.value = true;
  };
  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop);
}

function startCutDrag(event: PointerEvent, item: TimelineItem, edge: "in" | "out") {
  if (!item.rawClip) return;
  selectItem(item);
  const clip = (event.currentTarget as HTMLElement).closest(".timeline-clip");
  const rect = clip?.getBoundingClientRect();
  if (!rect?.width) return;
  const startX = event.clientX;
  const startIn = cutIn.value;
  const startOut = cutOut.value || item.rawDuration;
  const secondsPerPixel = Math.max(item.rawDuration, 0.1) / rect.width;
  const move = (moveEvent: PointerEvent) => {
    const delta = (moveEvent.clientX - startX) * secondsPerPixel;
    if (edge === "in") cutIn.value = Math.max(0, Math.min(snapSeconds(startIn + delta, item), cutOut.value - frameStep(item)));
    else cutOut.value = Math.max(cutIn.value + frameStep(item), snapSeconds(startOut + delta, item));
  };
  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop);
}

function startZoomDrag(event: PointerEvent) {
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  const startX = event.clientX;
  const startZoom = timelineZoom.value;
  const move = (moveEvent: PointerEvent) => {
    timelineZoom.value = Math.max(1, Math.min(8, startZoom + (moveEvent.clientX - startX) / 160));
  };
  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
    drawWaveform();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop);
}

function undoTimeline() {
  rawPreview.value = null;
  if (restoreUndoSnapshot()) timelineDirty.value = true;
}

function redoTimeline() {
  rawPreview.value = null;
  if (restoreRedoSnapshot()) timelineDirty.value = true;
}

function sceneIndex(sceneNumber: number): number {
  return scenes.value.findIndex((scene) => Number(scene.scene) === sceneNumber);
}

function cloneScene(scene: RenderScene): RenderScene {
  return JSON.parse(JSON.stringify(scene)) as RenderScene;
}

function snapSeconds(value: number, scene?: RenderScene | TimelineItem): number {
  const step = scene ? frameStep(scene) : 0.1;
  return Math.round(value / step) * step;
}

function frameStep(scene: RenderScene | TimelineItem): number {
  const fps = Number("fps" in scene ? scene.fps : sceneDurationScene(scene.scene)?.fps) || 24;
  return 1 / fps;
}

function snapNear(value: number, targets: number[]): number {
  return targets.find((target) => Math.abs(target - value) <= 0.25) ?? value;
}

function blockStyle(start: number, duration: number): Record<string, string> {
  return timelineBlockStyle(start, duration, totalDuration.value);
}

function thumbnailFrames(path: string, duration: number): { time: number; url: string }[] {
  return thumbnailFrameTimes(duration, timelineZoom.value).map((time) => ({ time, url: thumbnailUrl(projectId.value, path, time) }));
}

function thumbnailRequests(): { path: string; times: number[] }[] {
  return buildThumbnailRequests(timelineItems.value, timelineZoom.value);
}

function sourceWindowStyle(item: TimelineItem): Record<string, string> {
  const edit = clipEdits.value.find((candidate) => candidate.scene === item.scene);
  const scene = sceneDurationScene(item.scene);
  if (!edit || !scene) return blockStyle(item.start, item.duration);
  const fps = sceneFps(scene);
  return blockStyle(item.start - edit.rawInFrame / fps, (edit.maxRawOutFrame - edit.minRawInFrame) / fps);
}

function updateRawPreview(sceneNumber: number, mode: "left" | "right", edits: ClipEdit[]) {
  const preview = rawPreviewForEdit({ sceneNumber, mode, items: timelineItems.value, scenes: scenes.value, edits });
  if (preview) {
    rawPreview.value = preview;
    void seekPreview();
  }
}

async function seekWaveform(event: MouseEvent) {
  const target = event.currentTarget as HTMLElement;
  const rect = target.getBoundingClientRect();
  scrubSeconds.value = Math.max(0, Math.min(totalDuration.value, ((event.clientX - rect.left) / rect.width) * totalDuration.value));
  await scrub();
}

async function drawWaveform() {
  await nextTick();
  const canvas = waveformCanvas.value;
  const url = audioUrl.value;
  if (!canvas || !url) return;
  const run = ++waveformRun;
  const context = new AudioContext();
  try {
    const buffer = await fetch(url).then((response) => response.arrayBuffer()).then((data) => context.decodeAudioData(data));
    if (run !== waveformRun) return;
    const pixels = Math.max(240, canvas.clientWidth * window.devicePixelRatio);
    const height = Math.max(70, canvas.clientHeight * window.devicePixelRatio);
    canvas.width = pixels;
    canvas.height = height;
    const draw = canvas.getContext("2d");
    if (!draw) return;
    const samples = buffer.getChannelData(0);
    const step = Math.max(1, Math.floor(samples.length / pixels));
    draw.clearRect(0, 0, pixels, height);
    draw.fillStyle = "#eef0f4";
    draw.fillRect(0, 0, pixels, height);
    draw.strokeStyle = "#5b5ce2";
    draw.beginPath();
    for (let x = 0; x < pixels; x += 1) {
      let peak = 0;
      for (let i = 0; i < step; i += 1) peak = Math.max(peak, Math.abs(samples[x * step + i] ?? 0));
      const y = (peak * height) / 2;
      draw.moveTo(x, height / 2 - y);
      draw.lineTo(x, height / 2 + y);
    }
    draw.stroke();
  } catch {
    const draw = canvas.getContext("2d");
    draw?.clearRect(0, 0, canvas.width, canvas.height);
  } finally {
    void context.close();
  }
}

</script>

<template>
  <section class="page">
    <ReviewToolbar
      :redo-count="redoStack.length"
      :selected-scene="selectedScene"
      :stale-count="staleScenes.length"
      :timeline-dirty="timelineDirty"
      :timeline-zoom="timelineZoom"
      :undo-count="undoStack.length"
      @cleanup-thumbnails="cleanupThumbnails"
      @prebuild-thumbnails="prebuildThumbnails"
      @rebuild-stale-scenes="rebuildStaleScenes"
      @render-retake="pendingRetake = true"
      @save-timeline="pendingTimelineSave = true"
      @undo-timeline="undoTimeline"
      @redo-timeline="redoTimeline"
      @update-timeline-zoom="timelineZoom = $event"
      @zoom-input="drawWaveform"
    />

    <section class="timeline-editor">
      <div class="timeline-preview panel">
        <div class="timeline-preview-media">
          <video
            v-if="selectedClipUrl"
            ref="videoRef"
            :key="selectedClipUrl"
            :src="selectedClipUrl"
            controls
            @ended="playNextClip"
            @pause="pauseAudio"
            @play="playAudio"
            @timeupdate="syncScrubber"
          />
          <div v-else class="timeline-missing-preview">No clip exists for selected scene.</div>
          <div v-if="rawPreview" class="raw-preview-badge">
            Raw {{ rawPreview.edge }} preview at {{ rawPreview.seconds.toFixed(2) }}s
          </div>
        </div>
        <audio v-if="audioUrl" ref="audioRef" :src="audioUrl" preload="metadata" />
        <aside v-if="selectedItem">
          <h2>Scene {{ selectedItem.scene }}</h2>
          <span class="status-badge" :class="selectedItem.status">{{ selectedItem.status }}</span>
          <span v-if="isSceneStale(selectedItem.scene)" class="status-badge warning">
            stale
          </span>
          <p>{{ selectedItem.preview }}</p>
          <dl>
            <div><dt>Start</dt><dd>{{ formatTime(selectedItem.start) }}</dd></div>
            <div><dt>End</dt><dd>{{ formatTime(selectedItem.end) }}</dd></div>
            <div><dt>Duration</dt><dd>{{ selectedItem.duration.toFixed(2) }}s</dd></div>
            <div><dt>Raw</dt><dd>{{ formatTime(selectedItem.rawStart) }} - {{ formatTime(selectedItem.rawEnd) }}</dd></div>
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
            <label class="checkbox-row">
              <input v-model="exactRecut" type="checkbox" />
              Exact recut
            </label>
          </div>
        </aside>
      </div>

      <ReviewTransport
        :playable-count="playableItems.length"
        :playing-timeline="playingTimeline"
        :scrub-seconds="scrubSeconds"
        :total-duration="totalDuration"
        @play-timeline="playTimeline"
        @stop-timeline="stopTimeline"
        @update-scrub-seconds="scrubSeconds = $event"
        @scrub="scrub"
      />

      <section class="panel timeline-panel">
        <div class="timeline-scroll">
          <div class="timeline-lanes" :style="timelineScaleStyle">
          <ReviewTimelineClipLanes
            :block-style="blockStyle"
            :format-time="formatTime"
            :is-scene-stale="isSceneStale"
            :playhead-style="playheadStyle"
            :selected-scene="selectedScene"
            :source-window-style="sourceWindowStyle"
            :thumbnail-frames="thumbnailFrames"
            :timeline-items="timelineItems"
            :timeline-ticks="timelineTicks"
            @select-item="selectItem"
            @start-final-drag="startFinalDrag"
            @start-zoom-drag="startZoomDrag"
          />
          <div class="timeline-lane">
            <span class="timeline-lane-label">Wave</span>
            <div class="timeline-track waveform-track" @click="seekWaveform">
              <span class="timeline-playhead" :style="playheadStyle" />
              <canvas v-if="audioUrl" ref="waveformCanvas" class="waveform-canvas" />
              <small v-else class="timeline-empty-note">No audio artifact found</small>
            </div>
          </div>
          </div>
        </div>
      </section>
    </section>

    <details v-if="otherMedia.length" class="panel collapsed-list">
      <summary>Other media artifacts</summary>
      <div class="path-list">
        <RouterLink v-for="path in otherMedia" :key="path" :to="`/projects/${projectId}/artifacts?path=${encodeURIComponent(path)}`">{{ path }}</RouterLink>
      </div>
    </details>

    <ReviewConfirmDialogs
      :pending-retake="pendingRetake"
      :pending-recut="pendingRecut"
      :pending-timeline-save="pendingTimelineSave"
      :selected-scene="selectedScene"
      @cancel-retake="pendingRetake = false"
      @confirm-retake="runRetake"
      @cancel-recut="pendingRecut = false"
      @confirm-recut="runRecut"
      @cancel-timeline-save="pendingTimelineSave = false"
      @confirm-timeline-save="saveTimeline"
    />
  </section>
</template>
