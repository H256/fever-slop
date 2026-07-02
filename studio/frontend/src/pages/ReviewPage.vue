<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { Clapperboard, Play, Redo2, RotateCcw, Save, Scissors, Sparkles, Square, Trash2, Undo2, ZoomIn } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { api, mediaUrl, thumbnailUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import { applyBoundaryTrim, buildEditState, type ClipEdit } from "../lib/timelineTrim";
import type { RenderScene } from "../types";

interface TimelineItem {
  scene: number;
  start: number;
  end: number;
  duration: number;
  rawStart: number;
  rawEnd: number;
  rawDuration: number;
  finalClip: string;
  rawClip: string;
  clip: string;
  status: "final" | "raw" | "missing";
  preview: string;
  hasManifestTiming: boolean;
}

interface RenderManifestEntry {
  scene: number;
  audio_start_seconds?: number;
  audio_duration_seconds?: number;
  trim_front_frames?: number;
  scene_frame_count?: number;
  render_frame_count?: number;
  tail_loss_frames?: number;
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
const pendingTimelineSave = ref(false);
const playingTimeline = ref(false);
const cutIn = ref(0);
const cutOut = ref(0);
const exactRecut = ref(true);
const renderManifest = ref<Record<number, RenderManifestEntry>>({});
const timelineDirty = ref(false);
const timelineZoom = ref(1);
const undoStack = ref<RenderScene[][]>([]);
const redoStack = ref<RenderScene[][]>([]);
const rawPreview = ref<{ scene: number; clip: string; seconds: number; edge: "IN" | "OUT" } | null>(null);
const videoRef = ref<HTMLVideoElement | null>(null);
const audioRef = ref<HTMLAudioElement | null>(null);
const waveformCanvas = ref<HTMLCanvasElement | null>(null);
let waveformRun = 0;
const allVideos = computed(() => studio.currentProject?.artifacts.videos ?? []);
const allImages = computed(() => studio.currentProject?.artifacts.images ?? []);
const audioSource = computed(() => studio.currentProject?.artifacts.audio?.[0] ?? "");
const timelineItems = computed<TimelineItem[]>(() =>
  scenes.value.map((scene, index) => {
    const sceneNumber = Number(scene.scene);
    const edit = clipEdits.value[index];
    const fps = sceneFps(scene);
    const start = clipEdits.value
      .slice(0, index)
      .reduce((total, clip, clipIndex) => total + (clip.rawOutFrame - clip.rawInFrame) / sceneFps(scenes.value[clipIndex]), 0);
    const duration = edit ? (edit.rawOutFrame - edit.rawInFrame) / fps : Number(scene.duration_seconds ?? 0);
    const manifest = renderManifest.value[sceneNumber];
    const fallbackRaw = fallbackRawTiming(scene, start, duration);
    const rawStart = Number(manifest?.audio_start_seconds ?? fallbackRaw.start);
    const rawDuration = Number(manifest?.audio_duration_seconds ?? fallbackRaw.duration);
    const finalClip = findSceneClip(sceneNumber, false);
    const rawClip = findSceneClip(sceneNumber, true);
    const clip = finalClip || rawClip;
    return {
      scene: sceneNumber,
      start,
      end: start + duration,
      duration,
      rawStart,
      rawEnd: rawStart + rawDuration,
      rawDuration,
      finalClip,
      rawClip,
      clip,
      status: finalClip ? "final" : rawClip ? "raw" : "missing",
      preview: scenePreview(scene),
      hasManifestTiming: Boolean(manifest)
    };
  })
);
const clipEdits = computed(() => scenes.value.map((scene) => editForScene(scene)));
const totalDuration = computed(() => Math.max(0, ...timelineItems.value.map((item) => item.end)));
const selectedItem = computed(() => timelineItems.value.find((item) => item.scene === selectedScene.value) ?? timelineItems.value[0]);
const selectedClipUrl = computed(() => {
  const clip = rawPreview.value?.clip || selectedItem.value?.clip;
  return clip ? mediaUrl(projectId.value, clip) : "";
});
const audioUrl = computed(() => (audioSource.value ? mediaUrl(projectId.value, audioSource.value) : ""));
const playableItems = computed(() => timelineItems.value.filter((item) => item.clip));
const staleScenes = computed(() =>
  scenes.value
    .filter((scene) => Boolean((scene.edit as Record<string, unknown> | undefined)?.studio_stale))
    .map((scene) => Number(scene.scene))
);
const otherMedia = computed(() => [...allVideos.value, ...allImages.value].filter((path) => !isTimelineMedia(path)));
const timelineScaleStyle = computed(() => ({ width: `${timelineZoom.value * 100}%`, minWidth: "100%", "--timeline-zoom": String(timelineZoom.value) }));
const playheadStyle = computed(() => ({ left: `${((scrubSeconds.value || 0) / (totalDuration.value || 1)) * 100}%` }));
const timelineTicks = computed(() => {
  const total = totalDuration.value || 0;
  const step = total > 240 ? 30 : total > 90 ? 15 : 5;
  const ticks = [] as number[];
  for (let value = 0; value <= total; value += step) ticks.push(value);
  if (!ticks.includes(total)) ticks.push(total);
  return ticks;
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

function selectItem(item: TimelineItem) {
  rawPreview.value = null;
  selectedScene.value = item.scene;
  scrubSeconds.value = item.start;
  seekPreview();
}

async function scrub() {
  playingTimeline.value = false;
  const item = timelineItems.value.find((candidate) => scrubSeconds.value >= candidate.start && scrubSeconds.value <= candidate.end);
  if (item) selectedScene.value = item.scene;
  if (audioRef.value) audioRef.value.currentTime = scrubSeconds.value;
  await seekPreview();
}

async function seekPreview() {
  await nextTick();
  const video = videoRef.value;
  const item = selectedItem.value;
  if (!video || !item?.clip) return;
  video.currentTime = rawPreview.value ? rawPreview.value.seconds : Math.max(0, scrubSeconds.value - previewStart(item));
}

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

async function playTimeline() {
  const first = playableItems.value[0];
  if (!first) return;
  playingTimeline.value = true;
  selectItem(first);
  await nextTick();
  if (audioRef.value) {
    audioRef.value.currentTime = first.start;
    await audioRef.value.play();
  }
  await videoRef.value?.play();
}

function stopTimeline() {
  rawPreview.value = null;
  playingTimeline.value = false;
  videoRef.value?.pause();
  audioRef.value?.pause();
}

async function playNextClip() {
  if (!playingTimeline.value || !selectedItem.value) return;
  const index = playableItems.value.findIndex((item) => item.scene === selectedItem.value?.scene);
  const next = playableItems.value[index + 1];
  if (!next) {
    playingTimeline.value = false;
    audioRef.value?.pause();
    return;
  }
  selectItem(next);
  await nextTick();
  await videoRef.value?.play();
}

function syncScrubber() {
  if (!selectedItem.value || !videoRef.value) return;
  if (rawPreview.value) return;
  scrubSeconds.value = previewStart(selectedItem.value) + videoRef.value.currentTime;
  if (audioRef.value && Math.abs(audioRef.value.currentTime - scrubSeconds.value) > 0.25) audioRef.value.currentTime = scrubSeconds.value;
}

async function playAudio() {
  if (rawPreview.value) return;
  if (!audioRef.value) return;
  audioRef.value.currentTime = scrubSeconds.value;
  await audioRef.value.play();
}

function pauseAudio() {
  if (!playingTimeline.value) audioRef.value?.pause();
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

async function loadRenderManifest() {
  const manifestPath = studio.currentProject?.artifacts.generated_json.find((path) => path.endsWith("render_manifest.json"));
  if (!manifestPath) return;
  const data = (await api.artifact(projectId.value, manifestPath)).data;
  if (!Array.isArray(data)) return;
  renderManifest.value = Object.fromEntries(
    data
      .filter((entry): entry is RenderManifestEntry => Boolean(entry) && typeof entry === "object" && "scene" in entry)
      .map((entry) => [Number(entry.scene), entry])
  );
}

function fallbackRawTiming(scene: RenderScene, start: number, duration: number): { start: number; duration: number } {
  const fps = Number(scene.fps ?? 24) || 24;
  const edit = (scene.edit ?? {}) as Record<string, unknown>;
  const editedOut = Number(edit.raw_out_seconds ?? 0);
  const frameCount = Number(scene.frame_count ?? 0);
  const renderFrameCount = Number(readPath(scene, ["rolling", "render_frame_count"]) ?? readPath(scene, ["ltx", "render_frame_count"]) ?? 0);
  const trimFrontFrames = Number(readPath(scene, ["rolling", "trim_front_frames"]) ?? readPath(scene, ["ltx", "trim_front_frames"]) ?? 0);
  if (renderFrameCount > frameCount) {
    return { start: Math.max(0, start - trimFrontFrames / fps), duration: renderFrameCount / fps };
  }
  if (editedOut > duration) return { start, duration: editedOut };
  return { start: Math.max(0, start - Math.min(2, start)), duration: duration + Math.min(2, start) + 1 };
}

function previewStart(item: TimelineItem): number {
  return item.finalClip ? item.start : item.rawStart;
}

function startFinalDrag(event: PointerEvent, item: TimelineItem, mode: "left" | "right") {
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
  const previous = undoStack.value.pop();
  if (!previous) return;
  redoStack.value.push(scenes.value.map((scene) => cloneScene(scene)));
  scenes.value = previous.map((scene) => cloneScene(scene));
  timelineDirty.value = true;
}

function redoTimeline() {
  const next = redoStack.value.pop();
  if (!next) return;
  undoStack.value.push(scenes.value.map((scene) => cloneScene(scene)));
  scenes.value = next.map((scene) => cloneScene(scene));
  timelineDirty.value = true;
}

function pushUndo() {
  undoStack.value.push(scenes.value.map((scene) => cloneScene(scene)));
  if (undoStack.value.length > 30) undoStack.value.shift();
  redoStack.value = [];
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
  const total = totalDuration.value || 1;
  return {
    left: `${(start / total) * 100}%`,
    width: `${Math.max((duration / total) * 100, 0)}%`
  };
}

function thumbnailFrames(path: string, duration: number): { time: number; url: string }[] {
  const count = Math.max(1, Math.min(8, Math.floor((duration * timelineZoom.value) / 4) + 1));
  return Array.from({ length: count }, (_, index) => {
    const time = count === 1 ? Math.max(0, duration * 0.2) : Math.max(0, (duration * index) / (count - 1));
    return { time, url: thumbnailUrl(projectId.value, path, time) };
  });
}

function thumbnailRequests(): { path: string; times: number[] }[] {
  const requests = new Map<string, Set<number>>();
  for (const item of timelineItems.value) {
    for (const path of [item.rawClip, item.finalClip].filter(Boolean)) {
      const times = requests.get(path) ?? new Set<number>();
      thumbnailFrames(path, path === item.rawClip ? item.rawDuration : item.duration).forEach((frame) => times.add(frame.time));
      requests.set(path, times);
    }
  }
  return [...requests].map(([path, times]) => ({ path, times: [...times] }));
}

function sceneDurationScene(sceneNumber: number): RenderScene | undefined {
  return scenes.value.find((scene) => Number(scene.scene) === sceneNumber);
}

function sceneFps(scene: RenderScene): number {
  return Number(scene.fps ?? 24) || 24;
}

function editForScene(scene: RenderScene): ClipEdit {
  const sceneNumber = Number(scene.scene);
  const manifest = renderManifest.value[sceneNumber];
  const fps = sceneFps(scene);
  const sceneStart = Number(scene.abs_start_seconds ?? 0);
  const sceneDuration = Number(scene.duration_seconds ?? 0);
  const rawTiming = fallbackRawTiming(scene, sceneStart, sceneDuration);
  const frameCount = Number(manifest?.scene_frame_count ?? scene.frame_count ?? Math.round(fps * sceneDuration));
  const trimFrontFrames = Number(manifest?.trim_front_frames ?? Math.max(0, Math.round((sceneStart - rawTiming.start) * fps)));
  const fallbackRenderFrameCount = Math.max(trimFrontFrames + frameCount, Math.round(rawTiming.duration * fps));
  const renderFrameCount = Number(manifest?.render_frame_count ?? fallbackRenderFrameCount);
  const explicitTailFrames = readPath(scene, ["rolling", "tail_loss_frames"]) ?? readPath(scene, ["ltx", "tail_loss_frames"]);
  const tailFrames = Math.max(0, Number(manifest?.tail_loss_frames ?? explicitTailFrames ?? renderFrameCount - trimFrontFrames - frameCount));
  const base = buildEditState({
    scene: sceneNumber,
    frameCount,
    trimFrontFrames,
    tailFrames
  });
  const edit = (scene.edit ?? {}) as Record<string, unknown>;
  return {
    ...base,
    rawInFrame: Number(edit.raw_in_frame ?? base.rawInFrame),
    rawOutFrame: Number(edit.raw_out_frame ?? base.rawOutFrame)
  };
}

function sourceWindowStyle(item: TimelineItem): Record<string, string> {
  const edit = clipEdits.value.find((candidate) => candidate.scene === item.scene);
  const scene = sceneDurationScene(item.scene);
  if (!edit || !scene) return blockStyle(item.start, item.duration);
  const fps = sceneFps(scene);
  return blockStyle(item.start - edit.rawInFrame / fps, (edit.maxRawOutFrame - edit.minRawInFrame) / fps);
}

function updateRawPreview(sceneNumber: number, mode: "left" | "right", edits: ClipEdit[]) {
  const item = timelineItems.value.find((candidate) => candidate.scene === sceneNumber);
  const scene = sceneDurationScene(sceneNumber);
  const edit = edits.find((candidate) => candidate.scene === sceneNumber);
  if (!item?.rawClip || !scene || !edit) return;
  const seconds = (mode === "left" ? edit.rawInFrame : edit.rawOutFrame) / sceneFps(scene);
  rawPreview.value = { scene: sceneNumber, clip: item.rawClip, seconds, edge: mode === "left" ? "IN" : "OUT" };
  void seekPreview();
}

function applyClipEdits(edits: ClipEdit[]) {
  for (const edit of edits) {
    const scene = sceneDurationScene(edit.scene);
    if (!scene) continue;
    const fps = sceneFps(scene);
    scene.edit = {
      ...((scene.edit ?? {}) as Record<string, unknown>),
      raw_in_frame: edit.rawInFrame,
      raw_out_frame: edit.rawOutFrame,
      min_raw_in_frame: edit.minRawInFrame,
      max_raw_out_frame: edit.maxRawOutFrame,
      raw_in_seconds: edit.rawInFrame / fps,
      raw_out_seconds: edit.rawOutFrame / fps
    };
  }
}

function markChangedScenesStale(before: ClipEdit[], after: ClipEdit[]) {
  for (const edit of after) {
    const previous = before.find((candidate) => candidate.scene === edit.scene);
    if (!previous || (previous.rawInFrame === edit.rawInFrame && previous.rawOutFrame === edit.rawOutFrame)) continue;
    markScenesStale([edit.scene], "clip trim changed");
  }
}

function editSeconds(sceneNumber: number): { in: number; out: number } {
  const scene = sceneDurationScene(sceneNumber);
  const edit = clipEdits.value.find((candidate) => candidate.scene === sceneNumber);
  const fps = scene ? sceneFps(scene) : 24;
  return { in: Number(edit?.rawInFrame ?? 0) / fps, out: Number(edit?.rawOutFrame ?? 0) / fps };
}

function markScenesStale(sceneNumbers: number[], reason: string) {
  for (const scene of scenes.value) {
    if (!sceneNumbers.includes(Number(scene.scene))) continue;
    scene.edit = { ...((scene.edit ?? {}) as Record<string, unknown>), studio_stale: true, studio_stale_reason: reason };
  }
}

function clearSceneStale(sceneNumber: number) {
  const scene = sceneDurationScene(sceneNumber);
  if (!scene?.edit || typeof scene.edit !== "object") return;
  const edit = { ...(scene.edit as Record<string, unknown>) };
  delete edit.studio_stale;
  delete edit.studio_stale_reason;
  scene.edit = edit;
}

function isSceneStale(sceneNumber?: number): boolean {
  if (!sceneNumber) return false;
  return Boolean((sceneDurationScene(sceneNumber)?.edit as Record<string, unknown> | undefined)?.studio_stale);
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
        <button class="icon-button" :disabled="undoStack.length === 0" title="Undo" @click="undoTimeline"><Undo2 :size="18" /></button>
        <button class="icon-button" :disabled="redoStack.length === 0" title="Redo" @click="redoTimeline"><Redo2 :size="18" /></button>
        <button class="icon-button" title="Prebuild thumbnails" @click="prebuildThumbnails"><Sparkles :size="18" /></button>
        <button class="icon-button" title="Clear thumbnail cache" @click="cleanupThumbnails"><Trash2 :size="18" /></button>
        <label class="zoom-control" title="Timeline zoom">
          <ZoomIn :size="16" />
          <input v-model.number="timelineZoom" type="range" min="1" max="6" step="0.25" @input="drawWaveform" />
        </label>
        <button class="button secondary" :disabled="staleScenes.length === 0" @click="rebuildStaleScenes">
          <RotateCcw :size="18" /> Rebuild stale {{ staleScenes.length || "" }}
        </button>
        <button class="button secondary" :disabled="!timelineDirty" @click="pendingTimelineSave = true"><Save :size="18" /> Save timeline</button>
        <button class="button" :disabled="!selectedScene" @click="pendingRetake = true"><Clapperboard :size="18" /> Render retake</button>
      </div>
    </header>

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

      <div class="panel timeline-transport">
        <div class="button-row">
          <button class="button secondary" :disabled="playableItems.length === 0" @click="playTimeline"><Play :size="18" /> Play timeline</button>
          <button class="button secondary" :disabled="!playingTimeline" @click="stopTimeline"><Square :size="16" /> Stop</button>
        </div>
        <div class="timeline-scrubber">
          <span>{{ formatTime(scrubSeconds) }}</span>
          <input v-model.number="scrubSeconds" type="range" min="0" :max="totalDuration" step="0.01" @input="scrub" />
          <span>{{ formatTime(totalDuration) }}</span>
        </div>
      </div>

      <section class="panel timeline-panel">
        <div class="timeline-scroll">
          <div class="timeline-lanes" :style="timelineScaleStyle">
          <div class="timeline-lane time-ruler-lane">
            <span class="timeline-lane-label">Time</span>
            <div class="timeline-track time-ruler-track" title="Drag left/right to zoom" @pointerdown="startZoomDrag">
              <span class="timeline-playhead" :style="playheadStyle" />
              <span v-for="tick in timelineTicks" :key="tick" class="timeline-tick" :style="blockStyle(tick, 0.1)">{{ formatTime(tick) }}</span>
            </div>
          </div>
          <div class="timeline-lane">
            <span class="timeline-lane-label">Plan</span>
            <div class="timeline-track">
              <span class="timeline-playhead" :style="playheadStyle" />
              <button
                v-for="item in timelineItems"
                :key="`plan-${item.scene}`"
                class="timeline-clip planned"
                :class="{ active: item.scene === selectedScene, stale: isSceneStale(item.scene) }"
                :style="blockStyle(item.start, item.duration)"
                @click="selectItem(item)"
              >
                <strong>Scene {{ item.scene }}</strong>
                <small>{{ formatTime(item.start) }} - {{ formatTime(item.end) }}</small>
              </button>
            </div>
          </div>
          <div class="timeline-lane">
            <span class="timeline-lane-label">Final</span>
            <div class="timeline-track">
              <span class="timeline-playhead" :style="playheadStyle" />
              <span
                v-for="item in timelineItems"
                :key="`source-${item.scene}`"
                class="timeline-source-window"
                :class="{ active: item.scene === selectedScene }"
                :style="sourceWindowStyle(item)"
                :title="`Raw source bounds for scene ${item.scene}`"
              />
              <button
                v-for="item in timelineItems"
                :key="`final-${item.scene}`"
                class="timeline-clip final"
                :class="{ missing: !item.finalClip, active: item.scene === selectedScene, stale: isSceneStale(item.scene) }"
                :style="blockStyle(item.start, item.duration)"
                @click="selectItem(item)"
              >
                <span v-if="item.finalClip" class="clip-filmstrip" aria-hidden="true">
                  <img v-for="frame in thumbnailFrames(item.finalClip, item.duration)" :key="frame.time" class="clip-thumb" :src="frame.url" alt="" />
                </span>
                <span
                  v-if="item.scene === selectedScene"
                  class="timeline-edge-handle left"
                  title="Trim left: borrow frames from previous clip"
                  @click.stop
                  @pointerdown.stop="startFinalDrag($event, item, 'left')"
                >IN</span>
                <strong>Scene {{ item.scene }}</strong>
                <small>{{ item.finalClip ? "Concat clip" : item.rawClip ? "Recut to create" : "Gap" }}</small>
                <span
                  v-if="item.scene === selectedScene"
                  class="timeline-edge-handle right"
                  title="Trim right: borrow frames from next clip"
                  @click.stop
                  @pointerdown.stop="startFinalDrag($event, item, 'right')"
                >OUT</span>
              </button>
            </div>
          </div>
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
    <ConfirmDialog
      :open="pendingTimelineSave"
      title="Save timeline timing?"
      message="This rewrites scene start/end/duration/frame_count values in the active render plan. Existing rendered clips may no longer match until rerendered or recut."
      confirm-label="Save timeline"
      @cancel="pendingTimelineSave = false"
      @confirm="saveTimeline"
    />
  </section>
</template>
