<script setup lang="ts">
import { Clapperboard, Redo2, RotateCcw, Save, Sparkles, Trash2, Undo2, ZoomIn } from "lucide-vue-next";

defineProps<{
  redoCount: number;
  selectedScene: number | null;
  staleCount: number;
  timelineDirty: boolean;
  timelineZoom: number;
  undoCount: number;
}>();

defineEmits<{
  cleanupThumbnails: [];
  prebuildThumbnails: [];
  rebuildStaleScenes: [];
  renderRetake: [];
  saveTimeline: [];
  undoTimeline: [];
  redoTimeline: [];
  updateTimelineZoom: [value: number];
  zoomInput: [];
}>();
</script>

<template>
  <header class="page-header toolbar-header">
    <div>
      <h1>Review</h1>
      <p>Timeline preview from the active render plan and available raw/final scene clips.</p>
    </div>
    <div class="button-row">
      <button class="icon-button" :disabled="undoCount === 0" title="Undo" @click="$emit('undoTimeline')"><Undo2 :size="18" /></button>
      <button class="icon-button" :disabled="redoCount === 0" title="Redo" @click="$emit('redoTimeline')"><Redo2 :size="18" /></button>
      <button class="icon-button" title="Prebuild thumbnails" @click="$emit('prebuildThumbnails')"><Sparkles :size="18" /></button>
      <button class="icon-button" title="Clear thumbnail cache" @click="$emit('cleanupThumbnails')"><Trash2 :size="18" /></button>
      <label class="zoom-control" title="Timeline zoom">
        <ZoomIn :size="16" />
        <input
          :value="timelineZoom"
          type="range"
          min="1"
          max="6"
          step="0.25"
          @input="$emit('updateTimelineZoom', Number(($event.target as HTMLInputElement).value)); $emit('zoomInput')"
        />
      </label>
      <button class="button secondary" :disabled="staleCount === 0" @click="$emit('rebuildStaleScenes')">
        <RotateCcw :size="18" /> Rebuild stale {{ staleCount || "" }}
      </button>
      <button class="button secondary" :disabled="!timelineDirty" @click="$emit('saveTimeline')"><Save :size="18" /> Save timeline</button>
      <button class="button" :disabled="!selectedScene" @click="$emit('renderRetake')"><Clapperboard :size="18" /> Render retake</button>
    </div>
  </header>
</template>
