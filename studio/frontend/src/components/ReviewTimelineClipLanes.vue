<script setup lang="ts">
import type { TimelineItem } from "../composables/reviewTimeline";

defineProps<{
  blockStyle: (start: number, duration: number) => Record<string, string>;
  formatTime: (value: number) => string;
  isSceneStale: (sceneNumber?: number) => boolean;
  playheadStyle: Record<string, string>;
  selectedScene: number | null;
  sourceWindowStyle: (item: TimelineItem) => Record<string, string>;
  thumbnailFrames: (path: string, duration: number) => { time: number; url: string }[];
  timelineItems: TimelineItem[];
  timelineTicks: number[];
}>();

defineEmits<{
  selectItem: [item: TimelineItem];
  startFinalDrag: [event: PointerEvent, item: TimelineItem, mode: "left" | "right"];
  startZoomDrag: [event: PointerEvent];
}>();
</script>

<template>
  <div class="timeline-lane time-ruler-lane">
    <span class="timeline-lane-label">Time</span>
    <div class="timeline-track time-ruler-track" title="Drag left/right to zoom" @pointerdown="$emit('startZoomDrag', $event)">
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
        @click="$emit('selectItem', item)"
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
        @click="$emit('selectItem', item)"
      >
        <span v-if="item.finalClip" class="clip-filmstrip" aria-hidden="true">
          <img v-for="frame in thumbnailFrames(item.finalClip, item.duration)" :key="frame.time" class="clip-thumb" :src="frame.url" alt="" />
        </span>
        <span
          v-if="item.scene === selectedScene"
          class="timeline-edge-handle left"
          title="Trim left: borrow frames from previous clip"
          @click.stop
          @pointerdown.stop="$emit('startFinalDrag', $event, item, 'left')"
        >IN</span>
        <strong>Scene {{ item.scene }}</strong>
        <small>{{ item.finalClip ? "Concat clip" : item.rawClip ? "Recut to create" : "Gap" }}</small>
        <span
          v-if="item.scene === selectedScene"
          class="timeline-edge-handle right"
          title="Trim right: borrow frames from next clip"
          @click.stop
          @pointerdown.stop="$emit('startFinalDrag', $event, item, 'right')"
        >OUT</span>
      </button>
    </div>
  </div>
</template>
