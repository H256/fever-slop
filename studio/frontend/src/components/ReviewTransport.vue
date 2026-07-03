<script setup lang="ts">
import { Play, Square } from "lucide-vue-next";
import { formatTime } from "../composables/reviewTimelinePresentation";

defineProps<{
  playableCount: number;
  playingTimeline: boolean;
  scrubSeconds: number;
  totalDuration: number;
}>();

defineEmits<{
  playTimeline: [];
  stopTimeline: [];
  scrub: [];
  updateScrubSeconds: [value: number];
}>();
</script>

<template>
  <div class="panel timeline-transport">
    <div class="button-row">
      <button class="button secondary" :disabled="playableCount === 0" @click="$emit('playTimeline')"><Play :size="18" /> Play timeline</button>
      <button class="button secondary" :disabled="!playingTimeline" @click="$emit('stopTimeline')"><Square :size="16" /> Stop</button>
    </div>
    <div class="timeline-scrubber">
      <span>{{ formatTime(scrubSeconds) }}</span>
      <input
        :value="scrubSeconds"
        type="range"
        min="0"
        :max="totalDuration"
        step="0.01"
        @input="$emit('updateScrubSeconds', Number(($event.target as HTMLInputElement).value)); $emit('scrub')"
      />
      <span>{{ formatTime(totalDuration) }}</span>
    </div>
  </div>
</template>
