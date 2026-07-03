<script setup lang="ts">
import ConfirmDialog from "./ConfirmDialog.vue";

defineProps<{
  pendingRetake: boolean;
  pendingRecut: boolean;
  pendingTimelineSave: boolean;
  selectedScene: number | null;
}>();

defineEmits<{
  cancelRetake: [];
  confirmRetake: [];
  cancelRecut: [];
  confirmRecut: [];
  cancelTimelineSave: [];
  confirmTimelineSave: [];
}>();
</script>

<template>
  <ConfirmDialog
    :open="pendingRetake"
    title="Render retake?"
    :message="`This will rerender scene ${selectedScene}. Existing generated scene clips may be overwritten and jobs cannot be cancelled yet.`"
    confirm-label="Render retake"
    @cancel="$emit('cancelRetake')"
    @confirm="$emit('confirmRetake')"
  />
  <ConfirmDialog
    :open="pendingRecut"
    title="Recut scene?"
    :message="`This will trim the raw clip for scene ${selectedScene} and overwrite the derived scene clip. The raw clip remains unchanged.`"
    confirm-label="Recut scene"
    @cancel="$emit('cancelRecut')"
    @confirm="$emit('confirmRecut')"
  />
  <ConfirmDialog
    :open="pendingTimelineSave"
    title="Save timeline timing?"
    message="This rewrites scene start/end/duration/frame_count values in the active render plan. Existing rendered clips may no longer match until rerendered or recut."
    confirm-label="Save timeline"
    @cancel="$emit('cancelTimelineSave')"
    @confirm="$emit('confirmTimelineSave')"
  />
</template>
