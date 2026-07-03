<script setup lang="ts">
defineProps<{
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
}>();

const emit = defineEmits<{ confirm: []; cancel: [] }>();
</script>

<template>
  <div v-if="open" class="modal-backdrop" role="presentation" @click.self="emit('cancel')">
    <section class="confirm-dialog" role="dialog" aria-modal="true" :aria-label="title">
      <h2>{{ title }}</h2>
      <p>{{ message }}</p>
      <div class="button-row">
        <button class="button secondary" @click="emit('cancel')">Cancel</button>
        <button class="button danger" @click="emit('confirm')">{{ confirmLabel ?? "Run job" }}</button>
      </div>
    </section>
  </div>
</template>
