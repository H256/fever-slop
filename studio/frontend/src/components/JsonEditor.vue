<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{ modelValue: unknown }>();
const emit = defineEmits<{ "update:modelValue": [value: unknown] }>();
const text = ref("");
const error = ref("");

watch(
  () => props.modelValue,
  (value) => {
    text.value = JSON.stringify(value, null, 2);
    error.value = "";
  },
  { immediate: true }
);

const valid = computed(() => !error.value);

function apply() {
  try {
    emit("update:modelValue", JSON.parse(text.value));
    error.value = "";
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Invalid JSON";
  }
}
</script>

<template>
  <div class="json-editor">
    <textarea v-model="text" spellcheck="false" @blur="apply" />
    <footer>
      <button class="button secondary" @click="apply">Apply JSON</button>
      <span v-if="!valid" class="error-text">{{ error }}</span>
    </footer>
  </div>
</template>
