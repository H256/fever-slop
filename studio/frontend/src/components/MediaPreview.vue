<script setup lang="ts">
import { computed } from "vue";
import { mediaUrl } from "../api";

const props = defineProps<{ projectId: string; path: string }>();
const url = computed(() => mediaUrl(props.projectId, props.path));
const isVideo = computed(() => /\.(mp4|mov|webm)$/i.test(props.path));
</script>

<template>
  <figure class="media-preview">
    <video v-if="isVideo" :src="url" controls />
    <img v-else :src="url" :alt="path" />
    <figcaption>{{ path }}</figcaption>
  </figure>
</template>
