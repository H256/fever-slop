<script setup lang="ts">
import type { Job } from "../types";
import StatusBadge from "./StatusBadge.vue";

defineProps<{ jobs: Job[] }>();
</script>

<template>
  <section class="panel job-panel">
    <header class="panel-header">
      <h2>Jobs</h2>
    </header>
    <p class="job-note">Status refreshes from Studio job state. Rich CLI progress is still printed in the backend terminal.</p>
    <div v-if="jobs.length === 0" class="empty">No jobs yet.</div>
    <article v-for="job in jobs" :key="job.id" class="job-row">
      <div>
        <strong>{{ job.action }}</strong>
        <p>{{ job.id }}</p>
      </div>
      <StatusBadge :status="job.status" />
      <pre v-if="job.logs.length || job.error">{{ [...job.logs, job.error].filter(Boolean).join('\n') }}</pre>
    </article>
  </section>
</template>
