<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import { Play, RotateCcw } from "lucide-vue-next";
import { useRoute } from "vue-router";
import { jobLogsUrl } from "../api";
import { useStudioStore } from "../stores/studio";
import type { Job } from "../types";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import JobDrawer from "../components/JobDrawer.vue";
import StatusBadge from "../components/StatusBadge.vue";

const route = useRoute();
const studio = useStudioStore();
const projectId = computed(() => String(route.params.projectId));
const pendingAction = ref<Action | null>(null);
const streamedLogs = ref<string[]>([]);
const streamState = ref<"idle" | "connected" | "disconnected" | "complete">("idle");
const logPane = ref<HTMLElement | null>(null);
let pollTimer: number | undefined;
let source: EventSource | undefined;

type Action = readonly [string, string];

const standardActions = [
  ["main-pipeline", "Main pipeline"],
  ["relay-compact", "Relay compact"],
  ["anchor-fix", "Anchor fix"],
  ["msr-references", "MSR references"],
  ["msr-reference-sheets", "MSR reference sheets"],
  ["msr-prompt-enrich", "MSR prompt enrichment"],
  ["storyboard-frames", "Storyboard frames"],
  ["storyboard-page", "Storyboard page"],
  ["ltx-render-scenes", "Render selected scenes"],
  ["concat-video-only", "Concat video only"],
  ["mux-original-audio", "Mux original audio"],
  ["full-pipeline", "Full pipeline"]
] as const;

const fullAutoActions = [["full-auto", "Full-auto pipeline"]] as const;

const actions = computed(() => (studio.currentProject?.project_type === "full_auto" ? fullAutoActions : standardActions));
const activeJob = computed(() => studio.jobs.find((job) => ["running", "queued"].includes(job.status)) ?? studio.jobs[0] ?? null);
const hasRunningPipeline = computed(() => studio.jobs.some((job) => ["running", "queued"].includes(job.status)));
const logs = computed(() => (streamedLogs.value.length ? streamedLogs.value : activeJob.value?.recent_logs ?? activeJob.value?.logs ?? []));
const overallProgress = computed(() => activeJob.value?.overall_progress ?? activeJob.value?.progress ?? 0);

onMounted(async () => {
  await studio.loadProject(projectId.value);
  await refreshJobs();
  pollTimer = window.setInterval(refreshJobs, 2000);
});

onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer);
  closeStream();
});

watch(
  () => activeJob.value?.id,
  () => connectLogs(),
  { immediate: true }
);

async function refreshJobs() {
  await studio.loadJobs(projectId.value);
}

function askToRun(action: Action) {
  pendingAction.value = action;
}

async function runConfirmed() {
  if (!pendingAction.value) return;
  const [action] = pendingAction.value;
  pendingAction.value = null;
  try {
    await studio.startJob(projectId.value, action);
    await refreshJobs();
  } catch (caught) {
    streamedLogs.value = [caught instanceof Error ? caught.message : String(caught)];
  }
}

function connectLogs() {
  closeStream();
  streamedLogs.value = [];
  const job = activeJob.value;
  if (!job) return;
  if (!["running", "queued"].includes(job.status)) {
    streamState.value = "complete";
    return;
  }
  streamState.value = "idle";
  source = new EventSource(jobLogsUrl(job.id));
  source.onopen = () => {
    streamState.value = "connected";
  };
  source.onmessage = (event) => {
    const wasAtBottom = isAtLogBottom();
    const payload = JSON.parse(event.data) as { line?: string };
    if (payload.line) streamedLogs.value = [...streamedLogs.value, payload.line].slice(-500);
    if (wasAtBottom) void nextTick(scrollLogsToBottom);
  };
  source.addEventListener("status", () => {
    streamState.value = "complete";
    closeStream(false);
    void refreshJobs();
  });
  source.onerror = () => {
    streamState.value = "disconnected";
    closeStream(false);
  };
}

function reconnectLogs() {
  connectLogs();
}

function closeStream(reset = true) {
  source?.close();
  source = undefined;
  if (reset) streamState.value = "idle";
}

function isAtLogBottom(): boolean {
  const node = logPane.value;
  if (!node) return true;
  return node.scrollHeight - node.scrollTop - node.clientHeight < 24;
}

function scrollLogsToBottom() {
  const node = logPane.value;
  if (node) node.scrollTop = node.scrollHeight;
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null) return "unknown";
  const total = Math.max(0, Math.round(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return minutes ? `${minutes}m ${seconds.toString().padStart(2, "0")}s` : `${seconds}s`;
}

function stepProgress(job: Job, stepProgressValue: number | null): number {
  return stepProgressValue ?? (job.status === "succeeded" ? 100 : 0);
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <h1>Pipeline</h1>
      <p>Run existing FeverSlop pipeline stages as background jobs. Jobs can overwrite generated artifacts and cannot be cancelled yet.</p>
    </header>
    <div class="split pipeline-layout">
      <section class="panel action-list">
        <p v-if="hasRunningPipeline" class="job-note">Pipeline is already running. Start buttons are disabled until it finishes.</p>
        <button
          v-for="action in actions"
          :key="action[0]"
          class="action-button"
          :disabled="hasRunningPipeline"
          :title="hasRunningPipeline ? 'Pipeline is already running' : ''"
          @click="askToRun(action)"
        >
          <Play :size="18" />
          <span>{{ action[1] }}</span>
        </button>
      </section>
      <section class="panel pipeline-monitor">
        <header class="panel-header">
          <div>
            <h2>{{ activeJob ? activeJob.pipeline_type ?? activeJob.action : "No running job" }}</h2>
            <p v-if="activeJob" class="job-note">
              Elapsed {{ formatSeconds(activeJob.elapsed_seconds) }} · ETA {{ formatSeconds(activeJob.eta_seconds) }} · Current
              {{ activeJob.current_step ?? "none" }}
            </p>
          </div>
          <StatusBadge v-if="activeJob" :status="activeJob.status" />
        </header>
        <div v-if="!activeJob" class="empty">Start a pipeline job to monitor progress here.</div>
        <template v-else>
          <div class="progress-row">
            <span>Overall</span>
            <progress :value="overallProgress" max="100" />
            <strong>{{ overallProgress }}%</strong>
          </div>
          <div class="step-list">
            <article v-for="step in activeJob.steps ?? []" :key="step.name" class="step-row" :class="step.status">
              <div>
                <strong>{{ step.name }}</strong>
                <small>{{ formatSeconds(step.elapsed_seconds) }}</small>
              </div>
              <StatusBadge :status="step.status" />
              <progress
                v-if="step.progress !== null"
                :value="stepProgress(activeJob, step.progress)"
                max="100"
              />
              <div v-else class="indeterminate-progress" :class="{ running: step.status === 'running' }" />
            </article>
          </div>
          <section v-if="activeJob.error" class="pipeline-error">
            <strong>Job failed</strong>
            <p>{{ activeJob.error }}</p>
          </section>
          <section class="job-log-section">
            <header class="panel-header">
              <h3>Recent output</h3>
              <button v-if="streamState === 'disconnected'" class="button secondary" @click="reconnectLogs">
                <RotateCcw :size="18" /> Reconnect
              </button>
              <span v-else class="job-note">Log stream: {{ streamState }}</span>
            </header>
            <pre ref="logPane" class="live-log"><code>{{ logs.join("\n") || "Waiting for pipeline output..." }}</code></pre>
          </section>
        </template>
      </section>
      <JobDrawer :jobs="studio.jobs" />
    </div>
    <ConfirmDialog
      :open="Boolean(pendingAction)"
      title="Run pipeline job?"
      :message="`This will start '${pendingAction?.[1]}' for ${projectId}. It may overwrite generated project artifacts, can trigger external tools, and cannot be cancelled from Studio yet.`"
      confirm-label="Run job"
      @cancel="pendingAction = null"
      @confirm="runConfirmed"
    />
  </section>
</template>
