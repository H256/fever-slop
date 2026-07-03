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
const pendingAction = ref<PipelineAction | null>(null);
const streamedLogs = ref<string[]>([]);
const streamState = ref<"idle" | "connected" | "disconnected" | "complete">("idle");
const logPane = ref<HTMLElement | null>(null);
let pollTimer: number | undefined;
let source: EventSource | undefined;

type PipelinePhase = "core" | "preparation" | "storyboard" | "generation" | "post_processing";
type PipelineAction = {
  id: string;
  label: string;
  phase: PipelinePhase;
};

type ActionGroup = {
  phase: PipelinePhase;
  label: string;
  actions: PipelineAction[];
};

const phaseLabels: Record<PipelinePhase, string> = {
  core: "Core runs",
  preparation: "Preparation",
  storyboard: "Storyboard",
  generation: "Generation",
  post_processing: "Post-processing"
};

const phaseOrder: PipelinePhase[] = ["core", "preparation", "storyboard", "generation", "post_processing"];

const standardActions: PipelineAction[] = [
  { id: "full-pipeline", label: "Full pipeline", phase: "core" },
  { id: "main-pipeline", label: "Main pipeline", phase: "core" },
  { id: "relay-compact", label: "Relay compact", phase: "preparation" },
  { id: "anchor-fix", label: "Anchor fix", phase: "preparation" },
  { id: "rebuild-plan", label: "Rebuild plan", phase: "preparation" },
  { id: "storyboard", label: "Storyboard", phase: "storyboard" },
  { id: "storyboard-frames", label: "Storyboard frames", phase: "storyboard" },
  { id: "storyboard-page", label: "Storyboard page", phase: "storyboard" },
  { id: "msr-references", label: "MSR references", phase: "generation" },
  { id: "msr-reference-sheets", label: "MSR reference sheets", phase: "generation" },
  { id: "msr-enrich", label: "MSR enrichment", phase: "generation" },
  { id: "msr-prompt-enrich", label: "MSR prompt enrichment", phase: "generation" },
  { id: "ltx-render-scenes", label: "Render selected scenes", phase: "generation" },
  { id: "final-concat", label: "Final concat", phase: "post_processing" },
  { id: "concat-video-only", label: "Concat video only", phase: "post_processing" },
  { id: "mux-original-audio", label: "Mux original audio", phase: "post_processing" }
];

const fullAutoActions: PipelineAction[] = [{ id: "full-auto", label: "Full-auto pipeline", phase: "core" }];
const movieActions: PipelineAction[] = [
  { id: "movie-references", label: "Movie references", phase: "preparation" },
  { id: "movie-full-auto", label: "Movie full-auto production", phase: "core" }
];

const actions = computed(() => {
  if (studio.currentProject?.project_type === "full_auto") return fullAutoActions;
  if (studio.currentProject?.project_type === "movie") return movieActions;
  return standardActions;
});
const actionGroups = computed<ActionGroup[]>(() =>
  phaseOrder
    .map((phase) => ({
      phase,
      label: phaseLabels[phase],
      actions: actions.value.filter((action) => action.phase === phase)
    }))
    .filter((group) => group.actions.length > 0)
);
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

function askToRun(action: PipelineAction) {
  pendingAction.value = action;
}

async function runConfirmed() {
  if (!pendingAction.value) return;
  const action = pendingAction.value.id;
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
        <div class="phase-list">
          <section v-for="group in actionGroups" :key="group.phase" class="pipeline-phase">
            <h2 class="phase-header">{{ group.label }}</h2>
            <div class="phase-actions">
              <button
                v-for="action in group.actions"
                :key="action.id"
                class="action-button"
                :disabled="hasRunningPipeline"
                :title="hasRunningPipeline ? 'Pipeline is already running' : ''"
                @click="askToRun(action)"
              >
                <Play :size="18" />
                <span>{{ action.label }}</span>
              </button>
            </div>
          </section>
        </div>
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
      :message="`This will start '${pendingAction?.label}' for ${projectId}. It may overwrite generated project artifacts, can trigger external tools, and cannot be cancelled from Studio yet.`"
      confirm-label="Run job"
      @cancel="pendingAction = null"
      @confirm="runConfirmed"
    />
  </section>
</template>
