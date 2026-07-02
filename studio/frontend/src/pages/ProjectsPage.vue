<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { Folder, Music, Plus, WandSparkles, X } from "lucide-vue-next";
import { api } from "../api";
import { useStudioStore } from "../stores/studio";
import type { ProjectCreatePayload } from "../types";
import type { ProjectSummary } from "../types";

const studio = useStudioStore();
const router = useRouter();
onMounted(() => studio.loadProjects());
type ProjectKind = "standard_music_video" | "full_auto";

const creating = ref(false);
const selectedKind = ref<ProjectKind | "">("");
const submitting = ref(false);
const error = ref("");
const form = reactive({
  name: "",
  idea: "",
  songStyle: "",
  durationSeconds: 120,
  width: 1280,
  height: 704,
  fps: 24 as 16 | 24 | 50,
  pipelineMode: "msr" as "classic" | "msr"
});

const badgeLabels: Record<string, string> = {
  config: "Config",
  render_plan: "Render plan",
  references: "References"
};

const slug = computed(() => slugifyProjectName(form.name));
const slugConflict = computed(() => Boolean(slug.value && studio.projects.some((project) => project.id === slug.value)));
const validationError = computed(() => {
  if (!selectedKind.value) return "";
  if (!form.name.trim()) return "Project name is required.";
  if (!slug.value) return "Project name must contain at least one letter or number.";
  if (slugConflict.value) return `A project folder named "${slug.value}" already exists.`;
  if (selectedKind.value === "full_auto" && !form.idea.trim()) return "Idea is required for full-auto projects.";
  if (selectedKind.value === "full_auto" && !form.songStyle.trim()) return "Song style is required for full-auto projects.";
  if (selectedKind.value === "full_auto" && (!Number.isFinite(form.durationSeconds) || form.durationSeconds <= 0)) return "Duration must be a positive number.";
  if (selectedKind.value === "full_auto" && (!Number.isInteger(form.width) || form.width <= 0)) return "Width must be a positive integer.";
  if (selectedKind.value === "full_auto" && (!Number.isInteger(form.height) || form.height <= 0)) return "Height must be a positive integer.";
  if (selectedKind.value === "full_auto" && ![16, 24, 50].includes(form.fps)) return "FPS must be 16, 24, or 50.";
  return "";
});

function presentBadges(project: ProjectSummary): string[] {
  return Object.entries(project.status)
    .filter(([, status]) => status === "present")
    .map(([name]) => name);
}

function slugifyProjectName(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").replace(/-+/g, "-");
}

function startCreate(kind?: ProjectKind) {
  creating.value = true;
  selectedKind.value = kind ?? "";
  error.value = "";
}

function cancelCreate() {
  creating.value = false;
  selectedKind.value = "";
  error.value = "";
  form.name = "";
  form.idea = "";
  form.songStyle = "";
  form.durationSeconds = 120;
  form.width = 1280;
  form.height = 704;
  form.fps = 24;
  form.pipelineMode = "msr";
}

async function createProject(startFullAuto = false) {
  error.value = validationError.value;
  if (error.value || !selectedKind.value) return;
  submitting.value = true;
  try {
    const payload: ProjectCreatePayload = {
      project_type: selectedKind.value,
      name: form.name.trim()
    };
    if (selectedKind.value === "full_auto") {
      payload.idea = form.idea.trim();
      payload.song_style = form.songStyle.trim();
      payload.duration_seconds = form.durationSeconds;
      payload.width = form.width;
      payload.height = form.height;
      payload.fps = form.fps;
      payload.pipeline_mode = form.pipelineMode;
    }
    const project = await api.createProject(payload);
    await studio.loadProjects();
    if (startFullAuto) await studio.startJob(project.id, "full-auto");
    cancelCreate();
    if (project.project_type === "full_auto") {
      await router.push(`/projects/${project.id}/pipeline`);
    } else {
      await router.push(`/projects/${project.id}/settings`);
    }
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught);
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <section class="page">
    <header class="page-header toolbar-header">
      <div>
        <h1>Projects</h1>
        <p>Local FeverSlop project folders under <code>projects/</code>.</p>
      </div>
      <button class="button" @click="startCreate()"><Plus :size="18" /> Create Project</button>
    </header>
    <section v-if="creating" class="panel create-project-panel">
      <header class="panel-header">
        <h2>Create Project</h2>
        <button class="icon-button" title="Cancel" @click="cancelCreate"><X :size="18" /></button>
      </header>
      <div v-if="!selectedKind" class="project-type-grid">
        <button class="project-type-card" @click="startCreate('standard_music_video')">
          <Music :size="22" />
          <strong>Standard - Music Video Project</strong>
          <span>Creates a normal project folder and opens the standard configuration form before you run generation.</span>
        </button>
        <button class="project-type-card" @click="startCreate('full_auto')">
          <WandSparkles :size="22" />
          <strong>Full-Auto Project</strong>
          <span>Creates the song and project automatically from a short brief, with render size and pipeline mode chosen up front.</span>
        </button>
      </div>
      <form v-else class="create-project-form" @submit.prevent="createProject(selectedKind === 'full_auto')">
        <label>
          <span>Project name</span>
          <input v-model="form.name" type="text" autocomplete="off" />
        </label>
        <p class="project-folder slug-preview"><Folder :size="15" /> {{ slug || "project-folder" }}</p>
        <template v-if="selectedKind === 'full_auto'">
          <label>
            <span>Idea</span>
            <textarea v-model="form.idea" class="transcript-area" />
          </label>
          <label>
            <span>Song style</span>
            <input v-model="form.songStyle" type="text" />
          </label>
          <div class="form-grid-compact">
            <label>
              <span>Desired video duration</span>
              <input v-model.number="form.durationSeconds" type="number" min="1" step="0.1" />
            </label>
            <label>
              <span>Width</span>
              <input v-model.number="form.width" type="number" min="1" step="1" />
            </label>
            <label>
              <span>Height</span>
              <input v-model.number="form.height" type="number" min="1" step="1" />
            </label>
            <label>
              <span>FPS</span>
              <select v-model.number="form.fps">
                <option :value="16">16</option>
                <option :value="24">24</option>
                <option :value="50">50</option>
              </select>
            </label>
          </div>
          <label>
            <span>Pipeline mode</span>
            <select v-model="form.pipelineMode">
              <option value="msr">MSR - uses Scene Bible, Actor Bible, and reference-guided generation</option>
              <option value="classic">Classic - skips Scene/Actor Bible and uses the original image-to-video path</option>
            </select>
          </label>
        </template>
        <p v-if="validationError || error" class="form-error">{{ validationError || error }}</p>
        <div class="button-row">
          <button type="button" class="button secondary" @click="cancelCreate">Cancel</button>
          <button
            v-if="selectedKind === 'standard_music_video'"
            type="submit"
            class="button"
            :disabled="Boolean(validationError) || submitting"
          >
            Create and configure
          </button>
          <button v-else type="submit" class="button" :disabled="Boolean(validationError) || submitting">Start full-auto pipeline</button>
        </div>
      </form>
    </section>
    <div class="project-grid">
      <RouterLink v-for="project in studio.projects" :key="project.id" class="project-card" :to="`/projects/${project.id}`">
        <h2>{{ project.name }}</h2>
        <p class="project-folder"><Folder :size="15" /> {{ project.id }}</p>
        <div class="badge-row">
          <span
            v-for="name in presentBadges(project)"
            :key="name"
            class="status-badge project-artifact-badge"
            :class="`artifact-${name}`"
          >
            {{ badgeLabels[name] ?? name }}
          </span>
        </div>
      </RouterLink>
    </div>
  </section>
</template>
