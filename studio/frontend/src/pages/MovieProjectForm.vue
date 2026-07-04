<script setup lang="ts">
import { computed, nextTick, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { ArrowLeft, Clapperboard, Folder, WandSparkles } from "lucide-vue-next";
import { useStudioStore } from "../stores/studio";
import type { ProjectCreatePayload } from "../types";

const router = useRouter();
const studio = useStudioStore();

const submitting = ref(false);
const submitStatus = ref("");
const error = ref("");
const form = reactive({
  name: "",
  sourceType: "short_story" as "short_story" | "screenplay",
  storyText: "",
  desiredLength: 180,
  width: 1280,
  height: 704,
  mode: "scaffold" as "scaffold" | "full_auto",
  plannerBackend: "llm" as "llm" | "deterministic",
  referenceBackend: "comfyui" as "comfyui" | "local",
  renderBackend: "comfyui" as "comfyui" | "local",
  heroWorkflow: "workflows/image_t2i_startframe_krea_v1.json",
  editWorkflow: "workflows/image_edit_flux2_klein_1ref_v1.json",
  msrWorkflow: "workflows/video_default_ltxv_msr_1actor_1background_v1.json"
});

const slug = computed(() => slugifyProjectName(form.name));
const validationError = computed(() => {
  if (!form.name.trim()) return "Project name is required.";
  if (!slug.value) return "Project name must contain at least one letter or number.";
  if (studio.projects.some((project) => project.id === slug.value)) return `A project folder named "${slug.value}" already exists.`;
  if (form.storyText.trim().length < 20) return "Story or screenplay input must be at least 20 characters.";
  if (form.sourceType === "screenplay" && !looksLikeScreenplay(form.storyText)) return "Screenplay input must contain scene headings such as INT. or EXT.";
  if (!Number.isFinite(form.desiredLength) || form.desiredLength <= 0) return "Desired length must be a positive number.";
  if (!Number.isInteger(form.width) || form.width <= 0) return "Width must be a positive integer.";
  if (!Number.isInteger(form.height) || form.height <= 0) return "Height must be a positive integer.";
  if (!form.heroWorkflow.trim() || !form.editWorkflow.trim() || !form.msrWorkflow.trim()) return "Movie workflow paths are required.";
  return "";
});

function slugifyProjectName(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").replace(/-+/g, "-");
}

function looksLikeScreenplay(value: string): boolean {
  const upper = value.toUpperCase();
  return upper.includes("INT.") || upper.includes("EXT.");
}

async function createMovieProject() {
  error.value = validationError.value;
  if (error.value) return;
  submitting.value = true;
  submitStatus.value = form.mode === "full_auto" ? "Creating movie scaffold..." : "Creating movie project...";
  await nextTick();
  try {
    const payload: ProjectCreatePayload = {
      project_type: "movie",
      name: form.name.trim(),
      source_type: form.sourceType,
      story_text: form.storyText.trim(),
      desired_length: form.desiredLength,
      width: form.width,
      height: form.height,
      movie_mode: form.mode,
      movie_planner_backend: form.plannerBackend,
      movie_reference_backend: form.referenceBackend,
      movie_render_backend: form.renderBackend,
      movie_hero_workflow: form.heroWorkflow.trim(),
      movie_edit_workflow: form.editWorkflow.trim(),
      movie_msr_workflow: form.msrWorkflow.trim()
    };
    const project = await studio.createProject(payload);
    if (form.mode === "full_auto") {
      submitStatus.value = "Starting movie production job...";
      await studio.startJob(project.id, "movie-full-auto");
      await router.push(`/projects/${project.id}/pipeline`);
    } else {
      await router.push(`/projects/${project.id}/render-plan`);
    }
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught);
  } finally {
    submitting.value = false;
    submitStatus.value = "";
  }
}
</script>

<template>
  <section class="page movie-create-page">
    <header class="page-header toolbar-header">
      <div>
        <button class="inline-link" type="button" @click="router.push('/')"><ArrowLeft :size="16" /> Projects</button>
        <h1>New Movie Project</h1>
        <p>Create a cinematic story project from prose or screenplay text.</p>
      </div>
    </header>

    <form class="panel movie-project-form" :aria-busy="submitting" @submit.prevent="createMovieProject">
      <section v-if="submitting" class="busy-banner" role="status" aria-live="polite">
        <span class="spinner" aria-hidden="true"></span>
        <div>
          <strong>{{ submitStatus }}</strong>
          <p>Generating story structure and project files. Keep this page open.</p>
        </div>
      </section>
      <section class="settings-section">
        <header class="section-header">
          <Clapperboard :size="20" />
          <div>
            <h2>Story Source</h2>
            <p>Scaffold creates story and render-plan artifacts. Full-auto also starts movie production.</p>
          </div>
        </header>
        <label>
          <span>Project name</span>
          <input v-model="form.name" type="text" autocomplete="off" />
        </label>
        <p class="project-folder slug-preview"><Folder :size="15" /> {{ slug || "project-folder" }}</p>
        <label>
          <span>Input format</span>
          <select v-model="form.sourceType">
            <option value="short_story">Short Story Idea</option>
            <option value="screenplay">Screenplay Format</option>
          </select>
        </label>
        <label>
          <span>{{ form.sourceType === "screenplay" ? "Screenplay" : "Short story idea" }}</span>
          <textarea
            v-model="form.storyText"
            class="transcript-area movie-source-text"
            placeholder="A locked observatory opens only during storms..."
          />
        </label>
      </section>

      <section class="settings-section">
        <header class="section-header">
          <WandSparkles :size="20" />
          <div>
            <h2>Movie Output</h2>
            <p>LTX MSR is used without a supplied audio track so LTX 2.3 can generate native synchronized audio.</p>
          </div>
        </header>
        <div class="form-grid-compact">
          <label>
            <span>Desired length</span>
            <input v-model.number="form.desiredLength" type="number" min="1" step="0.1" />
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
            <span>Mode</span>
            <select v-model="form.mode">
              <option value="scaffold">Scaffold - story arch and render plan only</option>
              <option value="full_auto">Full-Auto - start movie production after scaffold</option>
            </select>
          </label>
        </div>
      </section>

      <details class="settings-section">
        <summary class="section-header">
          <WandSparkles :size="20" />
          <div>
            <h2>Execution</h2>
            <p>Defaults use the configured LLM and ComfyUI workflows. Local is only for fast dev tests.</p>
          </div>
        </summary>
        <div class="form-grid-compact">
          <label>
            <span>Story planning</span>
            <select v-model="form.plannerBackend">
              <option value="llm">LLM</option>
              <option value="deterministic">Deterministic dev fallback</option>
            </select>
          </label>
          <label>
            <span>References</span>
            <select v-model="form.referenceBackend">
              <option value="comfyui">ComfyUI / Krea workflow</option>
              <option value="local">Local dev placeholder</option>
            </select>
          </label>
          <label>
            <span>Movie render</span>
            <select v-model="form.renderBackend">
              <option value="comfyui">ComfyUI / LTX MSR</option>
              <option value="local">Local dev placeholder</option>
            </select>
          </label>
        </div>
        <div class="settings-stack">
          <label>
            <span>Hero reference workflow</span>
            <input v-model="form.heroWorkflow" type="text" />
          </label>
          <label>
            <span>Edit reference workflow</span>
            <input v-model="form.editWorkflow" type="text" />
          </label>
          <label>
            <span>Movie MSR workflow</span>
            <input v-model="form.msrWorkflow" type="text" />
          </label>
        </div>
      </details>

      <p v-if="validationError || error" class="form-error">{{ validationError || error }}</p>
      <div class="button-row">
        <button type="button" class="button secondary" :disabled="submitting" @click="router.push('/')">Cancel</button>
        <button type="submit" class="button" :disabled="Boolean(validationError) || submitting">
          <span v-if="submitting" class="spinner small" aria-hidden="true"></span>
          {{ submitting ? submitStatus : form.mode === "full_auto" ? "Start movie production" : "Create scaffold" }}
        </button>
      </div>
    </form>
  </section>
</template>
