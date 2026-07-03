import { defineStore } from "pinia";
import { computed } from "vue";
import { useJobStore } from "./jobs";
import { useProjectStore } from "./projects";

export const useStudioStore = defineStore("studio", () => {
  const projectsStore = useProjectStore();
  const jobsStore = useJobStore();

  return {
    currentProject: computed(() => projectsStore.currentProject),
    error: "",
    jobs: computed(() => jobsStore.jobs),
    loadJobs: jobsStore.loadJobs,
    loadProject: projectsStore.loadProject,
    loadProjects: projectsStore.loadProjects,
    projects: computed(() => projectsStore.projects),
    createProject: projectsStore.createProject,
    startJob: jobsStore.startJob
  };
});
