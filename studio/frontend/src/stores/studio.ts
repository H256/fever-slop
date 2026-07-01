import { defineStore } from "pinia";
import { api } from "../api";
import type { Job, ProjectSummary } from "../types";

export const useStudioStore = defineStore("studio", {
  state: () => ({
    projects: [] as ProjectSummary[],
    currentProject: null as ProjectSummary | null,
    jobs: [] as Job[],
    error: ""
  }),
  actions: {
    async loadProjects() {
      this.projects = await api.projects();
    },
    async loadProject(projectId: string) {
      this.currentProject = await api.project(projectId);
    },
    async loadJobs(projectId?: string) {
      this.jobs = await api.jobs(projectId);
    },
    async startJob(projectId: string, action: string, scenes?: number[]) {
      const job = await api.startJob(projectId, action, scenes);
      this.jobs.unshift(job);
      return job;
    }
  }
});
