import { defineStore } from "pinia";
import { api } from "../api";
import type { Job, ProjectCreatePayload, ProjectSummary } from "../types";

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
    async createProject(payload: ProjectCreatePayload) {
      const project = await api.createProject(payload);
      this.projects = [project, ...this.projects.filter((item) => item.id !== project.id)].sort((a, b) => a.id.localeCompare(b.id));
      return project;
    },
    async loadJobs(projectId?: string) {
      this.jobs = await api.jobs(projectId);
    },
    async startJob(projectId: string, action: string, scenes?: number[], extra?: Record<string, unknown>) {
      const job = await api.startJob(projectId, action, scenes, extra);
      this.jobs.unshift(job);
      return job;
    }
  }
});
