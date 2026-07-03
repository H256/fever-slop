import { defineStore } from "pinia";
import { projectService } from "../services/projects";
import type { ProjectCreatePayload, ProjectSummary } from "../types";

interface ProjectState {
  projects: ProjectSummary[];
  currentProject: ProjectSummary | null;
}

export const useProjectStore = defineStore("projects", {
  state: (): ProjectState => ({
    projects: [],
    currentProject: null
  }),
  actions: {
    async loadProjects() {
      this.projects = await projectService.list();
    },
    async loadProject(projectId: string) {
      this.currentProject = await projectService.get(projectId);
    },
    async createProject(payload: ProjectCreatePayload) {
      const project = await projectService.create(payload);
      this.projects = [project, ...this.projects.filter((item) => item.id !== project.id)].sort((a, b) => a.id.localeCompare(b.id));
      return project;
    }
  }
});
