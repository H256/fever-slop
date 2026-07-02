import type { ProjectCreatePayload, ProjectSummary } from "../types";
import { jsonHeaders, request } from "./http";

export const projectService = {
  list: () => request<ProjectSummary[]>("/api/projects"),
  create: (payload: ProjectCreatePayload) =>
    request<ProjectSummary>("/api/projects", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  get: (projectId: string) => request<ProjectSummary>(`/api/projects/${projectId}`),
  uploadAudio: (projectId: string, file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<{ path: string }>(`/api/projects/${projectId}/upload-audio`, {
      method: "POST",
      body: formData
    });
  }
};
