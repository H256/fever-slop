import type { Job, ProjectArtifacts, ProjectSummary } from "./types";

const jsonHeaders = { "Content-Type": "application/json" };

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export const api = {
  projects: () => request<ProjectSummary[]>("/api/projects"),
  project: (projectId: string) => request<ProjectSummary>(`/api/projects/${projectId}`),
  artifacts: (projectId: string) => request<ProjectArtifacts>(`/api/projects/${projectId}/artifacts`),
  artifact: (projectId: string, path: string) =>
    request<{ path: string; data: unknown }>(`/api/projects/${projectId}/artifact?path=${encodeURIComponent(path)}`),
  saveArtifact: (projectId: string, path: string, data: unknown) =>
    request<{ path: string; data: unknown }>(`/api/projects/${projectId}/artifact`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify({ path, data })
    }),
  patchScene: (projectId: string, path: string, scene: number, updates: Record<string, unknown>) =>
    request(`/api/projects/${projectId}/render-plan`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify({ path, scene, updates })
    }),
  startJob: (projectId: string, action: string, scenes?: number[]) =>
    request<Job>(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ action, scenes })
    }),
  jobs: (projectId?: string) => request<Job[]>(`/api/jobs${projectId ? `?project_id=${projectId}` : ""}`)
};

export function mediaUrl(projectId: string, path: string): string {
  return `/api/projects/${projectId}/media?path=${encodeURIComponent(path)}`;
}
