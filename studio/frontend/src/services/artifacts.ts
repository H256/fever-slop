import { jsonHeaders, request } from "./http";

export interface ArtifactResponse<T = unknown> {
  path: string;
  data: T;
}

export const artifactService = {
  get: <T = unknown>(projectId: string, path: string) =>
    request<ArtifactResponse<T>>(`/api/projects/${projectId}/artifact?path=${encodeURIComponent(path)}`),
  save: (projectId: string, path: string, data: unknown) =>
    request<ArtifactResponse>(`/api/projects/${projectId}/artifact`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify({ path, data })
    }),
  patchScene: (projectId: string, path: string, scene: number, updates: Record<string, unknown>) =>
    request<ArtifactResponse>(`/api/projects/${projectId}/render-plan`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify({ path, scene, updates })
    })
};
