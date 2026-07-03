import { jsonHeaders, request } from "./http";

export const mediaService = {
  upload: (projectId: string, path: string, dataUrl: string) =>
    request<{ path: string }>(`/api/projects/${projectId}/media`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify({ path, data_url: dataUrl })
    })
};

export function mediaUrl(projectId: string, path: string): string {
  return `/api/projects/${projectId}/media?path=${encodeURIComponent(path)}`;
}

export function thumbnailUrl(projectId: string, path: string, at: number): string {
  return `/api/projects/${projectId}/thumbnail?path=${encodeURIComponent(path)}&at=${encodeURIComponent(at.toFixed(2))}`;
}
