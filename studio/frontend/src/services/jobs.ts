import type { Job } from "../types";
import { jsonHeaders, request } from "./http";

export const jobService = {
  start: (projectId: string, action: string, scenes?: number[], extra?: Record<string, unknown>) =>
    request<Job>(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ action, scenes, ...extra })
    }),
  list: (projectId?: string) => request<Job[]>(`/api/jobs${projectId ? `?project_id=${projectId}` : ""}`)
};

export function jobLogsUrl(jobId: string): string {
  return `/api/jobs/${jobId}/logs`;
}
