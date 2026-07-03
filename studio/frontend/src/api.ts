import { artifactService } from "./services/artifacts";
import { jobLogsUrl, jobService } from "./services/jobs";
import { mediaService, mediaUrl, thumbnailUrl } from "./services/media";
import { projectService } from "./services/projects";
import { request } from "./services/http";
import type { ProjectArtifacts, ProjectCreatePayload } from "./types";

export const api = {
  projects: () => projectService.list(),
  createProject: (payload: ProjectCreatePayload) => projectService.create(payload),
  project: (projectId: string) => projectService.get(projectId),
  uploadAudio: (projectId: string, file: File) => projectService.uploadAudio(projectId, file),
  artifacts: (projectId: string) => request<ProjectArtifacts>(`/api/projects/${projectId}/artifacts`),
  artifact: <T = unknown>(projectId: string, path: string) => artifactService.get<T>(projectId, path),
  saveArtifact: (projectId: string, path: string, data: unknown) => artifactService.save(projectId, path, data),
  uploadMedia: (projectId: string, path: string, dataUrl: string) => mediaService.upload(projectId, path, dataUrl),
  patchScene: (projectId: string, path: string, scene: number, updates: Record<string, unknown>) => artifactService.patchScene(projectId, path, scene, updates),
  startJob: (projectId: string, action: string, scenes?: number[], extra?: Record<string, unknown>) => jobService.start(projectId, action, scenes, extra),
  jobs: (projectId?: string) => jobService.list(projectId)
};

export { jobLogsUrl, mediaUrl, thumbnailUrl };
