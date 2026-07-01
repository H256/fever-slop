export type ProjectStatus = Record<string, "present" | "missing">;

export interface ProjectArtifacts {
  configs: string[];
  render_plans: string[];
  references: string[];
  generated_json: string[];
  videos: string[];
  images: string[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  path: string;
  status: ProjectStatus;
  artifacts: ProjectArtifacts;
}

export interface Job {
  id: string;
  project_id: string;
  action: string;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  logs: string[];
  error: string | null;
  result: string | null;
}

export type RenderScene = Record<string, unknown> & { scene: number };
