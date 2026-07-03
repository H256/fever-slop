export type ProjectStatus = Record<string, "present" | "missing">;

export interface ProjectArtifacts {
  configs: string[];
  render_plans: string[];
  references: string[];
  generated_json: string[];
  videos: string[];
  images: string[];
  audio: string[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  path: string;
  project_type?: "standard_music_video" | "full_auto";
  metadata?: {
    project_type?: "standard_music_video" | "full_auto";
    display_name?: string;
    slug?: string;
    full_auto?: {
      idea?: string;
      song_style?: string;
      duration_seconds?: number;
      width?: number;
      height?: number;
      fps?: 16 | 24 | 50;
      pipeline_mode?: "classic" | "msr";
    };
  };
  status: ProjectStatus;
  artifacts: ProjectArtifacts;
  artifact_sizes?: {
    total_bytes: number;
    by_type: Record<string, number>;
  };
}

export interface ProjectCreatePayload {
  project_type: "standard_music_video" | "full_auto";
  name: string;
  idea?: string;
  song_style?: string;
  duration_seconds?: number;
  width?: number;
  height?: number;
  fps?: 16 | 24 | 50;
  pipeline_mode?: "classic" | "msr";
}

export interface JobStep {
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped" | "cancelled";
  progress: number | null;
  started_at: number | null;
  completed_at: number | null;
  elapsed_seconds: number;
}

export interface Job {
  id: string;
  project_id: string;
  action: string;
  project_type?: string;
  pipeline_type?: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  overall_progress?: number;
  current_step?: string | null;
  steps?: JobStep[];
  logs: string[];
  recent_logs?: string[];
  error: string | null;
  result: string | null;
  created_at?: number;
  started_at?: number | null;
  completed_at?: number | null;
  updated_at?: number;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
}

export type RenderScene = Record<string, unknown> & { scene: number };
