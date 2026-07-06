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
  project_type?: "standard_music_video" | "full_auto" | "movie";
  silent_mode?: boolean;
  metadata?: {
    project_type?: "standard_music_video" | "full_auto" | "movie";
    display_name?: string;
    slug?: string;
    silent_mode?: boolean;
    full_auto?: {
      idea?: string;
      song_style?: string;
      duration_seconds?: number;
      width?: number;
      height?: number;
      fps?: 16 | 24 | 50;
      pipeline_mode?: "classic" | "msr";
    };
    movie?: {
      source_type?: "short_story" | "screenplay";
      story_text?: string;
      desired_length?: number;
      dialogue_language?: string;
      width?: number;
      height?: number;
      mode?: "scaffold" | "full_auto";
      planner_backend?: "llm" | "deterministic";
      reference_backend?: "comfyui" | "local";
      render_backend?: "comfyui" | "local";
      hero_workflow?: string;
      edit_workflow?: string;
      msr_workflow?: string;
      msr_i2v_workflow?: string;
      movie_video_workflow?: "msr" | "msr-i2v-startframe";
      continuity_keyframes?: "none" | "last-to-start";
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
  project_type: "standard_music_video" | "full_auto" | "movie";
  name: string;
  silent_mode?: boolean;
  idea?: string;
  song_style?: string;
  duration_seconds?: number;
  width?: number;
  height?: number;
  fps?: 16 | 24 | 50;
  pipeline_mode?: "classic" | "msr";
  source_type?: "short_story" | "screenplay";
  story_text?: string;
  desired_length?: number;
  dialogue_language?: string;
  movie_mode?: "scaffold" | "full_auto";
  movie_planner_backend?: "llm" | "deterministic";
  movie_reference_backend?: "comfyui" | "local";
  movie_render_backend?: "comfyui" | "local";
  movie_hero_workflow?: string;
  movie_edit_workflow?: string;
  movie_msr_workflow?: string;
  movie_msr_i2v_workflow?: string;
  movie_video_workflow?: "msr" | "msr-i2v-startframe";
  movie_continuity_keyframes?: "none" | "last-to-start";
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
