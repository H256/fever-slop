import { defineStore } from "pinia";
import { jobService } from "../services/jobs";
import type { Job } from "../types";

interface JobState {
  jobs: Job[];
}

export const useJobStore = defineStore("jobs", {
  state: (): JobState => ({
    jobs: []
  }),
  actions: {
    async loadJobs(projectId?: string) {
      this.jobs = await jobService.list(projectId);
    },
    async startJob(projectId: string, action: string, scenes?: number[], extra?: Record<string, unknown>) {
      const job = await jobService.start(projectId, action, scenes, extra);
      this.jobs.unshift(job);
      return job;
    }
  }
});
