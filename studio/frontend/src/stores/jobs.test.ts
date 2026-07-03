import { beforeEach, describe, expect, test } from "bun:test";
import { createPinia, setActivePinia } from "pinia";
import { useJobStore } from "./jobs";
import type { Job } from "../types";

describe("useJobStore", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  test("loads jobs for a project", async () => {
    const originalFetch = globalThis.fetch;
    const job = jobSummary("job-1");
    globalThis.fetch = async (url) => {
      expect(String(url)).toBe("/api/jobs?project_id=demo");
      return new Response(JSON.stringify([job]), { status: 200 });
    };

    try {
      const store = useJobStore();

      await store.loadJobs("demo");

      expect(store.jobs).toEqual([job]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("starts jobs and prepends them to state", async () => {
    const originalFetch = globalThis.fetch;
    const started = jobSummary("job-2");
    globalThis.fetch = async (_url, init) => {
      expect(init?.method).toBe("POST");
      expect(JSON.parse(String(init?.body))).toEqual({ action: "recut-scene", scenes: [1], raw_clip: "raw.mp4" });
      return new Response(JSON.stringify(started), { status: 200 });
    };

    try {
      const store = useJobStore();
      store.jobs = [jobSummary("job-1")];

      const result = await store.startJob("demo", "recut-scene", [1], { raw_clip: "raw.mp4" });

      expect(result).toEqual(started);
      expect(store.jobs.map((job) => job.id)).toEqual(["job-2", "job-1"]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

function jobSummary(id: string): Job {
  return {
    id,
    project_id: "demo",
    action: "test",
    status: "queued",
    progress: 0,
    logs: [],
    error: null,
    result: null
  };
}
