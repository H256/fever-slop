import { beforeEach, describe, expect, test } from "bun:test";
import { createPinia, setActivePinia } from "pinia";
import { useProjectStore } from "./projects";
import type { ProjectCreatePayload, ProjectSummary } from "../types";

describe("useProjectStore", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  test("loads project lists and a current project through typed API services", async () => {
    const originalFetch = globalThis.fetch;
    const project = projectSummary("demo");
    globalThis.fetch = async (url) => new Response(JSON.stringify(String(url).endsWith("/demo") ? project : [project]), { status: 200 });

    try {
      const store = useProjectStore();

      await store.loadProjects();
      await store.loadProject("demo");

      expect(store.projects).toEqual([project]);
      expect(store.currentProject).toEqual(project);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("inserts created projects once and keeps project ids sorted", async () => {
    const originalFetch = globalThis.fetch;
    const existing = projectSummary("zeta");
    const created = projectSummary("alpha", "Alpha");
    globalThis.fetch = async () => new Response(JSON.stringify(created), { status: 200 });

    try {
      const store = useProjectStore();
      store.projects = [existing, projectSummary("alpha", "Old Alpha")];

      const payload: ProjectCreatePayload = { project_type: "standard_music_video", name: "Alpha" };
      const result = await store.createProject(payload);

      expect(result).toEqual(created);
      expect(store.projects.map((project) => project.id)).toEqual(["alpha", "zeta"]);
      expect(store.projects[0].name).toBe("Alpha");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

function projectSummary(id: string, name = id): ProjectSummary {
  return {
    id,
    name,
    path: `/projects/${id}`,
    status: {},
    artifacts: {
      configs: [],
      render_plans: [],
      references: [],
      generated_json: [],
      videos: [],
      images: [],
      audio: []
    }
  };
}
