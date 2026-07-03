import { expect, test } from "@playwright/test";

test("shows movie production metadata and current stage on dashboard", async ({ page }) => {
  const project = {
    id: "door-below",
    name: "Door Below",
    path: "/tmp/door-below",
    project_type: "movie",
    metadata: {
      project_type: "movie",
      movie: {
        source_type: "screenplay",
        desired_length: 180,
        width: 1280,
        height: 704,
        mode: "full_auto"
      }
    },
    status: { config: "missing", render_plan: "present", references: "present", videos: "missing" },
    artifacts: {
      configs: [],
      render_plans: ["movie/render_plan.json"],
      references: ["movie/references/manifest.json"],
      generated_json: ["movie/story_arch.json", "movie/render_plan.json"],
      videos: [],
      images: [],
      audio: []
    },
    artifact_sizes: { total_bytes: 0, by_type: {} }
  };

  await page.route("**/api/projects/door-below", (route) => route.fulfill({ json: project }));
  await page.route("**/api/jobs?project_id=door-below", (route) =>
    route.fulfill({
      json: [
        {
          id: "movie-job-1",
          project_id: "door-below",
          action: "movie-full-auto",
          pipeline_type: "movie-full-auto",
          status: "running",
          progress: 41,
          overall_progress: 41,
          current_step: "Rendering shot 12/40",
          steps: [],
          logs: [],
          recent_logs: [],
          error: null,
          result: null,
          elapsed_seconds: 120,
          eta_seconds: 180
        }
      ]
    })
  );

  await page.goto("/projects/door-below");

  await expect(page.getByRole("heading", { name: "Movie Production" })).toBeVisible();
  await expect(page.getByText("screenplay")).toBeVisible();
  await expect(page.getByText("180s")).toBeVisible();
  await expect(page.getByText("1280 x 704")).toBeVisible();
  await expect(page.getByText("Rendering shot 12/40")).toBeVisible();
});
