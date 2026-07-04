import { expect, test } from "@playwright/test";

const movieProject = {
  id: "mud-and-silence",
  name: "Mud and Silence",
  path: "/tmp/mud-and-silence",
  project_type: "movie",
  status: { config: "present", render_plan: "present", references: "present", videos: "present" },
  artifacts: {
    configs: ["config.json"],
    render_plans: ["movie/render_plan_msr.json"],
    references: ["movie/references/manifest.json"],
    generated_json: ["movie/render_plan_msr.json"],
    videos: ["output/movie/ltx_msr/scene_0001.mp4", "output/movie/mud-and-silence.mp4"],
    images: [],
    audio: []
  }
};

test("disables movie final concat while a project pipeline is running", async ({ page }) => {
  await page.route("**/api/projects/mud-and-silence", (route) => route.fulfill({ json: movieProject }));
  await page.route("**/api/jobs?project_id=mud-and-silence", (route) =>
    route.fulfill({
      json: [
        {
          id: "job-1",
          project_id: "mud-and-silence",
          action: "movie-render",
          pipeline_type: "movie-render",
          status: "running",
          progress: 10,
          overall_progress: 10,
          current_step: "LTX MSR native-audio render",
          steps: [],
          logs: [],
          recent_logs: [],
          error: null,
          result: null,
          elapsed_seconds: 5,
          eta_seconds: null
        }
      ]
    })
  );

  await page.goto("/projects/mud-and-silence/final-video");

  await expect(page.getByRole("button", { name: /build final movie/i })).toBeDisabled();
  await expect(page.getByText("A backend process is running")).toBeVisible();
  await expect(page.getByText("Build final movie is disabled until the current job finishes.")).toBeVisible();
});

test("shows backend rejection when movie final concat cannot start", async ({ page }) => {
  await page.route("**/api/projects/mud-and-silence", (route) => route.fulfill({ json: movieProject }));
  await page.route("**/api/jobs?project_id=mud-and-silence", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/projects/mud-and-silence/jobs", (route) =>
    route.fulfill({
      status: 400,
      json: { detail: "Pipeline is already running for this project" }
    })
  );

  await page.goto("/projects/mud-and-silence/final-video");
  await page.getByRole("button", { name: /build final movie/i }).click();

  await expect(page.getByText("Pipeline is already running for this project")).toBeVisible();
});
