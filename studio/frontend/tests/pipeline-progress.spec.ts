import { expect, test } from "@playwright/test";

test("shows structured pipeline progress and recent logs", async ({ page }) => {
  const project = {
    id: "demo",
    name: "Demo",
    path: "/tmp/demo",
    project_type: "standard_music_video",
    status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
    artifacts: { configs: ["config.json"], render_plans: [], references: [], generated_json: [], videos: [], images: [], audio: [] }
  };
  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/jobs?project_id=demo", (route) =>
    route.fulfill({
      json: [
        {
          id: "job-1",
          project_id: "demo",
          action: "full-pipeline",
          pipeline_type: "full-pipeline",
          status: "running",
          progress: 33,
          overall_progress: 33,
          current_step: "MSR references",
          steps: [
            { name: "Main pipeline", status: "completed", progress: 100, started_at: 1, completed_at: 2, elapsed_seconds: 1 },
            { name: "MSR references", status: "running", progress: null, started_at: 2, completed_at: null, elapsed_seconds: 3 },
            { name: "LTX render", status: "pending", progress: null, started_at: null, completed_at: null, elapsed_seconds: 0 }
          ],
          logs: ["Starting full-pipeline", "MSR references"],
          recent_logs: ["Starting full-pipeline", "MSR references"],
          error: null,
          result: null,
          elapsed_seconds: 4,
          eta_seconds: null
        }
      ]
    })
  );
  await page.route("**/api/jobs/job-1/logs", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body: 'data: {"line":"Starting full-pipeline"}\n\ndata: {"line":"MSR references"}\n\n'
    })
  );

  await page.goto("/projects/demo/pipeline");

  await expect(page.getByRole("heading", { name: "full-pipeline" })).toBeVisible();
  await expect(page.getByText("Current MSR references")).toBeVisible();
  await expect(page.locator(".step-row", { hasText: "Main pipeline" })).toBeVisible();
  await expect(page.getByText("completed")).toBeVisible();
  await expect(page.locator(".step-row", { hasText: "MSR references" })).toBeVisible();
  await expect(page.locator(".live-log")).toContainText("Starting full-pipeline");
});
