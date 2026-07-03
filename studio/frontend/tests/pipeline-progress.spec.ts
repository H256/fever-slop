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
  await expect(page.getByRole("button", { name: /full pipeline/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /anchor fix/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /storyboard frames/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /mux original audio/i })).toBeDisabled();
  await expect(page.getByText("Pipeline is already running")).toBeVisible();
});

test("groups standard pipeline actions by phase and starts the selected action", async ({ page }) => {
  const project = {
    id: "demo",
    name: "Demo",
    path: "/tmp/demo",
    project_type: "standard_music_video",
    status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
    artifacts: { configs: ["config.json"], render_plans: [], references: [], generated_json: [], videos: [], images: [], audio: [] }
  };
  let startedAction = "";

  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/jobs?project_id=demo", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/projects/demo/jobs", async (route) => {
    const body = route.request().postDataJSON() as { action: string };
    startedAction = body.action;
    await route.fulfill({
      json: {
        id: "job-2",
        project_id: "demo",
        action: body.action,
        pipeline_type: body.action,
        status: "queued",
        progress: 0,
        overall_progress: 0,
        current_step: null,
        steps: [],
        logs: [],
        recent_logs: [],
        error: null,
        result: null,
        elapsed_seconds: 0,
        eta_seconds: null
      }
    });
  });

  await page.goto("/projects/demo/pipeline");

  const expectedGroups = ["Core runs", "Preparation", "Storyboard", "Generation", "Post-processing"];
  for (const group of expectedGroups) {
    await expect(page.getByRole("heading", { name: group })).toBeVisible();
  }

  const expectedActions = [
    "Full pipeline",
    "Main pipeline",
    "Relay compact",
    "Anchor fix",
    "Rebuild plan",
    "Storyboard",
    "Storyboard frames",
    "Storyboard page",
    "MSR references",
    "MSR reference sheets",
    "MSR enrichment",
    "MSR prompt enrichment",
    "Render selected scenes",
    "Final concat",
    "Concat video only",
    "Mux original audio"
  ];
  for (const action of expectedActions) {
    await expect(page.getByRole("button", { name: action, exact: true })).toBeVisible();
  }

  await page.getByRole("button", { name: "Mux original audio", exact: true }).click();
  await page.getByRole("button", { name: "Run job", exact: true }).click();

  expect(startedAction).toBe("mux-original-audio");
});

test("omits empty phases for full-auto projects", async ({ page }) => {
  const project = {
    id: "demo",
    name: "Demo",
    path: "/tmp/demo",
    project_type: "full_auto",
    status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
    artifacts: { configs: ["config.json"], render_plans: [], references: [], generated_json: [], videos: [], images: [], audio: [] }
  };

  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/jobs?project_id=demo", (route) => route.fulfill({ json: [] }));

  await page.goto("/projects/demo/pipeline");

  await expect(page.getByRole("heading", { name: "Core runs" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Full-auto pipeline", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Preparation" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Full pipeline", exact: true })).toHaveCount(0);
});
