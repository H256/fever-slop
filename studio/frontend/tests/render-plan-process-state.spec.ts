import { expect, test } from "@playwright/test";

test("disables render plan rerender controls while a backend process is active", async ({ page }) => {
  const project = {
    id: "demo",
    name: "Demo",
    path: "/tmp/demo",
    project_type: "standard_music_video",
    status: { config: "present", render_plan: "present", references: "missing", videos: "missing" },
    artifacts: {
      configs: ["config.json"],
      render_plans: ["output/render/render_plan_song.json"],
      references: [],
      generated_json: [],
      videos: [],
      images: [],
      audio: []
    }
  };
  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/projects/demo/artifact?path=output%2Frender%2Frender_plan_song.json", (route) =>
    route.fulfill({ json: { path: "output/render/render_plan_song.json", data: [{ scene: 1, prompt: "one" }] } })
  );
  await page.route("**/api/jobs?project_id=demo", (route) =>
    route.fulfill({
      json: [
        {
          id: "job-1",
          project_id: "demo",
          action: "ltx-render-scenes",
          status: "running",
          progress: 0,
          logs: [],
          error: null,
          result: null
        }
      ]
    })
  );

  await page.goto("/projects/demo/render-plan");

  await expect(page.getByRole("button", { name: /rerender selected/i })).toBeDisabled();
  await expect(page.getByRole("button", { name: /rerender all/i })).toBeDisabled();
  await expect(page.getByText("A backend process is running")).toBeVisible();
});
