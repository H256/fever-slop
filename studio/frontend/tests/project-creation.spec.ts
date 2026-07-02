import { expect, test } from "@playwright/test";

const emptyArtifacts = {
  configs: ["config.json"],
  render_plans: [],
  references: [],
  generated_json: [],
  videos: [],
  images: [],
  audio: []
};

test("creates a standard project and opens the config form", async ({ page }) => {
  await page.route("**/api/projects", async (route) => {
    if (route.request().method() === "POST") {
      const body = await route.request().postDataJSON();
      expect(body).toEqual({ project_type: "standard_music_video", name: "My Cool Video" });
      await route.fulfill({
        json: {
          id: "my-cool-video",
          name: "My Cool Video",
          path: "/tmp/my-cool-video",
          project_type: "standard_music_video",
          status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
          artifacts: emptyArtifacts
        }
      });
      return;
    }
    await route.fulfill({
      json: [
        {
          id: "existing",
          name: "Existing",
          path: "/tmp/existing",
          status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
          artifacts: emptyArtifacts
        }
      ]
    });
  });
  await page.route("**/api/projects/my-cool-video", (route) =>
    route.fulfill({
      json: {
        id: "my-cool-video",
        name: "My Cool Video",
        path: "/tmp/my-cool-video",
        project_type: "standard_music_video",
        status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
        artifacts: emptyArtifacts
      }
    })
  );
  await page.route("**/api/projects/my-cool-video/artifact?**", (route) =>
    route.fulfill({ json: { path: "config.json", data: { project_name: "My Cool Video", input_audio: "" } } })
  );
  await page.route("**/api/projects/my-cool-video/artifact", async (route) => {
    expect(route.request().method()).toBe("PUT");
    await route.fulfill({ json: { path: "config.json", data: { project_name: "My Cool Video", input_audio: "" } } });
  });
  await page.route("**/api/projects/my-cool-video/jobs", async (route) => {
    const body = await route.request().postDataJSON();
    expect(body.action).toBe("full-pipeline");
    await route.fulfill({
      json: {
        id: "standard-job",
        project_id: "my-cool-video",
        action: "full-pipeline",
        pipeline_type: "full-pipeline",
        status: "queued",
        progress: 0,
        logs: [],
        recent_logs: [],
        error: null,
        result: null
      }
    });
  });
  await page.route("**/api/jobs?project_id=my-cool-video", (route) =>
    route.fulfill({
      json: [
        {
          id: "standard-job",
          project_id: "my-cool-video",
          action: "full-pipeline",
          pipeline_type: "full-pipeline",
          status: "queued",
          progress: 0,
          logs: [],
          recent_logs: [],
          error: null,
          result: null
        }
      ]
    })
  );

  await page.goto("/");
  await page.getByRole("button", { name: /create project/i }).click();
  await page.getByRole("button", { name: /standard/i }).click();
  await page.getByLabel("Project name").fill("My Cool Video");
  await expect(page.getByText("my-cool-video")).toBeVisible();
  await page.getByRole("button", { name: /create and configure/i }).click();

  await expect(page).toHaveURL(/\/projects\/my-cool-video\/artifacts\?path=config\.json/);
  await expect(page.getByRole("heading", { name: "config.json" })).toBeVisible();
  await page.getByRole("button", { name: /save and run pipeline/i }).click();
  await page.getByRole("button", { name: "Save and run", exact: true }).click();

  await expect(page).toHaveURL(/\/projects\/my-cool-video\/pipeline/);
  await expect(page.getByRole("heading", { name: "full-pipeline" })).toBeVisible();
});

test("validates duplicate slugs before creation", async ({ page }) => {
  await page.route("**/api/projects", (route) =>
    route.fulfill({
      json: [
        {
          id: "my-cool-video",
          name: "Existing",
          path: "/tmp/my-cool-video",
          status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
          artifacts: emptyArtifacts
        }
      ]
    })
  );

  await page.goto("/");
  await page.getByRole("button", { name: /create project/i }).click();
  await page.getByRole("button", { name: /standard/i }).click();
  await page.getByLabel("Project name").fill("My Cool Video");

  await expect(page.getByText(/already exists/)).toBeVisible();
  await expect(page.getByRole("button", { name: /create and configure/i })).toBeDisabled();
});

test("creates a full-auto project and starts its pipeline", async ({ page }) => {
  let jobStarted = false;
  await page.route("**/api/projects", async (route) => {
    if (route.request().method() === "POST") {
      const body = await route.request().postDataJSON();
      expect(body).toEqual({
        project_type: "full_auto",
        name: "Neon Wolves",
        idea: "A cyberpunk chase through a futuristic city",
        song_style: "dark synthwave with cinematic drums"
      });
      await route.fulfill({
        json: {
          id: "neon-wolves",
          name: "Neon Wolves",
          path: "/tmp/neon-wolves",
          project_type: "full_auto",
          status: { config: "missing", render_plan: "missing", references: "missing", videos: "missing" },
          artifacts: { ...emptyArtifacts, configs: [] }
        }
      });
      return;
    }
    await route.fulfill({ json: [] });
  });
  await page.route("**/api/projects/neon-wolves/jobs", async (route) => {
    const body = await route.request().postDataJSON();
    expect(body.action).toBe("full-auto");
    jobStarted = true;
    await route.fulfill({
      json: {
        id: "job-1",
        project_id: "neon-wolves",
        action: "full-auto",
        pipeline_type: "full-auto",
        status: "queued",
        progress: 0,
        logs: [],
        recent_logs: [],
        error: null,
        result: null
      }
    });
  });
  await page.route("**/api/projects/neon-wolves", (route) =>
    route.fulfill({
      json: {
        id: "neon-wolves",
        name: "Neon Wolves",
        path: "/tmp/neon-wolves",
        project_type: "full_auto",
        status: { config: "missing", render_plan: "missing", references: "missing", videos: "missing" },
        artifacts: { ...emptyArtifacts, configs: [] }
      }
    })
  );
  await page.route("**/api/jobs?project_id=neon-wolves", (route) =>
    route.fulfill({
      json: [
        {
          id: "job-1",
          project_id: "neon-wolves",
          action: "full-auto",
          pipeline_type: "full-auto",
          status: "queued",
          progress: 0,
          logs: [],
          recent_logs: [],
          error: null,
          result: null
        }
      ]
    })
  );

  await page.goto("/");
  await page.getByRole("button", { name: /create project/i }).click();
  await page.getByRole("button", { name: /full-auto/i }).click();
  await page.getByLabel("Project name").fill("Neon Wolves");
  await page.getByLabel("Idea").fill("A cyberpunk chase through a futuristic city");
  await page.getByLabel("Song style").fill("dark synthwave with cinematic drums");
  await page.getByRole("button", { name: /start full-auto pipeline/i }).click();

  await expect(page).toHaveURL(/\/projects\/neon-wolves\/pipeline/);
  expect(jobStarted).toBe(true);
});
