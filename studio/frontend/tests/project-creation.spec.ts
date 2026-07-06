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
      expect(body).toEqual({ project_type: "standard_music_video", name: "My Cool Video", silent_mode: true });
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
    const body = await route.request().postDataJSON();
    expect(body.data.project_name).toBe("My Cool Video");
    expect(body.data.video).toEqual({ fps: 24, width: 1280, height: 704 });
    expect(body.data.lyrics).toBeUndefined();
    expect(body.data.story_idea).toBeUndefined();
    expect(body.data.actors).toEqual([{ name: "Mara" }]);
    await route.fulfill({ json: { path: "config.json", data: body.data } });
  });
  await page.goto("/");
  await page.getByRole("button", { name: /create project/i }).click();
  await page.getByRole("button", { name: /standard/i }).click();
  await page.getByLabel("Project name").fill("My Cool Video");
  await page.getByRole("switch", { name: /silent mode/i }).click();
  await expect(page.getByText("my-cool-video")).toBeVisible();
  await page.getByRole("button", { name: /create and configure/i }).click();

  await expect(page).toHaveURL(/\/projects\/my-cool-video\/settings/);
  await expect(page.getByRole("heading", { name: "Project Settings" })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: /fps/i })).toHaveValue("24");
  await expect(page.getByRole("spinbutton", { name: /word count min/i })).toHaveValue("40");
  await page.getByRole("textbox", { name: /input audio/i }).fill("input/song.mp3");
  await page.getByRole("button", { name: "Add actors" }).click();
  await page.locator(".array-item-block", { hasText: "actors 1" }).getByRole("textbox", { name: /^name/i }).fill("Mara");
  await page.getByRole("button", { name: /^save$/i }).click();
  await expect(page.getByText("Settings saved")).toBeVisible();
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
        song_style: "dark synthwave with cinematic drums",
        duration_seconds: 150,
        width: 1280,
        height: 704,
        fps: 24,
        pipeline_mode: "msr",
        silent_mode: true
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
  await page.getByLabel("Desired video duration").fill("150");
  await page.getByLabel("Pipeline mode").selectOption("msr");
  await page.getByRole("switch", { name: /silent mode/i }).click();
  await page.getByRole("button", { name: /start full-auto pipeline/i }).click();

  await expect(page).toHaveURL(/\/projects\/neon-wolves\/pipeline/);
  expect(jobStarted).toBe(true);
});

test("creates a movie project and starts movie full-auto production", async ({ page }) => {
  let jobStarted = false;
  await page.route("**/api/projects", async (route) => {
    if (route.request().method() === "POST") {
      const body = await route.request().postDataJSON();
      expect(body).toEqual({
        project_type: "movie",
        name: "Door Below",
        source_type: "short_story",
        story_text: "A locksmith finds a glowing door below an abandoned station.",
        desired_length: 180,
        width: 1280,
        height: 704,
        movie_mode: "full_auto"
      });
      await route.fulfill({
        json: {
          id: "door-below",
          name: "Door Below",
          path: "/tmp/door-below",
          project_type: "movie",
          status: { config: "missing", render_plan: "present", references: "missing", videos: "missing" },
          artifacts: {
            ...emptyArtifacts,
            configs: [],
            render_plans: ["movie/render_plan.json"],
            generated_json: ["movie/story_arch.json", "movie/render_plan.json"]
          }
        }
      });
      return;
    }
    await route.fulfill({ json: [] });
  });
  await page.route("**/api/projects/door-below/jobs", async (route) => {
    const body = await route.request().postDataJSON();
    expect(body.action).toBe("movie-full-auto");
    jobStarted = true;
    await route.fulfill({
      json: {
        id: "movie-job-1",
        project_id: "door-below",
        action: "movie-full-auto",
        pipeline_type: "movie-full-auto",
        status: "queued",
        progress: 0,
        logs: [],
        recent_logs: [],
        error: null,
        result: null
      }
    });
  });
  await page.route("**/api/projects/door-below", (route) =>
    route.fulfill({
      json: {
        id: "door-below",
        name: "Door Below",
        path: "/tmp/door-below",
        project_type: "movie",
        status: { config: "missing", render_plan: "present", references: "missing", videos: "missing" },
        artifacts: {
          ...emptyArtifacts,
          configs: [],
          render_plans: ["movie/render_plan.json"],
          generated_json: ["movie/story_arch.json", "movie/render_plan.json"]
        }
      }
    })
  );
  await page.route("**/api/jobs?project_id=door-below", (route) => route.fulfill({ json: [] }));

  await page.goto("/");
  await page.getByRole("button", { name: /create project/i }).click();
  await page.getByRole("link", { name: /new movie project/i }).click();
  await page.getByLabel("Project name").fill("Door Below");
  await page.getByRole("textbox", { name: "Short story idea" }).fill("A locksmith finds a glowing door below an abandoned station.");
  await page.getByLabel("Mode").selectOption("full_auto");
  await page.getByRole("button", { name: /start movie production/i }).click();

  await expect(page).toHaveURL(/\/projects\/door-below\/pipeline/);
  expect(jobStarted).toBe(true);
});
