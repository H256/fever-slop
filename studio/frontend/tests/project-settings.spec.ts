import { expect, test } from "@playwright/test";

const project = {
  id: "demo",
  name: "Demo",
  path: "/tmp/demo",
  project_type: "standard_music_video",
  status: { config: "present", render_plan: "missing", references: "missing", videos: "missing" },
  artifacts: { configs: ["config.json"], render_plans: [], references: [], generated_json: [], videos: [], images: [], audio: [] }
};

test("project settings edits saves and resets config", async ({ page }) => {
  let savedBody: Record<string, unknown> | null = null;
  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/projects/demo/artifact?**", (route) =>
    route.fulfill({
      json: {
        path: "config.json",
        data: {
          project_name: "Demo",
          input_audio: "input/song.mp3",
          custom_plugin: { empty_but_intentional: "" }
        }
      }
    })
  );
  await page.route("**/api/projects/demo/artifact", async (route) => {
    savedBody = await route.request().postDataJSON();
    await route.fulfill({ json: savedBody });
  });

  await page.goto("/projects/demo/settings");

  await expect(page.getByRole("heading", { name: "Project Settings" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: /project name/i })).toHaveValue("Demo");
  await expect(page.getByRole("spinbutton", { name: /fps/i })).toHaveValue("24");
  await page.getByRole("textbox", { name: /project name/i }).fill("Changed");
  await expect(page.getByText("Unsaved changes")).toBeVisible();
  await page.getByRole("button", { name: /discard changes/i }).click();
  await expect(page.getByRole("textbox", { name: /project name/i })).toHaveValue("Demo");

  await page.getByRole("textbox", { name: /project name/i }).fill("Changed");
  await page.getByRole("button", { name: /^save$/i }).click();

  expect(savedBody?.data.project_name).toBe("Changed");
  expect(savedBody?.data.custom_plugin).toEqual({ empty_but_intentional: "" });
  await expect(page.getByText("Settings saved")).toBeVisible();
});

test("project settings shows frontend validation errors", async ({ page }) => {
  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/projects/demo/artifact?**", (route) =>
    route.fulfill({ json: { path: "config.json", data: { project_name: "Demo", input_audio: "input/song.mp3" } } })
  );

  await page.goto("/projects/demo/settings");
  await page.getByRole("textbox", { name: /input audio/i }).fill("");

  await expect(page.getByText("Input audio is required.")).toBeVisible();
  await expect(page.getByRole("button", { name: /^save$/i })).toBeDisabled();
});
