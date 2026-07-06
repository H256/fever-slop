import { expect, test } from "@playwright/test";

test("shows and downloads the final project video", async ({ page }) => {
  await page.route("**/api/projects/demo", (route) =>
    route.fulfill({
      json: {
        id: "demo",
        name: "Demo",
        path: "/tmp/demo",
        project_type: "standard_music_video",
        status: { config: "present", render_plan: "present", references: "present", videos: "present" },
        artifacts: {
          configs: ["config.json"],
          render_plans: [],
          references: [],
          generated_json: [],
          videos: ["output/render/ltx_msr/scene_0001.mp4", "output/render/ltx_msr/Demo_video_only.mp4", "output/render/ltx_msr/Demo.mp4"],
          images: [],
          audio: []
        }
      }
    })
  );

  await page.goto("/projects/demo/final-video");

  await expect(page.getByRole("heading", { name: "Final Video" })).toBeVisible();
  await expect(page.locator("video")).toHaveAttribute("src", /output%2Frender%2Fltx_msr%2FDemo\.mp4/);
  await expect(page.getByRole("link", { name: /download/i })).toHaveAttribute("download", "");
});

test("prefers movie final output over raw and scene clips", async ({ page }) => {
  await page.route("**/api/projects/mud-and-silence", (route) =>
    route.fulfill({
      json: {
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
          videos: [
            "output/movie/ltx_msr/raw/scene_0001_raw.mp4",
            "output/movie/ltx_msr/scene_0001.mp4",
            "output/movie/ltx_msr/scene_0002.mp4",
            "output/movie/mud-and-silence.mp4"
          ],
          images: [],
          audio: []
        }
      }
    })
  );
  await page.route("**/api/jobs?project_id=mud-and-silence", (route) => route.fulfill({ json: [] }));

  await page.goto("/projects/mud-and-silence/final-video");

  await expect(page.locator("video")).toHaveAttribute("src", /output%2Fmovie%2Fmud-and-silence\.mp4/);
  await expect(page.getByText("output/movie/mud-and-silence.mp4")).toBeVisible();
});

