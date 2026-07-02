import { expect, test } from "@playwright/test";

const renderPlan = [
  { scene: 1, fps: 24, frame_count: 48, abs_start_seconds: 0, abs_end_seconds: 2, duration_seconds: 2, ltx: { base_prompt: "first" } },
  { scene: 2, fps: 24, frame_count: 48, abs_start_seconds: 2, abs_end_seconds: 4, duration_seconds: 2, ltx: { base_prompt: "second" } },
  { scene: 3, fps: 24, frame_count: 48, abs_start_seconds: 4, abs_end_seconds: 6, duration_seconds: 2, ltx: { base_prompt: "third" } }
];

const project = {
  id: "demo",
  name: "Demo",
  path: "/tmp/demo",
  status: { config: "present", render_plan: "present", references: "missing", videos: "present" },
  artifacts: {
    configs: ["config.json"],
    render_plans: ["output/render/render_plan.json"],
    references: [],
    generated_json: ["output/render/render_manifest.json"],
    videos: [
      "output/render/ltx_msr/raw/scene_0001_raw.mp4",
      "output/render/ltx_msr/raw/scene_0002_raw.mp4",
      "output/render/ltx_msr/raw/scene_0003_raw.mp4",
      "output/render/ltx_msr/scene_0001.mp4",
      "output/render/ltx_msr/scene_0002.mp4",
      "output/render/ltx_msr/scene_0003.mp4"
    ],
    images: [],
    audio: []
  }
};

const manifest = [
  { scene: 1, scene_frame_count: 48, render_frame_count: 72, trim_front_frames: 12, tail_loss_frames: 12, audio_start_seconds: 0, audio_duration_seconds: 3 },
  { scene: 2, scene_frame_count: 48, render_frame_count: 72, trim_front_frames: 12, tail_loss_frames: 12, audio_start_seconds: 1.5, audio_duration_seconds: 3 },
  { scene: 3, scene_frame_count: 48, render_frame_count: 72, trim_front_frames: 12, tail_loss_frames: 12, audio_start_seconds: 3, audio_duration_seconds: 3 }
];

test.beforeEach(async ({ page }) => {
  await page.route("**/api/projects/demo", (route) => route.fulfill({ json: project }));
  await page.route("**/api/projects/demo/artifact?**", (route) => {
    const url = new URL(route.request().url());
    const path = url.searchParams.get("path");
    route.fulfill({ json: { path, data: path?.endsWith("render_manifest.json") ? manifest : renderPlan } });
  });
  await page.route("**/api/projects/demo/thumbnail?**", (route) =>
    route.fulfill({ contentType: "image/svg+xml", body: `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="90"><rect width="160" height="90" fill="#556"/></svg>` })
  );
  await page.route("**/api/projects/demo/media?**", (route) => route.fulfill({ status: 204 }));
  await page.route("**/api/jobs**", (route) => route.fulfill({ json: [] }));
});

test("review timeline exposes zoom, seek, and constrained trim handles", async ({ page }) => {
  await page.goto("/projects/demo/review");
  await expect(page.getByText("Scene 1").first()).toBeVisible();

  const lanes = page.locator(".timeline-lanes");
  const before = await lanes.evaluate((node) => Number(getComputedStyle(node).getPropertyValue("--timeline-zoom")));
  const ruler = page.locator(".time-ruler-track");
  await ruler.scrollIntoViewIfNeeded();
  const box = await ruler.boundingBox();
  if (!box) throw new Error("missing time ruler");
  await ruler.hover({ position: { x: 40, y: 12 } });
  await page.mouse.down();
  await page.mouse.move(box.x + 220, box.y + 12);
  await page.mouse.up();
  const after = await lanes.evaluate((node) => Number(getComputedStyle(node).getPropertyValue("--timeline-zoom")));
  expect(after).toBeGreaterThan(before);

  await page.locator(".timeline-clip.final").first().click();
  await expect(page.locator(".timeline-edge-handle.left")).toBeVisible();
  await expect(page.locator(".timeline-edge-handle.right")).toBeVisible();

  const out = page.locator(".timeline-edge-handle.right");
  const outBox = await out.boundingBox();
  if (!outBox) throw new Error("missing OUT handle");
  await page.mouse.move(outBox.x + outBox.width / 2, outBox.y + outBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(outBox.x + 80, outBox.y + outBox.height / 2);
  await page.mouse.up();
  await expect(page.getByText(/Raw OUT preview/)).toBeVisible();
  await expect(page.locator(".status-badge.warning", { hasText: "stale" })).toBeVisible();
});
