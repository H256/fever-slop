import { describe, expect, test } from "bun:test";
import { computed, ref } from "vue";
import { useReviewTimelinePlayback, type TimelineMediaElement } from "./reviewTimelinePlaybackState";
import type { TimelineItem } from "./reviewTimeline";
import type { RawPreview } from "./reviewTimelineMedia";

describe("useReviewTimelinePlayback", () => {
  test("selects items and seeks preview relative to timeline start", async () => {
    const state = setup();

    state.playback.selectItem(state.items.value[1]);
    await state.playback.seekPreview();

    expect(state.selectedScene.value).toBe(2);
    expect(state.scrubSeconds.value).toBe(5);
    expect(state.video.currentTime).toBe(0);
  });

  test("scrubs to the item under the playhead and syncs audio", async () => {
    const state = setup();
    state.scrubSeconds.value = 6;

    await state.playback.scrub();

    expect(state.playingTimeline.value).toBe(false);
    expect(state.selectedScene.value).toBe(2);
    expect(state.audio.currentTime).toBe(6);
  });

  test("starts and stops timeline playback with audio and video elements", async () => {
    const state = setup();
    state.selectedScene.value = 2;
    state.scrubSeconds.value = 6;

    await state.playback.playTimeline();
    expect(state.playingTimeline.value).toBe(true);
    expect(state.video.currentTime).toBe(1);
    expect(state.audio.currentTime).toBe(6);
    expect(state.video.playCount).toBe(1);
    expect(state.audio.playCount).toBe(1);

    state.playback.stopTimeline();
    expect(state.playingTimeline.value).toBe(false);
    expect(state.video.pauseCount).toBe(1);
    expect(state.audio.pauseCount).toBe(1);
  });

  test("advances to the next clip or stops at the end", async () => {
    const state = setup();
    state.playingTimeline.value = true;
    state.selectedScene.value = 1;

    await state.playback.playNextClip();
    expect(state.selectedScene.value).toBe(2);
    expect(state.video.playCount).toBe(1);

    await state.playback.playNextClip();
    expect(state.playingTimeline.value).toBe(false);
    expect(state.audio.pauseCount).toBe(1);
  });
});

function setup() {
  const items = ref<TimelineItem[]>([item({ scene: 1, start: 0, end: 5 }), item({ scene: 2, start: 5, end: 10 })]);
  const selectedScene = ref<number | null>(1);
  const scrubSeconds = ref(0);
  const playingTimeline = ref(false);
  const rawPreview = ref<RawPreview | null>(null);
  const video = mediaElement();
  const audio = mediaElement();
  const playback = useReviewTimelinePlayback({
    timelineItems: computed(() => items.value),
    playableItems: computed(() => items.value.filter((item) => item.clip)),
    selectedItem: computed(() => items.value.find((item) => item.scene === selectedScene.value) ?? items.value[0]),
    rawPreview,
    selectedScene,
    scrubSeconds,
    playingTimeline,
    videoRef: ref(video),
    audioRef: ref(audio)
  });

  return { audio, items, playback, playingTimeline, scrubSeconds, selectedScene, video };
}

function mediaElement(): TimelineMediaElement & { playCount: number; pauseCount: number } {
  return {
    currentTime: 0,
    playCount: 0,
    pauseCount: 0,
    async play() {
      this.playCount += 1;
    },
    pause() {
      this.pauseCount += 1;
    }
  };
}

function item(overrides: Partial<TimelineItem>): TimelineItem {
  return {
    scene: 0,
    start: 0,
    end: 0,
    duration: 0,
    rawStart: 0,
    rawEnd: 0,
    rawDuration: 0,
    finalClip: "final.mp4",
    rawClip: "raw.mp4",
    clip: "final.mp4",
    status: "final",
    preview: "",
    hasManifestTiming: false,
    ...overrides
  };
}
