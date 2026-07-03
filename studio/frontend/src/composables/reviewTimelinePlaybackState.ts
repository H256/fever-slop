import { nextTick, type ComputedRef, type Ref } from "vue";
import type { TimelineItem } from "./reviewTimeline";
import type { RawPreview } from "./reviewTimelineMedia";
import { choosePlaybackItem, previewStart } from "./reviewTimelinePlayback";

export interface TimelineMediaElement {
  currentTime: number;
  play(): Promise<void>;
  pause(): void;
}

export interface UseReviewTimelinePlaybackInput {
  timelineItems: ComputedRef<TimelineItem[]>;
  playableItems: ComputedRef<TimelineItem[]>;
  selectedItem: ComputedRef<TimelineItem | undefined>;
  rawPreview: Ref<RawPreview | null>;
  selectedScene: Ref<number | null>;
  scrubSeconds: Ref<number>;
  playingTimeline: Ref<boolean>;
  videoRef: Ref<TimelineMediaElement | null>;
  audioRef: Ref<TimelineMediaElement | null>;
}

export function useReviewTimelinePlayback(input: UseReviewTimelinePlaybackInput) {
  function selectItem(item: TimelineItem) {
    input.rawPreview.value = null;
    input.selectedScene.value = item.scene;
    input.scrubSeconds.value = item.start;
    void seekPreview();
  }

  async function scrub() {
    input.playingTimeline.value = false;
    const item = input.timelineItems.value.find((candidate) => input.scrubSeconds.value >= candidate.start && input.scrubSeconds.value <= candidate.end);
    if (item) input.selectedScene.value = item.scene;
    if (input.audioRef.value) input.audioRef.value.currentTime = input.scrubSeconds.value;
    await seekPreview();
  }

  async function seekPreview() {
    await nextTick();
    const video = input.videoRef.value;
    const item = input.selectedItem.value;
    if (!video || !item?.clip) return;
    video.currentTime = input.rawPreview.value ? input.rawPreview.value.seconds : Math.max(0, input.scrubSeconds.value - previewStart(item));
  }

  async function playTimeline() {
    const item = choosePlaybackItem(input.playableItems.value, input.scrubSeconds.value, input.selectedScene.value);
    if (!item) return;
    const startSeconds = Math.max(item.start, Math.min(input.scrubSeconds.value, item.end));
    input.rawPreview.value = null;
    input.playingTimeline.value = true;
    input.selectedScene.value = item.scene;
    input.scrubSeconds.value = startSeconds;
    await nextTick();
    if (input.videoRef.value) input.videoRef.value.currentTime = Math.max(0, startSeconds - previewStart(item));
    if (input.audioRef.value) {
      input.audioRef.value.currentTime = startSeconds;
      await input.audioRef.value.play();
    }
    await input.videoRef.value?.play();
  }

  function stopTimeline() {
    input.rawPreview.value = null;
    input.playingTimeline.value = false;
    input.videoRef.value?.pause();
    input.audioRef.value?.pause();
  }

  async function playNextClip() {
    const selected = input.selectedItem.value;
    if (!input.playingTimeline.value || !selected) return;
    const index = input.playableItems.value.findIndex((item) => item.scene === selected.scene);
    const next = input.playableItems.value[index + 1];
    if (!next) {
      input.playingTimeline.value = false;
      input.audioRef.value?.pause();
      return;
    }
    selectItem(next);
    await nextTick();
    await input.videoRef.value?.play();
  }

  function syncScrubber() {
    if (!input.selectedItem.value || !input.videoRef.value) return;
    if (input.rawPreview.value) return;
    input.scrubSeconds.value = previewStart(input.selectedItem.value) + input.videoRef.value.currentTime;
    if (input.audioRef.value && Math.abs(input.audioRef.value.currentTime - input.scrubSeconds.value) > 0.25) {
      input.audioRef.value.currentTime = input.scrubSeconds.value;
    }
  }

  async function playAudio() {
    if (input.rawPreview.value) return;
    if (!input.audioRef.value) return;
    input.audioRef.value.currentTime = input.scrubSeconds.value;
    await input.audioRef.value.play();
  }

  function pauseAudio() {
    if (!input.playingTimeline.value) input.audioRef.value?.pause();
  }

  return {
    pauseAudio,
    playAudio,
    playNextClip,
    playTimeline,
    scrub,
    seekPreview,
    selectItem,
    stopTimeline,
    syncScrubber
  };
}
