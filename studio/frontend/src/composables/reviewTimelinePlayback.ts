import type { TimelineItem } from "./reviewTimeline";

export function choosePlaybackItem(playableItems: TimelineItem[], scrubSeconds: number, selectedScene: number | null): TimelineItem | undefined {
  const atScrubber = playableItems.find((item) => scrubSeconds >= item.start && scrubSeconds < item.end);
  if (atScrubber) return atScrubber;
  const selected = playableItems.find((item) => item.scene === selectedScene);
  if (selected) return selected;
  return playableItems.find((item) => item.start >= scrubSeconds) ?? playableItems[0];
}

export function previewStart(item: TimelineItem): number {
  return item.finalClip ? item.start : item.rawStart;
}
