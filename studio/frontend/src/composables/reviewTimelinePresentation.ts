import type { TimelineItem } from "./reviewTimeline";

export interface ThumbnailRequest {
  path: string;
  times: number[];
}

export function formatTime(value: number): string {
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function timelineTicks(totalDuration: number): number[] {
  const total = totalDuration || 0;
  const step = total > 240 ? 30 : total > 90 ? 15 : 5;
  const ticks: number[] = [];
  for (let value = 0; value <= total; value += step) ticks.push(value);
  if (!ticks.includes(total)) ticks.push(total);
  return ticks;
}

export function blockStyle(start: number, duration: number, totalDuration: number): Record<string, string> {
  const total = totalDuration || 1;
  return {
    left: `${(start / total) * 100}%`,
    width: `${Math.max((duration / total) * 100, 0)}%`
  };
}

export function thumbnailFrameTimes(duration: number, zoom: number): number[] {
  const count = Math.max(1, Math.min(8, Math.floor((duration * zoom) / 4) + 1));
  return Array.from({ length: count }, (_, index) =>
    count === 1 ? Math.max(0, duration * 0.2) : Math.max(0, (duration * index) / (count - 1))
  );
}

export function buildThumbnailRequests(items: TimelineItem[], zoom: number): ThumbnailRequest[] {
  const requests = new Map<string, Set<number>>();
  for (const item of items) {
    for (const path of [item.rawClip, item.finalClip].filter(Boolean)) {
      const times = requests.get(path) ?? new Set<number>();
      thumbnailFrameTimes(path === item.rawClip ? item.rawDuration : item.duration, zoom).forEach((frame) => times.add(frame));
      requests.set(path, times);
    }
  }
  return [...requests].map(([path, times]) => ({ path, times: [...times] }));
}
