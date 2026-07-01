export interface ClipEdit {
  scene: number;
  rawInFrame: number;
  rawOutFrame: number;
  minRawInFrame: number;
  maxRawOutFrame: number;
}

export interface BoundaryTrimRequest {
  scene: number;
  edge: "left" | "right";
  deltaFrames: number;
}

export interface BuildEditInput {
  scene: number;
  frameCount: number;
  trimFrontFrames: number;
  tailFrames: number;
}

export function buildEditState(input: BuildEditInput): ClipEdit {
  return {
    scene: input.scene,
    rawInFrame: input.trimFrontFrames,
    rawOutFrame: input.trimFrontFrames + input.frameCount,
    minRawInFrame: 0,
    maxRawOutFrame: input.trimFrontFrames + input.frameCount + input.tailFrames
  };
}

export function applyBoundaryTrim(clips: ClipEdit[], request: BoundaryTrimRequest): ClipEdit[] {
  const next = clips.map((clip) => ({ ...clip }));
  const index = next.findIndex((clip) => clip.scene === request.scene);
  if (index < 0 || request.deltaFrames === 0) return next;
  if (request.edge === "right") trimRight(next, index, request.deltaFrames);
  else trimLeft(next, index, request.deltaFrames);
  return next;
}

function trimRight(clips: ClipEdit[], index: number, deltaFrames: number) {
  const clip = clips[index];
  const neighbor = clips[index + 1];
  if (!neighbor) return;
  const applied = clampDelta(deltaFrames, clip.maxRawOutFrame - clip.rawOutFrame, neighbor.rawOutFrame - neighbor.rawInFrame - 1, neighbor.rawInFrame - neighbor.minRawInFrame);
  clip.rawOutFrame += applied;
  neighbor.rawInFrame += applied;
}

function trimLeft(clips: ClipEdit[], index: number, deltaFrames: number) {
  const clip = clips[index];
  const neighbor = clips[index - 1];
  if (!neighbor) return;
  const applied = clampDelta(-deltaFrames, clip.rawInFrame - clip.minRawInFrame, neighbor.rawOutFrame - neighbor.rawInFrame - 1, neighbor.maxRawOutFrame - neighbor.rawOutFrame);
  clip.rawInFrame -= applied;
  neighbor.rawOutFrame -= applied;
}

function clampDelta(deltaFrames: number, growCapacity: number, shrinkCapacity: number, reverseCapacity: number): number {
  if (deltaFrames > 0) return Math.max(0, Math.min(deltaFrames, growCapacity, shrinkCapacity));
  return -Math.max(0, Math.min(-deltaFrames, reverseCapacity, shrinkCapacity));
}
