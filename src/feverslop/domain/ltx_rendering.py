from __future__ import annotations

from feverslop.errors import FeverSlopValidationError

from dataclasses import dataclass


ROLLING_FRAME_PROFILES = {
    "original": (50, 25, True),
    "safe": (6, 0, False),
    "off": (0, 0, False),
}


def resolve_rolling_frame_profile(profile: str) -> tuple[int, int, bool]:
    try:
        return ROLLING_FRAME_PROFILES[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(ROLLING_FRAME_PROFILES))
        raise FeverSlopValidationError(
            f"rolling_frame_profile {profile!r} is invalid; expected one of: {supported}"
        ) from exc


@dataclass(frozen=True)
class AudioWindowSpec:
    scene_frame_count: int
    render_frame_count: int
    trim_front_frames: int
    tail_loss_frames: int
    fps: int
    audio_start_seconds: float
    audio_duration_seconds: float

    @property
    def output_duration_seconds(self) -> float:
        return self.scene_frame_count / float(self.fps)

    def as_dict(self) -> dict:
        return {
            "scene_frame_count": self.scene_frame_count,
            "render_frame_count": self.render_frame_count,
            "trim_front_frames": self.trim_front_frames,
            "tail_loss_frames": self.tail_loss_frames,
            "fps": self.fps,
            "audio_start_seconds": self.audio_start_seconds,
            "audio_duration_seconds": self.audio_duration_seconds,
            "output_duration_seconds": self.output_duration_seconds,
        }

    def __getitem__(self, key: str):
        return self.as_dict()[key]


@dataclass(frozen=True)
class PromptRelayPayload:
    global_prompt: str
    local_prompts: str
    segment_lengths: str


def round_up_8n1(frame_count: int) -> int:
    frame_count = max(1, int(frame_count))
    remainder = (frame_count - 1) % 8
    if remainder == 0:
        return frame_count
    return frame_count + (8 - remainder)


def round_down_8n1(frame_count: int) -> int:
    frame_count = max(1, int(frame_count))
    return max(1, ((frame_count - 1) // 8) * 8 + 1)


def build_audio_window_spec(
    *,
    scene_number: int,
    fps: int,
    scene_frame_count: int,
    scene_start_seconds: float,
    preroll_frames: int,
    tail_loss_frames: int,
    round_render_frames_to_8n1: bool = False,
) -> AudioWindowSpec:
    fps = int(fps)
    scene_frame_count = int(scene_frame_count)
    preroll = 0 if int(scene_number) == 1 else max(0, int(preroll_frames))
    tail = max(0, int(tail_loss_frames))

    audio_start = max(0.0, float(scene_start_seconds) - preroll / float(fps))
    effective_preroll = int(round((float(scene_start_seconds) - audio_start) * fps))

    base_render_frame_count = scene_frame_count + effective_preroll + tail
    render_frame_count = (
        round_up_8n1(base_render_frame_count)
        if round_render_frames_to_8n1
        else base_render_frame_count
    )
    effective_tail = tail + (render_frame_count - base_render_frame_count)
    audio_duration = round(max(0.0, (render_frame_count - 1) / float(fps)), 6)

    return AudioWindowSpec(
        scene_frame_count=scene_frame_count,
        render_frame_count=render_frame_count,
        trim_front_frames=effective_preroll,
        tail_loss_frames=effective_tail,
        fps=fps,
        audio_start_seconds=audio_start,
        audio_duration_seconds=audio_duration,
    )


class PromptRelayPayloadBuilder:
    min_prompt_relay_frames = 6

    def __init__(self, segment_length_mode: str = "frames_minus_one"):
        if segment_length_mode not in {"frames_minus_one", "frames"}:
            raise FeverSlopValidationError("segment_length_mode must be 'frames_minus_one' or 'frames'")
        self.segment_length_mode = segment_length_mode

    def build(
        self,
        *,
        scene: dict,
        render_frame_count: int,
        trim_front_frames: int,
        tail_loss_frames: int,
        preroll_prompt: str | None = None,
        tail_prompt: str | None = None,
    ) -> PromptRelayPayload:
        ltx = scene["ltx"]
        global_prompt = ltx["base_prompt"].strip()
        relays = ltx.get("prompt_relay", [])

        timeline_frames = (
            int(render_frame_count)
            if self.segment_length_mode == "frames"
            else max(1, int(render_frame_count) - 1)
        )
        scene_timeline_frames = (
            int(scene["frame_count"])
            if self.segment_length_mode == "frames"
            else max(1, int(scene["frame_count"]) - 1)
        )

        relay_segments: list[dict] = []

        if trim_front_frames > 0:
            relay_segments.append({
                "prompt": preroll_prompt or _default_preroll_prompt(scene),
                "length": int(trim_front_frames),
            })

        if not relays:
            relay_segments.append({
                "prompt": "continue the main scene motion with stable subject identity",
                "length": scene_timeline_frames,
            })
        else:
            relays = sorted(relays, key=lambda item: int(item["frame_start"]))
            cursor = 0

            for relay in relays:
                start = max(0, min(int(relay["frame_start"]), scene_timeline_frames))
                end = max(start, min(int(relay["frame_end"]), scene_timeline_frames))

                if start > cursor:
                    relay_segments.append({
                        "prompt": "the scene continues its established motion and atmosphere",
                        "length": start - cursor,
                    })
                    cursor = start

                length = max(1, end - start)
                relay_segments.append({
                    "prompt": str(relay["prompt"]).strip(),
                    "length": length,
                })
                cursor = end

            if cursor < scene_timeline_frames:
                relay_segments.append({
                    "prompt": "the scene continues its established motion and atmosphere",
                    "length": scene_timeline_frames - cursor,
                })

        if tail_loss_frames > 0:
            relay_segments.append({
                "prompt": tail_prompt or _default_tail_prompt(scene),
                "length": int(tail_loss_frames),
            })

        relay_segments = self.normalize_segments(relay_segments)
        local_prompts = [segment["prompt"] for segment in relay_segments]
        segment_lengths = [int(segment["length"]) for segment in relay_segments]

        total = sum(segment_lengths)
        if total != timeline_frames:
            raise FeverSlopValidationError(
                f"PromptRelay segment length mismatch for scene {scene.get('scene')}: "
                f"sum={total}, expected={timeline_frames}, mode={self.segment_length_mode}, "
                f"render_frame_count={render_frame_count}, scene_frame_count={scene.get('frame_count')}, "
                f"preroll={trim_front_frames}, tail={tail_loss_frames}"
            )

        return PromptRelayPayload(
            global_prompt=global_prompt,
            local_prompts="\n|".join(local_prompts),
            segment_lengths=",".join(str(int(x)) for x in segment_lengths),
        )

    @classmethod
    def normalize_segments(cls, segments: list[dict]) -> list[dict]:
        normalized = [
            {"prompt": str(segment["prompt"]).strip(), "length": int(segment["length"])}
            for segment in segments
            if int(segment["length"]) > 0
        ]

        while len(normalized) > 1:
            short_index = next(
                (
                    index
                    for index, segment in enumerate(normalized)
                    if int(segment["length"]) < cls.min_prompt_relay_frames
                ),
                None,
            )
            if short_index is None:
                break

            if short_index == 0:
                target_index = 1
            elif short_index == len(normalized) - 1:
                target_index = short_index - 1
            else:
                previous_length = int(normalized[short_index - 1]["length"])
                next_length = int(normalized[short_index + 1]["length"])
                target_index = short_index + 1 if next_length <= previous_length else short_index - 1

            normalized[target_index]["length"] = (
                int(normalized[target_index]["length"]) + int(normalized[short_index]["length"])
            )
            del normalized[short_index]

        return normalized


def _default_preroll_prompt(scene: dict) -> str:
    metadata = scene.get("metadata") or {}
    camera = str(metadata.get("camera_motion") or scene.get("camera") or "").strip()
    concept = str(metadata.get("base_concept") or scene.get("concept") or "").strip()
    base = "cinematic atmosphere holds"
    if camera:
        base = "atmospheric tension builds"
    return f"{base}; {concept or 'the scene settles into its opening frame'}; {camera or 'the camera holds a steady cinematic composition'} before the main action begins."


def _default_tail_prompt(scene: dict) -> str:
    metadata = scene.get("metadata") or {}
    camera = str(metadata.get("camera_motion") or scene.get("camera") or "").strip()
    concept = str(metadata.get("base_concept") or scene.get("concept") or "").strip()
    return f"The scene carries forward its last motion; {concept or 'the atmosphere persists'}; {camera or 'the camera continues its movement'}; the energy resolves without introducing a new scene."
