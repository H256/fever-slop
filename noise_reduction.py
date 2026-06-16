from pathlib import Path
from pydub import AudioSegment


def estimate_dynamic_silence_threshold(
    audio_file: str | Path,
    frame_ms: int = 100,
    noise_percentile: float = 20.0,
    margin_db: float = 6.0,
    min_threshold_db: float = -50.0,
    max_threshold_db: float = -25.0,
) -> float:
    audio = AudioSegment.from_file(audio_file)

    values = []

    for pos in range(0, len(audio), frame_ms):
        chunk = audio[pos:pos + frame_ms]
        if chunk.dBFS != float("-inf"):
            values.append(chunk.dBFS)

    if not values:
        return -40.0

    values.sort()

    idx = int(len(values) * (noise_percentile / 100.0))
    idx = max(0, min(idx, len(values) - 1))

    noise_floor = values[idx]
    threshold = noise_floor + margin_db

    threshold = max(min_threshold_db, min(max_threshold_db, threshold))

    return threshold

def estimate_adaptive_threshold(
    audio_file: str | Path,
    frame_ms: int = 100,
    low_percentile: float = 20.0,
    high_percentile: float = 85.0,
    ratio: float = 0.35,
) -> float:
    audio = AudioSegment.from_file(audio_file)

    values = []

    for pos in range(0, len(audio), frame_ms):
        chunk = audio[pos:pos + frame_ms]
        if chunk.dBFS != float("-inf"):
            values.append(chunk.dBFS)

    if not values:
        return -40.0

    values.sort()

    low = values[int(len(values) * low_percentile / 100)]
    high = values[int(len(values) * high_percentile / 100)]

    # dBFS ist negativ: low z.B. -48, high z.B. -18
    threshold = low + ((high - low) * ratio)

    return max(-50.0, min(-25.0, threshold))