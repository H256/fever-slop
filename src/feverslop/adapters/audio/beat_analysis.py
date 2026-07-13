from __future__ import annotations

from pathlib import Path
import json
import random

import librosa
import numpy as np


class BeatImpactAnalyzer:
    def __init__(self, sample_rate: int | None = None):
        self.sample_rate = sample_rate

    def _load_mono(self, path: str | Path | None):
        if not path:
            return None, None
        y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return y, sr

    def _stem_usable(self, y_stem, y_ref, sr) -> bool:
        if y_stem is None or y_ref is None:
            return False

        if (len(y_ref) - len(y_stem)) / sr > 1.0:
            return False

        rms = librosa.feature.rms(y=y_stem, frame_length=2048, hop_length=512)[0]
        if rms.size == 0:
            return False

        overall = float(np.median(rms))
        tail_frames = max(1, int(10.0 * sr / 512))
        tail = float(np.median(rms[-tail_frames:]))

        if overall <= 1e-8:
            return False

        return tail >= overall * 0.1

    def _track_beats(self, y, sr):
        tempo, frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        times = librosa.frames_to_time(frames, sr=sr)
        return tempo, times

    def _onset_strength(self, y, sr):
        if y is None:
            return None
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        return onset / (np.max(onset) + 1e-6)

    def analyze(
        self,
        final_mix_path: str | Path,
        drums_path: str | Path | None = None,
        bass_path: str | Path | None = None,
        vocals_path: str | Path | None = None,
        other_path: str | Path | None = None,
    ) -> dict:
        y_mix, sr = self._load_mono(final_mix_path)
        if y_mix is None:
            raise ValueError("final_mix_path is invalid")

        y_drums, _ = self._load_mono(drums_path)
        y_bass, _ = self._load_mono(bass_path)
        y_vocals, _ = self._load_mono(vocals_path)
        y_other, _ = self._load_mono(other_path)

        mix_duration = float(len(y_mix) / sr)

        tempo_mix, beat_times_mix = self._track_beats(y_mix, sr)
        tempo = tempo_mix
        beat_times = beat_times_mix
        source_used = "final_mix"

        if self._stem_usable(y_drums, y_mix, sr):
            tempo, beat_times = self._track_beats(y_drums, sr)
            source_used = "drums"
        elif self._stem_usable(y_other, y_mix, sr):
            tempo, beat_times = self._track_beats(y_other, sr)
            source_used = "other"

        onset_mix = self._onset_strength(y_mix, sr)
        onset_drums = self._onset_strength(y_drums, sr)
        onset_bass = self._onset_strength(y_bass, sr)
        onset_vocals = self._onset_strength(y_vocals, sr)
        onset_other = self._onset_strength(y_other, sr)

        onset_times = (
            librosa.frames_to_time(np.arange(len(onset_mix)), sr=sr)
            if onset_mix is not None and len(onset_mix) > 0
            else np.array([], dtype=np.float32)
        )

        def safe_value(arr, idx):
            if arr is None or idx is None:
                return None
            if idx < 0 or idx >= len(arr):
                return None
            return float(arr[idx])

        beats = []

        for i, t in enumerate(beat_times):
            idx = None if onset_times.size == 0 else int(np.argmin(np.abs(onset_times - t)))

            impact = 0.0
            weight_sum = 0.0

            for arr, weight in [
                (onset_drums, 0.45),
                (onset_bass, 0.25),
                (onset_vocals, 0.15),
                (onset_other, 0.15),
            ]:
                val = safe_value(arr, idx)
                if val is not None:
                    impact += val * weight
                    weight_sum += weight

            if weight_sum > 0:
                impact /= weight_sum
            else:
                val = safe_value(onset_mix, idx)
                impact = val if val is not None else 0.0

            beats.append({
                "time": round(float(t), 4),
                "beat_index": i,
                "downbeat": i % 4 == 0,
                "impact": round(float(impact), 4),
            })

        tempo_arr = np.asarray(tempo).reshape(-1)
        tempo_value = float(tempo_arr[0]) if tempo_arr.size else float(tempo)

        return {
            "bpm": round(tempo_value, 2),
            "source_used_for_beats": source_used,
            "duration": mix_duration,
            "beats": beats,
        }

    def analyze_to_json_file(
        self,
        final_mix_path: str | Path,
        output_json_path: str | Path,
        drums_path: str | Path | None = None,
        bass_path: str | Path | None = None,
        vocals_path: str | Path | None = None,
        other_path: str | Path | None = None,
    ) -> Path:
        data = self.analyze(
            final_mix_path=final_mix_path,
            drums_path=drums_path,
            bass_path=bass_path,
            vocals_path=vocals_path,
            other_path=other_path,
        )

        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_json_path


class BeatSceneDurationGenerator:
    def __init__(
        self,
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        bias: float = 0.7,
        duration_preset: str = "impact_weighted",
        seed: int = 0,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.bias = bias
        self.duration_preset = duration_preset
        self.seed = seed

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def generate(self, beat_data: dict) -> str:
        beats = beat_data["beats"]
        if not beats:
            raise ValueError("beat_data['beats'] is empty")

        song_end = float(beat_data.get("duration", beats[-1]["time"]))
        rng = random.Random(self.seed)

        srt_lines = []
        current_time = 0.0
        scene_index = 1
        current_index = 0
        prev_duration = None

        first_beat = float(beats[0]["time"])

        if first_beat > 1e-6:
            srt_lines += [
                str(scene_index),
                f"{self._format_time(0.0)} --> {self._format_time(first_beat)}",
                f"SCENE {scene_index}",
                "",
            ]
            scene_index += 1
            current_time = first_beat

        while current_index < len(beats) - 1:
            start_time = float(beats[current_index]["time"])
            min_time = start_time + self.min_duration
            max_time = start_time + self.max_duration

            candidates = []

            for i in range(current_index + 1, len(beats)):
                t = float(beats[i]["time"])
                if t < min_time:
                    continue
                if t > max_time:
                    break

                impact = float(beats[i]["impact"])
                downbeat = bool(beats[i]["downbeat"])
                base_weight = impact * (1.2 if downbeat else 1.0)
                duration = t - start_time
                candidates.append((i, t, base_weight, duration))

            if not candidates:
                forced_end = min(max_time, song_end)
                duration = forced_end - start_time

                if duration <= 0:
                    break

                srt_lines += [
                    str(scene_index),
                    f"{self._format_time(current_time)} --> {self._format_time(current_time + duration)}",
                    f"SCENE {scene_index}",
                    "",
                ]

                current_time += duration
                scene_index += 1
                prev_duration = duration

                next_index = current_index + 1
                while next_index < len(beats) and float(beats[next_index]["time"]) <= forced_end:
                    next_index += 1

                if next_index >= len(beats):
                    break

                current_index = next_index
                continue

            filtered = candidates

            if prev_duration is not None:
                non_repeat = [c for c in candidates if abs(c[3] - prev_duration) >= 0.20]
                if non_repeat:
                    filtered = non_repeat

            weights = []

            for _, _, base_weight, candidate_duration in filtered:
                w = (base_weight ** self.bias) + 1e-6

                if prev_duration is not None:
                    delta = abs(candidate_duration - prev_duration)

                    if self.duration_preset == "varied_no_repeat":
                        w *= 0.6 + min(2.0, delta / 0.8)
                        mid = (self.min_duration + self.max_duration) * 0.5
                        switched_band = (
                            (prev_duration >= mid and candidate_duration < mid)
                            or (prev_duration < mid and candidate_duration >= mid)
                        )
                        w *= 1.20 if switched_band else 0.85

                    elif self.duration_preset == "clustered_no_repeat":
                        w *= 1.30 if delta <= 1.5 else 0.75

                weights.append(max(w, 1e-9))

            chosen_index, chosen_time, _, _ = rng.choices(filtered, weights=weights, k=1)[0]
            duration = chosen_time - start_time

            srt_lines += [
                str(scene_index),
                f"{self._format_time(current_time)} --> {self._format_time(current_time + duration)}",
                f"SCENE {scene_index}",
                "",
            ]

            current_time += duration
            scene_index += 1
            current_index = chosen_index
            prev_duration = duration

        while current_time < song_end:
            end_time = min(current_time + self.max_duration, song_end)

            srt_lines += [
                str(scene_index),
                f"{self._format_time(current_time)} --> {self._format_time(end_time)}",
                f"SCENE {scene_index}",
                "",
            ]

            current_time = end_time
            scene_index += 1

        return "\n".join(srt_lines)

    def generate_from_json_file(
        self,
        beat_json_path: str | Path,
        output_srt_path: str | Path,
    ) -> Path:
        beat_json_path = Path(beat_json_path)
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)

        with beat_json_path.open("r", encoding="utf-8") as f:
            beat_data = json.load(f)

        srt_text = self.generate(beat_data)

        with output_srt_path.open("w", encoding="utf-8") as f:
            f.write(srt_text)

        return output_srt_path