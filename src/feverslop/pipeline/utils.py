from pathlib import Path

from feverslop.utils.io import atomic_write_json


def save_timeline_json(timeline, output_file: str | Path, *, whisper_raw=None):
    output_file = Path(output_file)

    data = [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "type": seg.kind,
            **({"lyrics": seg.text} if seg.text else {}),
            **({"word_timestamps": list(seg.word_timestamps)} if seg.word_timestamps else {}),
        }
        for seg in timeline
    ]

    atomic_write_json(output_file, data)

    if whisper_raw is not None:
        raw_output_file = output_file.with_name(f"{output_file.stem}_whisper_raw.json")
        atomic_write_json(raw_output_file, whisper_raw)

    return output_file
