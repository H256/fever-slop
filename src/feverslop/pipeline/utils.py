from pathlib import Path
import json

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

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    if whisper_raw is not None:
        raw_output_file = output_file.with_name(f"{output_file.stem}_whisper_raw.json")
        with raw_output_file.open("w", encoding="utf-8") as f:
            json.dump(whisper_raw, f, ensure_ascii=False, indent=2)

    return output_file
