from pathlib import Path
import json

def save_timeline_json(timeline, output_file: str | Path):
    output_file = Path(output_file)

    data = [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "type": seg.kind,
            **({"lyrics": seg.text} if seg.text else {}),
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

    return output_file
