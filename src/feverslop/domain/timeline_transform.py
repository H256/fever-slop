from __future__ import annotations

from feverslop.domain.timeline import TimelineSegment


def normalize_empty_vocals(
    timeline: list[TimelineSegment],
    min_text_chars: int = 3,
) -> list[TimelineSegment]:
    result = []
    for seg in timeline:
        if seg.kind == "vocals" and len(seg.text.strip()) < min_text_chars:
            result.append(
                TimelineSegment(
                    start=seg.start,
                    end=seg.end,
                    kind="instrumental",
                    text="",
                    word_timestamps=(),
                )
            )
        else:
            result.append(seg)
    return result


def merge_same_kind_segments(
    timeline: list[TimelineSegment],
    merge_gap: float = 0.5,
) -> list[TimelineSegment]:
    if not timeline:
        return []

    merged: list[TimelineSegment] = []
    current = timeline[0]

    for seg in timeline[1:]:
        same_kind = current.kind == seg.kind
        close_enough = seg.start - current.end <= merge_gap

        if same_kind and close_enough:
            new_end = max(current.end, seg.end)
            new_text = current.text
            new_word_timestamps = current.word_timestamps + seg.word_timestamps
            if seg.text.strip():
                if new_text:
                    new_text = (new_text + " " + seg.text).strip()
                else:
                    new_text = seg.text.strip()
            current = TimelineSegment(
                start=current.start,
                end=new_end,
                kind=current.kind,
                text=new_text,
                word_timestamps=new_word_timestamps,
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    return merged
