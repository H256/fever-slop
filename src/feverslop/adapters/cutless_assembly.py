from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from feverslop.domain.cutless_boundaries import CutlessAssemblyPlan
from feverslop.domain.postprocessing import TrimSpec


class CutlessAssemblyService:
    """Assemble validated continuation clips using hard cuts only."""

    def __init__(self, postprocessor):
        self.postprocessor = postprocessor

    def assemble(
        self,
        segment_clips: dict[str, Path],
        plan: CutlessAssemblyPlan,
        output_file: str | Path,
        *,
        segment_frame_counts: dict[str, int],
        fps: int = 24,
    ) -> Path:
        output = Path(output_file)
        clips: list[Path] = []
        with TemporaryDirectory(prefix="cutless-") as temp_dir:
            for index, segment_id in enumerate(plan.segment_ids):
                source = Path(segment_clips[segment_id])
                trim_frames = 1 if segment_id in plan.trim_first_frame_segments else 0
                frame_count = int(segment_frame_counts[segment_id])
                if frame_count <= trim_frames:
                    raise ValueError(f"cutless segment {segment_id} has no frames after boundary trim")
                if trim_frames:
                    derived = Path(temp_dir) / f"segment-{index:04d}.mp4"
                    clips.append(self.postprocessor.trim_clip(TrimSpec(
                        source_file=source,
                        output_file=derived,
                        fps=fps,
                        trim_front_frames=trim_frames,
                        keep_frames=frame_count - trim_frames,
                        scene=index + 1,
                    )))
                else:
                    clips.append(source)

            concat_list = output.with_suffix(".cutless.concat.txt")
            self.postprocessor.write_concat_list(clips, concat_list)
            return self.postprocessor.concat_clips(
                concat_list,
                output,
                video_only=True,
                reencode=True,
                fps=fps,
                frame_count=sum(
                    int(segment_frame_counts[segment_id])
                    - (1 if segment_id in plan.trim_first_frame_segments else 0)
                    for segment_id in plan.segment_ids
                ),
            )
