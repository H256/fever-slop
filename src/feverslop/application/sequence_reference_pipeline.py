"""End-to-end anchor-to-reference-sheet orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any

from feverslop.application.orbitsheets_logic import select_orbitsheet_frames
from feverslop.application.sequence_to_sheet import (
    compose_contact_sheet,
    compose_sheet_from_contact_sheet,
    extract_video_frames,
    recommended_sheet_layout,
    recommended_view_count,
)
from feverslop.ports.rendering import ImageRenderRequest


@dataclass(frozen=True, slots=True)
class SequenceReferenceRequest:
    kind: str
    asset_id: str
    name: str
    description: str
    output_dir: Path
    image_prompt: str = ""
    seed: int = 0
    frames: int = 124


@dataclass(frozen=True, slots=True)
class SequenceReferenceResult:
    kind: str
    asset_id: str
    anchor_path: Path
    sequence_path: Path
    contact_sheet_path: Path
    sheet_path: Path
    selected_frames: int


class SequenceReferencePipeline:
    """Generate a reusable character/location reference from one anchor."""

    def __init__(self, *, anchor_backend: Any, sequence_backend: Any):
        self.anchor_backend = anchor_backend
        self.sequence_backend = sequence_backend

    def generate(self, request: SequenceReferenceRequest) -> SequenceReferenceResult:
        kind = request.kind.strip().lower()
        if kind not in {"character", "location"}:
            raise ValueError("sequence reference kind must be character or location")
        if not request.asset_id.strip():
            raise ValueError("sequence reference asset_id is required")
        view_count = recommended_view_count(kind)
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        final_dir = Path(request.output_dir) / ("actors" if kind == "character" else "locations") / request.asset_id
        staging_dir = Path(tempfile.mkdtemp(prefix=f"sequence-reference-{request.asset_id}-", dir=request.output_dir))
        try:
            anchor_dir = staging_dir / "anchor"
            anchor_dir.mkdir(parents=True)
            anchor = self.anchor_backend.render_image(
                ImageRenderRequest(
                    scene={"reference_id": request.asset_id, "kind": kind, "view": "anchor"},
                    scene_number=1,
                    prompt=request.image_prompt or request.description,
                    workflow_path=Path(""),
                    output_dir=anchor_dir,
                    width=1920,
                    height=1080,
                    reference_image=None,
                )
            )
            anchor = Path(anchor)
            if not anchor.is_file():
                raise FileNotFoundError(f"anchor backend did not create an image: {anchor}")

            prompt = self.sequence_backend.build_sheet_prompt(
                request.description,
                kind=kind,
                shots=view_count,
                frames=request.frames,
            )
            sequence = staging_dir / "sequence.mp4"
            rendered = self.sequence_backend.render(
                anchor_images=[anchor],
                prompt=prompt.prompt,
                output_path=sequence,
                seed=request.seed,
                frames=request.frames,
            )
            sequence = Path(rendered)
            if not sequence.is_file():
                raise FileNotFoundError(f"sequence backend did not create a video: {sequence}")

            frames_dir = staging_dir / "frames"
            candidates = extract_video_frames(
                sequence,
                frames_dir,
                sample_count=max(view_count * 4, view_count),
            )
            selected = select_orbitsheet_frames(
                candidates,
                count=view_count,
                subject=request.description,
            )
            contact_sheet = staging_dir / "contact-sheet.png"
            compose_contact_sheet(
                selected,
                contact_sheet,
                columns=3,
                panel_size=(512, 288),
                include_labels=False,
            )
            columns, panel_size = recommended_sheet_layout(kind)
            sheet = staging_dir / "sheet.png"
            compose_sheet_from_contact_sheet(
                contact_sheet,
                sheet,
                frame_count=len(selected),
                source_columns=3,
                columns=columns,
                panel_size=panel_size,
            )

            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.mkdir(parents=True)
            destinations = {
                "anchor.png": anchor,
                "sequence.mp4": sequence,
                "contact-sheet.png": contact_sheet,
                "sheet.png": sheet,
            }
            for name, source in destinations.items():
                shutil.copy2(source, final_dir / name)
            frame_destination = final_dir / "frames"
            frame_destination.mkdir()
            for index, source in enumerate(selected):
                shutil.copy2(source, frame_destination / f"frame_{index:04}.png")
            return SequenceReferenceResult(
                kind=kind,
                asset_id=request.asset_id,
                anchor_path=final_dir / "anchor.png",
                sequence_path=final_dir / "sequence.mp4",
                contact_sheet_path=final_dir / "contact-sheet.png",
                sheet_path=final_dir / "sheet.png",
                selected_frames=len(selected),
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
