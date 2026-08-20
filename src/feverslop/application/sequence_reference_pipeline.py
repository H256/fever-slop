"""End-to-end anchor-to-reference-sheet orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable
import uuid

from feverslop.application.reference_sheet_planning import (
    DeterministicReferenceSheetPlanner,
    compile_reference_sheet_plan,
)
from feverslop.application.orbitsheets_logic import select_orbitsheet_frames
from feverslop.application.sequence_to_sheet import (
    compose_contact_sheet,
    compose_sheet_from_contact_sheet,
    extract_video_frames,
    recommended_sheet_layout,
    recommended_view_count,
)
from feverslop.ports.rendering import ImageRenderRequest

# LLM-generated ids must remain single, opaque path components.
_ASSET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class SequenceReferenceRequest:
    kind: str
    asset_id: str
    name: str
    description: str
    output_dir: Path
    image_prompt: str = ""
    visual_style: str = ""
    seed: int = 0
    frames: int = 124
    asset_context: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SequenceReferenceResult:
    kind: str
    asset_id: str
    anchor_path: Path
    sequence_path: Path
    contact_sheet_path: Path
    sheet_path: Path
    selected_frames: int
    planning_profile: str = "reference-sheet-from-sequence/v1"
    prompt_revision: str = "reference-sheet-compiler/v1"
    planner_source: str = "deterministic"
    fallback_reason: str | None = None
    semantic_plan_hash: str = ""
    prompt_hash: str = ""
    workflow_profile: str = ""
    seed: int = 0
    frames: int = 124
    anchor_prompt: str = ""


class SequenceReferencePipeline:
    """Generate a reusable character/location reference from one anchor."""

    def __init__(self, *, anchor_backend: Any, sequence_backend: Any, planner: Any | None = None, on_phase: Callable[[dict[str, Any]], None] | None = None):
        self.anchor_backend = anchor_backend
        self.sequence_backend = sequence_backend
        self.planner = planner or DeterministicReferenceSheetPlanner()
        self.on_phase = on_phase

    def generate(self, request: SequenceReferenceRequest) -> SequenceReferenceResult:
        kind = request.kind.strip().lower()
        if kind not in {"character", "location"}:
            raise ValueError("sequence reference kind must be character or location")
        if not request.asset_id.strip():
            raise ValueError("sequence reference asset_id is required")
        if not _ASSET_ID_PATTERN.fullmatch(request.asset_id):
            raise ValueError(f"sequence reference asset_id is not a safe path component: {request.asset_id!r}")
        view_count = recommended_view_count(kind)
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        final_dir = Path(request.output_dir) / ("actors" if kind == "character" else "locations") / request.asset_id
        staging_dir = Path(tempfile.mkdtemp(prefix=f"sequence-reference-{request.asset_id}-", dir=request.output_dir))
        try:
            semantic_plan = self.planner.plan(
                kind=kind,
                description=request.description,
                asset_context=dict(request.asset_context or {}),
            )
            compiled_plan = compile_reference_sheet_plan(
                semantic_plan,
                kind=kind,
                description=request.description,
                frames=request.frames,
            )
            anchor_dir = staging_dir / "anchor"
            anchor_dir.mkdir(parents=True)
            style = " ".join(str(request.visual_style or "").split())
            style_suffix = f" Visual style: {style}." if style else ""
            if kind == "character":
                anchor_prompt = (
                    f"{compiled_plan.anchor_description}. One character only, neutral relaxed pose, "
                    "plain seamless studio backdrop. Prioritize face, hair, body proportions, "
                    "wardrobe, materials, and colors. No performance action, no instrument, "
                    f"no handheld prop, no scene location.{style_suffix}"
                )
            else:
                anchor_prompt = f"{request.image_prompt or compiled_plan.anchor_description}{style_suffix}"
            self._report_phase(request, "anchor_start")
            anchor = self.anchor_backend.render_image(
                ImageRenderRequest(
                    scene={"reference_id": request.asset_id, "kind": kind, "view": "anchor"},
                    scene_number=1,
                    prompt=anchor_prompt,
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
            self._report_phase(request, "anchor_complete", path=anchor)

            if hasattr(self.sequence_backend, "build_sheet_prompt_from_plan"):
                prompt = self.sequence_backend.build_sheet_prompt_from_plan(compiled_plan)
            else:
                prompt = self.sequence_backend.build_sheet_prompt(
                    request.description,
                    kind=kind,
                    shots=view_count,
                    frames=request.frames,
                )
            sequence_prompt = prompt.prompt
            if style:
                sequence_prompt = f"{sequence_prompt}\n\nVisual style: {style}. Preserve this style throughout the sequence."
            sequence = staging_dir / "sequence.mp4"
            self._report_phase(request, "sequence_start")
            rendered = self.sequence_backend.render(
                anchor_images=[anchor],
                prompt=sequence_prompt,
                output_path=sequence,
                seed=request.seed,
                frames=request.frames,
            )
            sequence = Path(rendered)
            if not sequence.is_file():
                raise FileNotFoundError(f"sequence backend did not create a video: {sequence}")
            self._report_phase(request, "sequence_complete", path=sequence)
            semantic_plan_hash = hashlib.sha256(
                json.dumps(semantic_plan.model_dump() if hasattr(semantic_plan, "model_dump") else semantic_plan, sort_keys=True).encode()
            ).hexdigest()
            prompt_hash = hashlib.sha256(sequence_prompt.encode()).hexdigest()

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
            out_staging = Path(
                tempfile.mkdtemp(
                    prefix=f".sequence-reference-{request.asset_id}-swap-",
                    dir=final_dir.parent,
                )
            )
            backup_dir: Path | None = None
            try:
                destinations = {
                    "anchor.png": anchor,
                    "sequence.mp4": sequence,
                    "contact-sheet.png": contact_sheet,
                    "sheet.png": sheet,
                }
                for name, source in destinations.items():
                    shutil.copy2(source, out_staging / name)
                frame_destination = out_staging / "frames"
                frame_destination.mkdir()
                for index, source in enumerate(selected):
                    shutil.copy2(source, frame_destination / f"frame_{index:04}.png")
                if not all((out_staging / name).is_file() for name in destinations):
                    raise OSError("staged reference artifacts are incomplete")
                if len(list(frame_destination.iterdir())) != len(selected):
                    raise OSError("staged reference frames are incomplete")
                if final_dir.exists():
                    backup_dir = final_dir.with_name(f".{final_dir.name}.previous-{uuid.uuid4().hex}")
                    final_dir.replace(backup_dir)
                out_staging.replace(final_dir)
            except BaseException:
                if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
                    backup_dir.replace(final_dir)
                raise
            finally:
                shutil.rmtree(out_staging, ignore_errors=True)
                if backup_dir is not None:
                    shutil.rmtree(backup_dir, ignore_errors=True)
            return SequenceReferenceResult(
                kind=kind,
                asset_id=request.asset_id,
                anchor_path=final_dir / "anchor.png",
                sequence_path=final_dir / "sequence.mp4",
                contact_sheet_path=final_dir / "contact-sheet.png",
                sheet_path=final_dir / "sheet.png",
                selected_frames=len(selected),
                planning_profile="reference-sheet-from-sequence/v1",
                prompt_revision="reference-sheet-compiler/v1",
                planner_source=getattr(self.planner, "source", "deterministic"),
                fallback_reason=getattr(self.planner, "fallback_reason", None),
                semantic_plan_hash=semantic_plan_hash,
                prompt_hash=prompt_hash,
                workflow_profile=str(getattr(getattr(self.sequence_backend, "profile", None), "workflow_filename", "")),
                seed=request.seed,
                frames=request.frames,
                anchor_prompt=anchor_prompt,
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _report_phase(self, request: SequenceReferenceRequest, phase: str, **extra: Any) -> None:
        if self.on_phase is not None:
            self.on_phase({"kind": request.kind, "id": request.asset_id, "name": request.name, "phase": phase, **extra})
