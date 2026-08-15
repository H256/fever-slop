"""ComfyUI adapters for backend-neutral sequence-to-sheet rendering."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.sequence_to_sheet_workflow import (
    MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE,
    SequenceToSheetWorkflowProfile,
)
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.orbitsheets_prompts import (
    H3SheetPrompt,
    build_h3_character_prompt,
    build_h3_location_prompt,
)


class ComfyUISequenceToSheetBackend:
    """Patch and render one sequence-to-sheet workflow.

    The adapter deliberately exposes only semantic inputs. Backend-specific
    node IDs remain inside the workflow files and are addressed by meta-title.
    """

    def __init__(
        self,
        *,
        client: Any,
        workflow_path: str | Path,
        backend: str,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        model_resolver: Any | None = None,
    ) -> None:
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.profile = self._profile_for(backend)
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.model_resolver = model_resolver

    @staticmethod
    def _profile_for(backend: str) -> SequenceToSheetWorkflowProfile:
        normalized = backend.strip().lower()
        if normalized in {"minimax", "minimax_h3"}:
            return MINIMAX_H3_SEQUENCE_TO_SHEET_PROFILE
        raise ValueError(f"unsupported sequence-to-sheet backend: {backend}")

    def load_workflow(self) -> dict:
        workflow = json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))
        errors = self.profile.validate(workflow)
        if errors:
            raise ValueError(f"invalid sequence-to-sheet workflow: {', '.join(errors)}")
        return workflow

    @staticmethod
    def build_sheet_prompt(
        description: str,
        *,
        kind: str,
        shots: int = 5,
        frames: int = 124,
        framing: str = "full body, generous margin",
        coverage: str = "cut views",
        rotation: str = "auto",
    ) -> H3SheetPrompt:
        """Build the backend-neutral multi-view prompt used by H3 workflows."""
        normalized = kind.strip().lower()
        if normalized == "character":
            return build_h3_character_prompt(
                description,
                shots=shots,
                frames=frames,
                framing=framing,
            )
        if normalized == "location":
            return build_h3_location_prompt(
                description,
                shots=shots,
                frames=frames,
                coverage=coverage,
                rotation=rotation,
            )
        raise ValueError(f"unsupported sheet kind: {kind}")

    def build_workflow(
        self,
        *,
        anchor_images: list[str | Path],
        prompt: str,
        seed: int,
        width: int = 768,
        height: int = 512,
        frames: int = 97,
        output_prefix: str = "sequence_to_sheet/output",
    ) -> dict:
        if not anchor_images:
            raise ValueError("at least one anchor image is required")
        patcher = WorkflowPatcher(deepcopy(self.load_workflow()))
        uploaded = [
            self.asset_uploader.resolve_reference_image_name(path)
            for path in anchor_images[:9]
        ]
        for index in range(1, 10):
            # ComfyUI's LoadImage rejects an empty filename. Reusing the
            # anchor keeps every optional reference slot valid while the
            # R2V node still receives one semantic source image.
            patcher.set_input_by_title(f"#REF_{index}", "image", uploaded[index - 1] if index <= len(uploaded) else uploaded[0])
        patcher.set_input_by_title("#MEGAPIXELS", "megapixels", round(width * height / 1_000_000, 1))
        patcher.set_input_by_title("#FRAMECOUNT", "value", int(frames))
        patcher.set_input_by_title("#PROMPT", "value", prompt)
        if not patcher.try_set_existing_input_by_title("#SEED", "noise_seed", int(seed)):
            patcher.set_input_by_title("#SEED", "value", int(seed))
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", output_prefix)
        workflow = patcher.get()
        if self.model_resolver is not None:
            workflow = self.model_resolver.resolve_workflow_models(workflow, workflow_path=self.workflow_path)
        return workflow

    def render(
        self,
        *,
        anchor_images: list[str | Path],
        prompt: str,
        output_path: str | Path,
        seed: int = 0,
        **kwargs: Any,
    ) -> Path:
        workflow = self.build_workflow(anchor_images=anchor_images, prompt=prompt, seed=seed, **kwargs)
        return self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=1,
            output_path=Path(output_path),
        )
