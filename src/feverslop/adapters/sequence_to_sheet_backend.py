"""ComfyUI adapters for backend-neutral sequence-to-sheet rendering."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
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
from feverslop.domain.reference_sheet import CompiledReferenceSheetPlan


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
        self._aspect_ratio_options: tuple[str, ...] | None = None

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

    def build_sheet_prompt_from_plan(self, plan: CompiledReferenceSheetPlan) -> H3SheetPrompt:
        """Serialize a compiled semantic plan into this backend's H3 prompt format."""
        if plan.kind == "character":
            result = build_h3_character_prompt(
                plan.identity_constraints,
                shots=plan.view_count,
                frames=max(1, round(plan.duration_seconds * 24)),
                framing=plan.framing,
                backdrop=plan.backdrop,
            )
        elif plan.kind == "location":
            result = build_h3_location_prompt(
                plan.identity_constraints,
                shots=plan.view_count,
                frames=max(1, round(plan.duration_seconds * 24)),
                coverage=plan.coverage,
                rotation=plan.rotation,
            )
        else:
            raise ValueError(f"unsupported sheet kind: {plan.kind}")
        return H3SheetPrompt(
            prompt=(
                f"{result.prompt}\n\n{plan.anchor_rule}. "
                f"Preserve these constraints: {plan.identity_constraints}. "
                f"Avoid: {plan.negative_constraints}."
            ),
            shots=plan.view_count,
            frames=result.frames,
            rotation_degrees=result.rotation_degrees,
        )

    def build_workflow(
        self,
        *,
        anchor_images: list[str | Path],
        prompt: str,
        seed: int,
        width: int = 1216,
        height: int = 672,
        aspect_ratio: str | None = None,
        frames: int = 124,
        output_prefix: str = "sequence_to_sheet/output",
    ) -> dict:
        if not anchor_images:
            raise ValueError("at least one anchor image is required")
        patcher = WorkflowPatcher(deepcopy(self.load_workflow()))
        uploaded = [
            self.asset_uploader.resolve_reference_image_name(path)
            for path in anchor_images[:9]
        ]
        patcher.set_input_by_title("#STARTFRAME", "image", uploaded[0])
        patcher.set_input_by_title("#MEGAPIXELS", "megapixels", round(width * height / 1_000_000, 2))
        if aspect_ratio is not None:
            patcher.set_input_by_title(
                "#MEGAPIXELS",
                "aspect_ratio",
                self._resolve_aspect_ratio(aspect_ratio),
            )
        patcher.set_input_by_title("#FRAMECOUNT", "value", int(frames))
        patcher.set_input_by_title("#PROMPT", "prompt", prompt)
        patcher.set_input_by_title(
            "#TURBO_LORA",
            "lora_name",
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        )
        if not patcher.try_set_existing_input_by_title("#SEED", "noise_seed", int(seed)):
            patcher.set_input_by_title("#SEED", "value", int(seed))
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", output_prefix)
        workflow = patcher.get()
        if self.model_resolver is not None:
            workflow = self.model_resolver.resolve_workflow_models(workflow, workflow_path=self.workflow_path)
        return workflow

    def _resolve_aspect_ratio(self, requested: str) -> str:
        if requested not in {"portrait", "landscape"}:
            return requested

        target = (9, 16) if requested == "portrait" else (16, 9)
        options = self._load_aspect_ratio_options()
        for option in options:
            match = re.match(r"\s*(\d+)\s*:\s*(\d+)", option)
            if match and (int(match.group(1)), int(match.group(2))) == target:
                return option

        fallback = "9:16 (Portrait Widescreen)" if requested == "portrait" else "16:9 (Widescreen)"
        return fallback

    def _load_aspect_ratio_options(self) -> tuple[str, ...]:
        if self._aspect_ratio_options is not None:
            return self._aspect_ratio_options

        get_object_info = getattr(self.client, "get_object_info", None)
        if not callable(get_object_info):
            self._aspect_ratio_options = ()
            return self._aspect_ratio_options

        try:
            selector = get_object_info().get("ResolutionSelector", {})
            required = selector.get("input", {}).get("required", {})
            raw_options = required.get("aspect_ratio", [[]])
            options = raw_options[0] if raw_options and isinstance(raw_options[0], list) else raw_options
            self._aspect_ratio_options = tuple(str(option) for option in options)
        except (AttributeError, IndexError, TypeError, ValueError):
            self._aspect_ratio_options = ()
        return self._aspect_ratio_options

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
