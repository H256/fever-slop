from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoWorkflowProfile:
    name: str
    pipeline: str
    workflow_path: str
    purpose: str
    stages: int
    output_scale: float
    supports_per_pass_loras: bool
    satisfies_final_output: bool

    @classmethod
    def create(
        cls,
        *,
        name: str,
        pipeline: str,
        workflow_path: str,
        purpose: str,
        stages: int,
        output_scale: float,
        supports_per_pass_loras: bool,
        satisfies_final_output: bool | None = None,
    ) -> VideoWorkflowProfile:
        resolved_name = str(name).strip()
        resolved_pipeline = str(pipeline).strip()
        resolved_workflow_path = str(workflow_path).strip()
        resolved_purpose = str(purpose).strip().lower()

        if not resolved_name or not resolved_pipeline or not resolved_workflow_path:
            raise ValueError("workflow profile name, pipeline, and path are required")

        path = Path(resolved_workflow_path)
        if path.anchor or ".." in path.parts:
            raise ValueError("workflow profile path must be repository-relative")
        if resolved_purpose not in {"preview", "final"}:
            raise ValueError("workflow profile purpose must be preview or final")

        resolved_stages = int(stages)
        if resolved_stages not in {1, 2}:
            raise ValueError("workflow profile stages must be 1 or 2")

        resolved_output_scale = float(output_scale)
        if not resolved_output_scale > 0:
            raise ValueError("workflow profile output_scale must be greater than zero")

        final = resolved_purpose == "final" if satisfies_final_output is None else bool(satisfies_final_output)
        if resolved_purpose == "preview" and final:
            raise ValueError("preview profile cannot satisfy final output")

        return cls(
            name=resolved_name,
            pipeline=resolved_pipeline,
            workflow_path=path.as_posix(),
            purpose=resolved_purpose,
            stages=resolved_stages,
            output_scale=resolved_output_scale,
            supports_per_pass_loras=bool(supports_per_pass_loras),
            satisfies_final_output=final,
        )
