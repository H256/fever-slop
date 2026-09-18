"""Per-family boundary specs for the generic workflow inspector.

The inspector is family-agnostic; this module supplies each pipeline family's
boundary as pure data. Adding a new family is a new entry here, not a new
inspector implementation.

Notes on the families (verified against the repo's backends/workflows):

- ``minimax_h3``: core is ``MiniMaxH3ReferenceToVideo`` (R2V) or the H3
  video sampler; output is ``VAEDecode``/``VAEDecodeAudio``.
- ``ltx_ingredients`` / ``ltx_msr`` / ``ltx_i2v``: LTXV families whose core
  sampler is ``LTXVImgToVideoInplace`` (ingredients/MSR) or the native
  i2v path; terminal output is ``SaveVideo`` or ``VHS_VideoCombine``.
- ``krea``: a startframe/director *image* backend (``krea2``), not a
  ComfyUI video workflow family, so it has no video workflow boundary and
  is intentionally absent from ``FAMILY_SPECS``.
"""

from __future__ import annotations

from feverslop.domain.workflow_import import WorkflowFamilySpec

_VIDEO_OUTPUTS = ("SaveVideo", "VHS_VideoCombine", "VHS_VideoInfo", "CreateVideo")

#: LTXV video-generation core. The backends inject ``LTXVImgToVideoInplace``
#: into the base templates, but the committed MSR/ingredients templates carry
#: ``SamplerCustomAdvanced`` as the actual sampler. Recognize both so the
#: generic inspector maps the boundary on either form of a real workflow.
_LTXV_CORE = ("LTXVImgToVideoInplace", "SamplerCustomAdvanced")

FAMILY_SPECS: dict[str, WorkflowFamilySpec] = {
    "minimax_h3": WorkflowFamilySpec(
        pipeline="minimax_h3",
        core_class_types=(
            "MiniMaxH3ReferenceToVideo",
            "MiniMaxH3Video",
        ),
        required_inputs=(),
        output_class_types=("VAEDecode", "VAEDecodeAudio"),
    ),
    "ltx_ingredients": WorkflowFamilySpec(
        pipeline="ltx_ingredients",
        core_class_types=_LTXV_CORE,
        required_inputs=(),
        output_class_types=_VIDEO_OUTPUTS,
    ),
    "ltx_msr": WorkflowFamilySpec(
        pipeline="ltx_msr",
        core_class_types=_LTXV_CORE,
        required_inputs=(),
        output_class_types=_VIDEO_OUTPUTS,
    ),
    "ltx_i2v": WorkflowFamilySpec(
        pipeline="ltx_i2v",
        core_class_types=_LTXV_CORE,
        required_inputs=(),
        output_class_types=_VIDEO_OUTPUTS,
    ),
}

#: Families that have no ComfyUI video workflow boundary (image/director backends).
NON_WORKFLOW_FAMILIES: frozenset[str] = frozenset({"krea"})


def spec_for(pipeline: str) -> WorkflowFamilySpec | None:
    """Return the boundary spec for a pipeline family, or None if it has none."""
    return FAMILY_SPECS.get(pipeline)
