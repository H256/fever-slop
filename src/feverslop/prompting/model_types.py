from dataclasses import dataclass
from types import MappingProxyType

from feverslop.prompting.dspy_h3_models import PromptMode


@dataclass(frozen=True, slots=True)
class ModelTypeSpec:
    model_type: str
    prompt_mode: PromptMode
    is_minimax_h3: bool
    guide_filename: str


MODEL_TYPES = MappingProxyType(
    {
        "minimax-h3-t2v": ModelTypeSpec(
            model_type="minimax-h3-t2v",
            prompt_mode=PromptMode.T2V,
            is_minimax_h3=True,
            guide_filename="minimax-h3-base.md",
        ),
        "minimax-h3-i2v": ModelTypeSpec(
            model_type="minimax-h3-i2v",
            prompt_mode=PromptMode.I2V,
            is_minimax_h3=True,
            guide_filename="minimax-h3-base.md",
        ),
        "minimax-h3-fl2v": ModelTypeSpec(
            model_type="minimax-h3-fl2v",
            prompt_mode=PromptMode.FL2V,
            is_minimax_h3=True,
            guide_filename="minimax-h3-base.md",
        ),
        "minimax-h3-l2v": ModelTypeSpec(
            model_type="minimax-h3-l2v",
            prompt_mode=PromptMode.L2V,
            is_minimax_h3=True,
            guide_filename="minimax-h3-base.md",
        ),
        "minimax-h3-r2v": ModelTypeSpec(
            model_type="minimax-h3-r2v",
            prompt_mode=PromptMode.R2V,
            is_minimax_h3=True,
            guide_filename="minimax-h3-references.md",
        ),
    }
)


def resolve_model_type(value: str) -> ModelTypeSpec:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    try:
        return MODEL_TYPES[normalized]
    except KeyError as error:
        accepted = ", ".join(MODEL_TYPES)
        raise ValueError(f"unknown model type {value!r}; accepted values: {accepted}") from error