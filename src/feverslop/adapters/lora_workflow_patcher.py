from __future__ import annotations

from dataclasses import dataclass

from feverslop.adapters.workflow_patcher import WorkflowPatcher


@dataclass(frozen=True)
class LoraPatchSettings:
    character_lora_node_title: str | None
    character_lora_strength: float | None
    lora_1_enabled: bool
    lora_1_name: str
    lora_1_strength_model: float
    lora_1_strength_clip: float
    lora_1_strengths_explicit: bool
    lora_1_node_title: str
    lora_split_enabled: bool


class LoraWorkflowPatcher:
    def __init__(self, settings: LoraPatchSettings, loras: tuple[object, ...]):
        self.settings = settings
        self.loras = tuple(loras)

    def patch_lora_inputs(self, patcher: WorkflowPatcher) -> None:
        active_loras = self.active_loras()
        if active_loras:
            for lora in active_loras:
                split_anchor_exists = self.has_anchor(patcher, lora.split_title)
                patch_base_as_split_first_pass = self.settings.lora_split_enabled and split_anchor_exists

                if self.settings.lora_split_enabled:
                    self.patch_single_lora(
                        patcher,
                        title=lora.lora_title,
                        lora=lora,
                        strength_model=(
                            lora.strength_model * 0.5
                            if lora.strength_model_explicit and patch_base_as_split_first_pass
                            else lora.strength_model if lora.strength_model_explicit else None
                        ),
                        strength_clip=(
                            lora.strength_clip * 0.5
                            if lora.strength_clip_explicit and patch_base_as_split_first_pass
                            else lora.strength_clip if lora.strength_clip_explicit else None
                        ),
                    )
                else:
                    self.patch_single_lora(
                        patcher,
                        title=lora.lora_title,
                        lora=lora,
                        strength_model=lora.strength_model if lora.strength_model_explicit else None,
                        strength_clip=lora.strength_clip if lora.strength_clip_explicit else None,
                    )
                if split_anchor_exists:
                    self.patch_single_lora(
                        patcher,
                        title=lora.split_title,
                        lora=lora,
                        strength_model=lora.strength_model if lora.strength_model_explicit else None,
                        strength_clip=lora.strength_clip if lora.strength_clip_explicit else None,
                    )
        elif self.settings.lora_1_strengths_explicit:
            patcher.patch_lora_strengths_by_title(
                self.settings.lora_1_node_title,
                strength_model=self.settings.lora_1_strength_model,
                strength_clip=self.settings.lora_1_strength_clip,
            )
        elif self.settings.character_lora_node_title and self.settings.character_lora_strength is not None:
            try:
                patcher.patch_lora_strength_by_title(
                    self.settings.character_lora_node_title,
                    self.settings.character_lora_strength,
                )
            except KeyError:
                pass

    def active_loras(self) -> tuple[object, ...]:
        if self.loras:
            return tuple(lora for lora in self.loras if lora.enabled)
        if self.settings.lora_1_enabled:
            from feverslop.adapters.ltx_workflow_patcher import ResolvedLoraConfig

            return (
                ResolvedLoraConfig(
                    index=1,
                    enabled=True,
                    name=self.settings.lora_1_name,
                    strength_model=self.settings.lora_1_strength_model,
                    strength_clip=self.settings.lora_1_strength_clip,
                    name_explicit=bool(str(self.settings.lora_1_name).strip()),
                    strength_model_explicit=True,
                    strength_clip_explicit=True,
                ),
            )
        return ()

    @staticmethod
    def has_anchor(patcher: WorkflowPatcher, title: str) -> bool:
        try:
            patcher.find_node_by_meta_title(title)
            return True
        except KeyError:
            return False

    @staticmethod
    def patch_single_lora(
        patcher: WorkflowPatcher,
        *,
        title: str,
        lora: object,
        strength_model: float | None,
        strength_clip: float | None,
    ) -> None:
        if not lora.name_explicit and strength_model is None and strength_clip is None:
            return
        try:
            patcher.patch_lora_fields_by_title(
                title,
                lora_name=lora.name if lora.name_explicit else None,
                strength_model=strength_model,
                strength_clip=strength_clip,
            )
        except KeyError as exc:
            raise ValueError(f"Missing or incompatible LoRA workflow anchor {title}") from exc
