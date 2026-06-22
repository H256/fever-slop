from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from feverslop.config.comfyui import ComfyUIModelOverride
from feverslop.path_utils import coerce_local_path


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class ComfyUIConfig:
    base_url: str = "http://127.0.0.1:8188"
    prompt_timeout_seconds: float = 1800.0
    model_overrides: list[ComfyUIModelOverride] = field(default_factory=list)


@dataclass
class AppConfig:
    llm: LLMConfig
    comfyui: ComfyUIConfig

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = coerce_local_path(path)

        if not path.exists():
            return cls(
                llm=LLMConfig(),
                comfyui=ComfyUIConfig(),
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        llm_raw = raw.get("llm", {})
        comfyui_raw = raw.get("comfyui", {})

        return cls(
            llm=LLMConfig(
                base_url=llm_raw.get("base_url", "http://localhost:8080/v1"),
                model=llm_raw.get("model", "default"),
                temperature=float(llm_raw.get("temperature", 0.7)),
                max_tokens=int(llm_raw.get("max_tokens", 4096)),
            ),
            comfyui=ComfyUIConfig(
                base_url=comfyui_raw.get("base_url", "http://127.0.0.1:8188"),
                prompt_timeout_seconds=float(comfyui_raw.get("prompt_timeout_seconds", 1800.0)),
                model_overrides=[
                    ComfyUIModelOverride.from_dict(item)
                    for item in comfyui_raw.get("model_overrides", [])
                ],
            ),
        )
