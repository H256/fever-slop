from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://llm.elysium.lan/v1"
    model: str = "gemma4-26b-a4b:instruct"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass(frozen=True)
class ComfyUIConfig:
    base_url: str = "http://127.0.0.1:8188"
    enabled: bool = False


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    comfyui: ComfyUIConfig

    @classmethod
    def load(cls, config_path: str | Path = "app_config.json") -> "AppConfig":
        config_path = Path(config_path)

        if not config_path.exists():
            return cls(
                llm=LLMConfig(),
                comfyui=ComfyUIConfig(),
            )

        raw = json.loads(config_path.read_text(encoding="utf-8"))
        llm_raw = raw.get("llm", {})
        comfy_raw = raw.get("comfyui", {})

        return cls(
            llm=LLMConfig(
                base_url=llm_raw.get("base_url", "http://llm.elysium.lan/v1"),
                model=llm_raw.get("model", "gemma4-26b-a4b:instruct"),
                temperature=float(llm_raw.get("temperature", 0.7)),
                max_tokens=int(llm_raw.get("max_tokens", 4096)),
            ),
            comfyui=ComfyUIConfig(
                base_url=comfy_raw.get("base_url", "http://127.0.0.1:8188"),
                enabled=bool(comfy_raw.get("enabled", False)),
            ),
        )
