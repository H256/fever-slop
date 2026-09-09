from __future__ import annotations

from typing import Any

from feverslop.adapters.movie_planning import DeterministicMoviePlanner, LLMMoviePlanner


def build_movie_planner(config: dict[str, Any] | None = None):
    backend = str((config or {}).get("planner_backend") or "llm").strip().lower()
    if backend in {"deterministic", "local", "placeholder"}:
        return DeterministicMoviePlanner()
    if backend != "llm":
        raise ValueError("movie_planner_backend must be llm or deterministic")

    from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(str((config or {}).get("app_config_path") or "app_config.json"))
    return LLMMoviePlanner(
        OpenAICompatibleLLMClient.from_config(app_config, task_type="creative"),
        reference_hero_workflow=(config or {}).get("hero_workflow"),
    )
