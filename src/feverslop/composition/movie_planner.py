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

    app_config = AppConfig.load("app_config.json")
    return LLMMoviePlanner(
        OpenAICompatibleLLMClient(
            base_url=app_config.llm.base_url,
            api_key=app_config.llm.api_key,
            model=app_config.llm.model_for("creative"),
            temperature=app_config.llm.temperature,
            dspy_temperature=app_config.llm.dspy_temperature,
            max_tokens=app_config.llm.max_tokens,
            request_timeout_seconds=app_config.llm.request_timeout_seconds,
            max_concurrent_requests=app_config.llm.max_concurrent_requests,
        ),
        reference_hero_workflow=(config or {}).get("hero_workflow"),
    )
