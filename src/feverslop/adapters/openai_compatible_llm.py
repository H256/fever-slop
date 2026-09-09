from __future__ import annotations

from feverslop.adapters.llm_client import LocalOpenAIClient


class OpenAICompatibleLLMClient(LocalOpenAIClient):
    """Adapter name for OpenAI-compatible completion backends."""

    @classmethod
    def from_config(cls, app_config, *, task_type: str | None = None, **overrides):
        """Build a client from the shared LLM configuration."""
        llm = app_config.llm
        params = {
            "base_url": llm.base_url,
            "api_key": llm.api_key,
            "model": llm.model_for(task_type),
            "temperature": llm.temperature,
            "dspy_temperature": llm.dspy_temperature,
            "max_tokens": llm.max_tokens,
            "request_timeout_seconds": llm.request_timeout_seconds,
            "dspy_cache": llm.dspy_cache,
            "max_concurrent_requests": llm.max_concurrent_requests,
            "prompt_judge_attempts": llm.prompt_judge_attempts,
            "prompt_judge_max_tokens": llm.prompt_judge_max_tokens,
            "prompt_judge_blocking": llm.prompt_judge_blocking,
            "chat_template_kwargs": llm.chat_template_kwargs,
        }
        params.update(overrides)
        return cls(**params)
