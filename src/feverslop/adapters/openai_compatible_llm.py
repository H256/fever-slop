from __future__ import annotations

from feverslop.adapters.llm_client import LocalOpenAIClient


class OpenAICompatibleLLMClient(LocalOpenAIClient):
    """Adapter name for OpenAI-compatible completion backends."""
