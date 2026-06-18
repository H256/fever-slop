from __future__ import annotations

from llm_client import LocalOpenAIClient


class OpenAICompatibleLLMClient(LocalOpenAIClient):
    """Adapter name for OpenAI-compatible completion backends."""
