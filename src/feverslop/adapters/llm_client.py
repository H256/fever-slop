from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
import random
import time

from feverslop.adapters.api_observability import APIMetrics, default_api_metrics, record_api_call
from feverslop.errors import FeverSlopLMLError
from feverslop.prompting.vision_references import prepare_vision_image
from feverslop.security.url_validation import validate_api_url


RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)
logger = logging.getLogger(__name__)


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve API key: explicit > env var > error."""
    if api_key is not None:
        if api_key == "not-needed":
            raise ValueError(
                "Hardcoded API key 'not-needed' detected. Set LLM_API_KEY "
                "environment variable or pass a valid api_key."
            )
        if not api_key:
            raise ValueError(
                "LLM API key is empty. Set LLM_API_KEY, configure llm.api_key "
                "in app_config.json or LLM_API_KEY in .env, or pass a valid api_key."
            )
        return api_key
    env_key = os.environ.get("LLM_API_KEY", "")
    env_key = env_key.strip()
    if env_key == "not-needed":
        raise ValueError(
            "Hardcoded API key 'not-needed' in LLM_API_KEY. "
            "Set a valid API key."
        )
    if not env_key:
        raise ValueError(
            "No LLM API key provided. Set LLM_API_KEY, configure llm.api_key "
            "in app_config.json or LLM_API_KEY in .env, or pass a valid api_key argument."
        )
    return env_key


class LocalOpenAIClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        request_timeout_seconds: float = 180.0,
        dspy_cache: bool = False,
        metrics: APIMetrics | None = None,
        auth_headers: dict[str, str] | None = None,
    ):
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        resolved_key = _resolve_api_key(api_key)
        self.auth_headers = dict(auth_headers or {})
        validate_api_url(base_url)
        self.client = OpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=self.auth_headers or None,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.dspy_cache = dspy_cache
        self.metrics = metrics or default_api_metrics

    def complete_prompt(
        self,
        system_prompt: str,
        prompt: str,
        timeout: float | None = None,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        return self._complete(messages, timeout)

    def complete_prompt_with_images(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[Path],
        timeout: float | None = None,
    ) -> str:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            mime_type, image_bytes = prepare_vision_image(path)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self._complete(messages, timeout)

    def _complete(self, messages, timeout: float | None) -> str:
        last_error = None
        request_timeout = self.request_timeout_seconds if timeout is None else timeout
        for attempt in range(self.max_retries):
            started_at = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                    timeout=request_timeout,
                )
                choices = getattr(response, "choices", None)
                if not isinstance(choices, list) or not choices:
                    raise FeverSlopLMLError("LLM response missing choices")
                message = getattr(choices[0], "message", None)
                result = str(getattr(message, "content", "") or "").strip()
                if not result:
                    raise FeverSlopLMLError("LLM response contains empty content")
                record_api_call(self.metrics, logger, "llm", "chat_completions", started_at, success=True)
                return result
            except RETRYABLE_ERRORS as exc:
                record_api_call(self.metrics, logger, "llm", "chat_completions", started_at, success=False)
                last_error = exc
                if attempt < self.max_retries - 1:
                    base_delay = self.retry_base_delay * (2 ** attempt)
                    jitter = random.uniform(0, base_delay)
                    time.sleep(base_delay + jitter)
            except Exception:
                record_api_call(self.metrics, logger, "llm", "chat_completions", started_at, success=False)
                raise

        raise FeverSlopLMLError(
            f"LLM API error after {self.max_retries} attempts: {last_error}"
        ) from last_error
