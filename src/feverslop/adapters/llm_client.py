from __future__ import annotations

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
import time

from feverslop.errors import FeverSlopLMLError


RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


class LocalOpenAIClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "not-needed",
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def complete_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
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

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                return response.choices[0].message.content.strip()
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)

        raise FeverSlopLMLError(
            f"LLM API error after {self.max_retries} attempts: {last_error}"
        ) from last_error