from __future__ import annotations

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

from feverslop.errors import FeverSlopLMLError


class LocalOpenAIClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "not-needed",
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            raise FeverSlopLMLError(f"LLM API error: {exc}") from exc

        return response.choices[0].message.content.strip()