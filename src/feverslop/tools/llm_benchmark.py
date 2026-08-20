from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Any

from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient


def benchmark_prompts(client: Any, prompts: list[str], *, system_prompt: str = "Return a concise answer.") -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    started_total = perf_counter()
    for prompt in prompts:
        started = perf_counter()
        result = str(client.complete_prompt(system_prompt=system_prompt, prompt=prompt))
        duration_ms = (perf_counter() - started) * 1000
        telemetry = getattr(client, "last_response_telemetry", None)
        sample = {
            "latency_ms": duration_ms,
            "output_words": len(result.split()),
            "prompt_tokens": int(getattr(telemetry, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(telemetry, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(telemetry, "reasoning_tokens", 0) or 0),
            "total_tokens": int(getattr(telemetry, "total_tokens", 0) or 0),
            "finish_reason": getattr(telemetry, "finish_reason", None),
        }
        samples.append(sample)
    total_ms = (perf_counter() - started_total) * 1000
    requests = len(samples)
    return {
        "requests": requests,
        "latency_ms": {
            "total": total_ms,
            "average": total_ms / requests if requests else 0.0,
        },
        "output": {"words": sum(sample["output_words"] for sample in samples)},
        "tokens": {
            key: sum(sample[f"{key}_tokens"] for sample in samples)
            for key in ("prompt", "completion", "reasoning", "total")
        },
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark one configured LLM endpoint.", epilog="Authenticate with the LLM_API_KEY environment variable.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--dspy-temperature", type=float, default=0.4)
    parser.add_argument("--prompt-file", required=True, help="UTF-8 file with one prompt per line")
    parser.add_argument("--system-prompt", default="Return a concise answer.")
    args = parser.parse_args(argv)
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        parser.error("LLM_API_KEY environment variable is required")
    prompts = [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]
    client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        dspy_temperature=args.dspy_temperature,
        max_tokens=2048,
        max_concurrent_requests=1,
    )
    print(json.dumps(benchmark_prompts(client, prompts, system_prompt=args.system_prompt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
