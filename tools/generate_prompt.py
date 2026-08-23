from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from feverslop.adapters.llm_client import LocalOpenAIClient
from feverslop.application.prompt_generation import PromptGenerationService
from feverslop.config.app_config import AppConfig
from feverslop.prompting.dspy_h3_prompt_builder import build_dspy_generator
from feverslop.prompting.model_types import resolve_model_type


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a MiniMax H3 video prompt from a description.",
        epilog="LLM key: set the LLM_API_KEY environment variable or llm.api_key in the app config.",
    )
    parser.add_argument("--model-type", required=True, help="Supported model type, such as minimax-h3-t2v")
    parser.add_argument("--description", required=True, help="Natural-language description of the video")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="JSON_OR_FILE",
        help="Reference JSON object or path to a JSON object file; repeatable",
    )
    parser.add_argument("--duration", type=float, help="Requested duration in seconds")
    parser.add_argument("--notes", help="Additional generation notes")
    parser.add_argument("--music-intent", choices=("none", "generate", "reference"))
    parser.add_argument("--app-config", default="app_config.json", help="Path to app_config.json")
    parser.add_argument("--base-url", help="Override the configured LLM base URL")
    parser.add_argument("--model", help="Override the configured LLM model")
    fidelity = parser.add_mutually_exclusive_group()
    fidelity.add_argument("--strict-fidelity", dest="strict_fidelity", action="store_true")
    fidelity.add_argument("--no-strict-fidelity", dest="strict_fidelity", action="store_false")
    parser.set_defaults(strict_fidelity=True)
    return parser


def load_references(values: list[str]) -> list[dict[str, Any]]:
    references = []
    for value in values:
        candidate = Path(value).expanduser()
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8") if candidate.is_file() else value)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid reference JSON or file: {value}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"reference must be a JSON object: {value}")
        references.append(raw)
    return references


def _render_prompt(result: Any) -> str:
    rendered = getattr(result, "rendered_prompt", None)
    if rendered is None and isinstance(result, dict):
        rendered = result.get("rendered_prompt") or result.get("prompt")
    if not rendered:
        raise ValueError("generator returned no rendered prompt")
    return str(rendered).strip()


def _redact(message: str, secrets: list[str | None]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def main(
    argv: list[str] | None = None,
    *,
    config_loader: Callable[[str | Path], Any] = AppConfig.load,
    client_factory: Callable[..., Any] = LocalOpenAIClient,
    generator_factory: Callable[[Any], Any] = build_dspy_generator,
    service_factory: Callable[[Any], Any] = PromptGenerationService,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        resolve_model_type(args.model_type)
        references = load_references(args.reference)
        config = config_loader(args.app_config)
        llm = config.llm
        api_key = llm.api_key
        client = client_factory(
            base_url=args.base_url or llm.base_url,
            api_key=api_key,
            model=args.model or llm.model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            request_timeout_seconds=llm.request_timeout_seconds,
        )
        generator = generator_factory(client)
        service = service_factory(generator)
        result = service.generate(
            args.model_type,
            args.description,
            references=references,
            duration_seconds=args.duration,
            notes=args.notes,
            music_intent=args.music_intent,
            strict_fidelity=args.strict_fidelity,
        )
        print(_render_prompt(result))
        return 0
    except Exception as exc:
        secrets = [os.environ.get("LLM_API_KEY")]
        try:
            secrets.append(config.llm.api_key)
        except (UnboundLocalError, AttributeError):
            pass
        print(f"error: {_redact(str(exc), secrets)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
