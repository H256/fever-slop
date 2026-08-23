from __future__ import annotations

import ast
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from feverslop.adapters.api_observability import (
    APIMetrics,
    default_api_metrics,
    record_api_call,
)
from feverslop.domain.llm_parsing import extract_json_object
from feverslop.errors import FeverSlopLMLError

logger = logging.getLogger(__name__)


class Gemma4StartframeValidator:
    """Separate external LLM validation API, outside the DSPy prompt boundary."""

    def __init__(
        self,
        *,
        base_url: str = "http://your-llm-server.local/v1",
        model: str = "gemma4-26b-a4b:vision",
        timeout_seconds: float = 180.0,
        metrics: APIMetrics | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.metrics = metrics or default_api_metrics

    def validate_startframe(
        self,
        *,
        image_path: str | Path,
        shot_contract: dict[str, Any],
        identity_ledger: dict[str, Any],
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict film continuity and character identity validator. "
                        "Return ONLY JSON with keys: pass, score, issues, notes."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Validate whether this startframe matches the shot contract, story continuity, "
                                "character identity, wardrobe, location, and action moment. "
                                "Use a high bar; identity includes clothing. JSON contract:\n"
                                + json.dumps(
                                    {
                                        "shot_contract": shot_contract,
                                        "identity_ledger": identity_ledger,
                                    },
                                    ensure_ascii=False,
                                )
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path)},
                        },
                    ],
                },
            ],
        }
        started_at = time.perf_counter()
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_seconds)
        except Exception:
            record_api_call(self.metrics, logger, "llm", "startframe_validation", started_at, success=False)
            raise
        record_api_call(self.metrics, logger, "llm", "startframe_validation", started_at, success=response.ok)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return normalize_validation_response(content)


def normalize_validation_response(content: str) -> dict[str, Any]:
    try:
        result = extract_json_object(content)
        return {
            "pass": bool(result.get("pass")),
            "score": float(result.get("score") or 0.0),
            "issues": [str(item) for item in result.get("issues") or []],
            "notes": str(result.get("notes") or ""),
        }
    except FeverSlopLMLError:
        return _validation_from_text(content)


def _validation_from_text(content: str) -> dict[str, Any]:
    text = str(content or "")
    passed = _extract_bool(text, default=False)
    score = _extract_score(text)
    issues = _extract_issues(text)
    notes = _extract_notes(text)
    return {"pass": passed, "score": score, "issues": issues, "notes": notes}


def _extract_bool(text: str, *, default: bool) -> bool:
    matches = re.findall(r"\bpass\s*:\s*(true|false|yes|no)\b", text, flags=re.IGNORECASE)
    if not matches:
        return default
    value = matches[-1].lower()
    return value in {"true", "yes"}


def _extract_score(text: str) -> float:
    matches = re.findall(r"\bscore\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if not matches:
        return 0.0
    return float(matches[-1])


def _extract_issues(text: str) -> list[str]:
    matches = re.findall(r"\bissues\s*:\s*(\[[^\]]*\]|[^\n]+)", text, flags=re.IGNORECASE)
    if not matches:
        return []
    raw = matches[-1].strip()
    if raw.startswith("["):
        try:
            value = ast.literal_eval(raw)
            if isinstance(value, list):
                return [str(item) for item in value]
        except (SyntaxError, ValueError):
            pass
    return [item.strip(" -\"'") for item in re.split(r";|\n", raw) if item.strip(" -\"'")]


def _extract_notes(text: str) -> str:
    matches = re.findall(r"\bnotes\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if not matches:
        return text.strip()[:1000]
    return matches[-1].strip().strip("\"'")


def _image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
