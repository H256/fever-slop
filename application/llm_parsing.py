from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    return json.loads(text[start:end + 1])
