from __future__ import annotations

import json
import re

from feverslop.errors import FeverSlopLMLError


def extract_json_object(text: str) -> dict:
    text = text.strip()

    # Fast path: clean JSON response
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```.*$", "", text, flags=re.DOTALL).strip()

    # Try again after stripping fences
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    pos = 0
    last_error = None
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        # Find matching brace via depth counting, skipping string literals
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                if in_string:
                    escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string and ch in "{}":
                depth += 1 if ch == "{" else -1
                if depth == 0:
                    end = i
                    break
        else:
            break  # unmatched braces, give up
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            last_error = exc
            pos = start + 1

    raise FeverSlopLMLError(
        f"No valid JSON object found in LLM response:\n{text}",
    ) from last_error
