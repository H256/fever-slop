"""Native application logic inspired by OrbitSheets, without custom ComfyUI nodes."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _montage(paths: tuple[Any, ...], *, cell_width: int = 384) -> Image.Image:
    columns = 4
    cell_height = int(cell_width * 9 / 16)
    sheet = Image.new("RGB", (columns * cell_width, ((len(paths) + columns - 1) // columns) * cell_height), "black")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width, cell_height))
            x = (index % columns) * cell_width + (cell_width - image.width) // 2
            y = (index // columns) * cell_height + (cell_height - image.height) // 2
            sheet.paste(image, (x, y))
        draw.text(((index % columns) * cell_width + 8, (index // columns) * cell_height + 8), str(index + 1), fill="white")
    return sheet


def _vision_picks(paths: tuple[Any, ...], count: int, endpoint: str, subject: str) -> tuple[int, ...] | None:
    import requests

    buffer = io.BytesIO()
    _montage(paths).save(buffer, format="JPEG", quality=88)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    prompt = (
        f"Choose exactly {count} frames from this numbered montage of a single H3 reference take of {subject}. "
        "Pick sharp, distinct views that cover different angles; reject duplicates, blur, crops and mid-cut frames. "
        'Reply JSON only: {"picks":[1,2],"why":"short reason"}'
    )
    try:
        response = requests.post(
            endpoint.rstrip("/") + "/v1/chat/completions",
            json={"messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":data_uri}}]}],"temperature":0.2,"max_tokens":200},
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(re.sub(r"```(?:json)?", "", content).strip())
        picks = tuple(int(value) - 1 for value in payload.get("picks", ()))
        valid = tuple(index for index in picks if 0 <= index < len(paths))
        return valid[:count] if len(valid) >= count else None
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException):
        return None


def select_orbitsheet_frames(
    frame_paths: Iterable[Any],
    *,
    count: int,
    subject: str,
    vision_endpoint: str | None = None,
) -> tuple[Any, ...]:
    """Select views using a vision judge, then deterministic content spread fallback."""
    paths = tuple(frame_paths)
    if not paths or count < 1:
        raise ValueError("frame paths and positive count are required")
    candidates = paths[:16]
    if vision_endpoint:
        picks = _vision_picks(candidates, count, vision_endpoint, subject)
        if picks is not None:
            return tuple(candidates[index] for index in picks)

    descriptors = []
    sharpness = []
    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not read frame image: {path}")
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = (small - small.mean()) / (small.std() + 1e-6)
        descriptors.append(small.reshape(-1))
        sharpness.append(float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
    descriptors = np.asarray(descriptors)
    chosen = [0]
    while len(chosen) < min(count, len(candidates)):
        distances = np.min(np.linalg.norm(descriptors[:, None] - descriptors[chosen][None, :], axis=2), axis=1)
        distances[chosen] = -1
        score = distances / max(float(distances.max()), 1e-6) * 0.65
        normalized = (np.asarray(sharpness) - min(sharpness)) / max(max(sharpness) - min(sharpness), 1e-6)
        score += normalized * 0.35
        chosen.append(int(np.argmax(score)))
    return tuple(candidates[index] for index in sorted(chosen))
