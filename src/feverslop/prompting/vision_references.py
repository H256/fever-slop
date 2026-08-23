from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from feverslop.domain.vision_references import ReferenceImage

__all__ = ["ReferenceImage", "prepare_vision_image"]


def prepare_vision_image(path: Path, *, max_side: int = 1024) -> tuple[str, bytes]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_side:
            image = ImageOps.contain(
                image,
                (max_side, max_side),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True)
    return "image/jpeg", output.getvalue()
