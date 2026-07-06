from __future__ import annotations

from pathlib import Path

from PIL import Image

from feverslop.ports.rendering import ImageRenderRequest


class LocalMovieImageBackend:
    def render_image(self, request: ImageRenderRequest) -> Path:
        output = Path(request.output_dir) / f"scene_{int(request.scene_number):04}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (int(request.width or 1280), int(request.height or 704)), "white").save(output)
        return output
