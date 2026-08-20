from __future__ import annotations

import base64
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from feverslop.studio.projects import AUDIO_EXTENSIONS, AUDIO_MIME_TYPES, StudioPathError, sanitize_audio_filename
from feverslop.utils.io import atomic_write_bytes, atomic_write_json


class MediaStore:
    def __init__(
        self,
        project_root: Callable[[str], Path],
        resolve_project_path: Callable[[str, str], Path],
        read_json_file: Callable[[Path], Any],
        max_upload_size: int = 100 * 1024 * 1024,
    ):
        self.project_root = project_root
        self.resolve_project_path = resolve_project_path
        self.read_json_file = read_json_file
        self.max_upload_size = max_upload_size

    def write_media_data_url(self, project_id: str, path: str, data_url: str) -> dict[str, str]:
        media_path = self.resolve_project_path(project_id, path)
        if media_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise StudioPathError("Unsupported uploaded media type")
        header, separator, encoded = data_url.partition(",")
        if not separator or not header.startswith("data:image/"):
            raise StudioPathError("Expected an image data URL")
        media_path.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(encoded)
        if len(raw) > self.max_upload_size:
            raise StudioPathError(
                f"Media upload exceeds size limit ({len(raw)} bytes > {self.max_upload_size} bytes)"
            )
        atomic_write_bytes(media_path, raw)
        return {"path": path}

    def store_audio_upload(self, project_id: str, filename: str, content_type: str, source) -> dict[str, str]:
        safe_name = sanitize_audio_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS or str(content_type or "").lower() not in AUDIO_MIME_TYPES:
            raise ValueError("Unsupported audio type")
        root = self.project_root(project_id)
        input_dir = root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        target = (input_dir / safe_name).resolve()
        if not target.is_relative_to(input_dir):
            raise StudioPathError("Path escapes project input directory")

        fd, temp_name = tempfile.mkstemp(prefix=f".{safe_name}.", suffix=".tmp", dir=input_dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as temp_file:
                shutil.copyfileobj(source, temp_file)
            temp_path.replace(target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        relative_path = target.relative_to(root).as_posix()
        config_path = root / "config.json"
        if config_path.exists():
            config = self.read_json_file(config_path)
            if not isinstance(config, dict):
                config = {}
            config["input_audio"] = relative_path
            atomic_write_json(config_path, config)
        return {"path": relative_path}
