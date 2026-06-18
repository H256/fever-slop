from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AudioAnalyzerPort(Protocol):
    def analyze(self, audio_file: Path) -> dict:
        """Analyze an audio file and return backend-specific timeline data."""
