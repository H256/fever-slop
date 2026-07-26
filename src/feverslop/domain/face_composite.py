from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CompositeResult:
    """Result of feather-compositing repaired faces back into original frames."""

    composited_frames: np.ndarray
    diagnostic_mask_path: Path | None = None
