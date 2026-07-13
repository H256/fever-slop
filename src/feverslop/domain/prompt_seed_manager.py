from __future__ import annotations

import hashlib


class PromptSeedManager:
    """Manages reproducible per-prompt seeds derived from a global seed.

    Each prompt gets a unique seed computed as a hash of (global_seed, prompt_index).
    This allows changing the global seed to shift all seeds while keeping per-prompt
    independence.
    """

    MAX_SEED = 2**31 - 1

    def __init__(self, global_seed: int = 0):
        self._global_seed = int(global_seed) % (self.MAX_SEED + 1)

    def get_global_seed(self) -> int:
        """Return the global seed for backward compatibility."""
        return self._global_seed

    def get_seed_for_prompt(self, prompt_index: int) -> int:
        """Derive a unique, reproducible seed for a given prompt index.

        Uses a hash of (global_seed, prompt_index) to ensure seeds are
        uniformly distributed and independent, while remaining fully
        deterministic.
        """
        key = f"{self._global_seed}:{prompt_index}"
        hash_value = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return hash_value % (self.MAX_SEED + 1)
