from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[Path, threading.RLock] = {}


@contextmanager
def artifact_write_lock(path: Path) -> Iterator[None]:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(resolved, threading.RLock())
    with lock:
        yield
