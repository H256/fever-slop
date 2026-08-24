from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOCKS_GUARD = threading.Lock()
_LOCKS: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()


@contextmanager
def artifact_write_lock(path: Path) -> Iterator[None]:
    lock_key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(lock_key, threading.RLock())
    with lock:
        yield
