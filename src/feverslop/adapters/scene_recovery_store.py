from __future__ import annotations

import os
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from feverslop.domain.scene_recovery import RECOVERY_POLICY_VERSION, RecoveryStage
from feverslop.errors import FeverSlopDataError
from feverslop.utils.io import atomic_write_json, read_json_document


@contextmanager
def scene_file_lock(path: Path, *, waiting: Callable[[], None] | None = None) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    next_update = time.monotonic() + 5

    def report_wait() -> None:
        nonlocal next_update
        if waiting is not None and time.monotonic() >= next_update:
            waiting()
            next_update = time.monotonic() + 5

    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (13, 11, 36):
                        raise
                    report_wait()
                    time.sleep(0.05)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    report_wait()
                    time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PersistentSceneRecovery:
    def __init__(self, path: Path, fingerprint: str, *, replan: bool = False) -> None:
        self.path = path
        self.input_fingerprint = fingerprint
        self.saved_result: dict[str, Any] | None = None
        self._ledger = read_json_document(path) if path.is_file() else {'schema': 'feverslop.scene-recovery.v1', 'inputs': {}}
        if not isinstance(self._ledger, dict) or self._ledger.get('schema') != 'feverslop.scene-recovery.v1' or not isinstance(self._ledger.get('inputs'), dict):
            raise FeverSlopDataError('Invalid scene recovery ledger')
        inputs = self._ledger['inputs']
        history = inputs.setdefault(fingerprint, [])
        if not isinstance(history, list):
            raise FeverSlopDataError('Invalid scene recovery history')
        if not history or replan:
            history.append({'status': 'blocked', 'stage': 'pending', 'reason_codes': [], 'input_fingerprint': fingerprint, 'policy_version': RECOVERY_POLICY_VERSION, 'attempt_revision': len(history), 'attempts': []})
        self._state = history[-1]
        if not isinstance(self._state, dict) or not isinstance(self._state.get('attempts'), list):
            raise FeverSlopDataError('Invalid scene recovery attempts')
        self.attempt_revision = self._state['attempt_revision']
        self._write()

    @property
    def readiness(self) -> dict[str, Any]:
        return deepcopy(self._state)

    @property
    def state(self) -> dict[str, Any]:
        return self.readiness

    def _write(self) -> None:
        atomic_write_json(self.path, self._ledger)

    def reserve(self, stage: RecoveryStage) -> bool:
        if stage not in ('generate', 'repair', 'fallback'):
            raise ValueError(f'Unknown recovery stage: {stage}')
        if any(attempt['stage'] == stage for attempt in self._state['attempts']):
            return False
        self._state['attempts'].append({'stage': stage, 'status': 'reserved'})
        self._state.update(status='blocked', stage=stage, reason_codes=['attempt_interrupted'])
        self._write()
        return True

    def finish(self, status: Literal['ready', 'blocked'], reason_codes: list[str], *, stage: str | None = None) -> dict[str, Any]:
        if status not in ('ready', 'blocked'):
            raise ValueError('Readiness must be ready or blocked')
        self._state.update(status=status, reason_codes=list(dict.fromkeys(reason_codes)))
        if stage is not None:
            self._state['stage'] = stage
        self._write()
        return self.readiness
