from __future__ import annotations

import multiprocessing
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from feverslop.adapters.h3_prompt_checkpoints import H3PromptCheckpointStore
from feverslop.domain.h3_prompt_checkpoint import H3PromptCheckpointInput


def request():
    return H3PromptCheckpointInput(1, 'a', {'lyrics': 'hello', 'seed': 1}, 'concept', {}, {}, 'r2v', 'music_video')


def contender(path, queue, entered=None, release=None):
    with H3PromptCheckpointStore(path).recovery_session(request()) as session:
        queue.put(session.reserve('generate'))
        if entered is not None:
            entered.set()
        if release is not None:
            release.wait(10)


class SceneRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = H3PromptCheckpointStore(self.temp.name)

    def test_crash_reservations_and_exhausted_resumes(self):
        for stage in ('generate', 'repair', 'fallback'):
            with self.assertRaises(RuntimeError):
                with self.store.recovery_session(request()) as session:
                    self.assertTrue(session.reserve(stage))
                    raise RuntimeError('simulated crash')
        for _ in range(3):
            with self.store.recovery_session(request()) as session:
                self.assertFalse(any(session.reserve(stage) for stage in ('generate', 'repair', 'fallback')))
                self.assertEqual('blocked', session.readiness['status'])

    def test_budget_identity_and_explicit_revision(self):
        ref = Path(self.temp.name) / 'reference.png'
        ref.write_bytes(b'first')
        original = replace(request(), segment={'lyrics': 'hello', 'seed': 1, 'references': {'image_path': str(ref)}})
        with self.store.recovery_session(original) as session:
            self.assertTrue(session.reserve('generate'))
        os.utime(ref, (100, 100))
        same = replace(original, segment={**original.segment, 'seed': 42}, generator_revision={'model': 'different'})
        with self.store.recovery_session(same) as session:
            self.assertFalse(session.reserve('generate'))
        for changed in (replace(original, segment={**original.segment, 'lyrics': 'new'}),):
            with self.store.recovery_session(changed) as session:
                self.assertTrue(session.reserve('generate'))
        ref.write_bytes(b'other')
        with self.store.recovery_session(original) as session:
            self.assertTrue(session.reserve('generate'))
            revision = session.attempt_revision
        with self.store.recovery_session(original, replan=True) as session:
            self.assertEqual(revision + 1, session.attempt_revision)
            self.assertTrue(session.reserve('generate'))

    def test_processes_cannot_both_reserve(self):
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        processes = [context.Process(target=contender, args=(self.temp.name, queue)) for _ in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
            self.assertEqual(0, process.exitcode)
        self.assertEqual([False, True], sorted([queue.get(timeout=2), queue.get(timeout=2)]))

    def test_blocked_empty_checkpoint_roundtrips_without_reuse(self):
        generated = {'prompt': '', 'readiness': {'status': 'blocked', 'reason_codes': ['invalid']}}
        checkpoint = self.store.save(request(), generated)
        self.assertEqual('blocked', checkpoint.status)
        self.assertEqual(generated, self.store.load_for_resume(request()).generated)
        self.assertIsNone(self.store.load(request()))
        self.assertIsNone(self.store.load_advisory(request()))

    def test_lock_covers_entire_expensive_operation(self):
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        entered, release = context.Event(), context.Event()
        second_entered = context.Event()
        first = context.Process(target=contender, args=(self.temp.name, queue, entered, release))
        second = context.Process(target=contender, args=(self.temp.name, queue, second_entered))
        first.start()
        try:
            self.assertTrue(entered.wait(10))
            second.start()
            self.assertFalse(second_entered.wait(0.5))
        finally:
            release.set()
            first.join(10)
            if second.pid is not None:
                second.join(10)
        self.assertEqual(0, first.exitcode)
        self.assertEqual(0, second.exitcode)
        self.assertTrue(second_entered.is_set())

    def test_saved_blocked_result_matches_only_current_revision(self):
        with self.store.recovery_session(request()) as session:
            readiness = session.finish('blocked', ['missing_timing'])
            self.store.save(request(), {'prompt': '', 'readiness': readiness})
        with self.store.recovery_session(request()) as session:
            self.assertEqual(readiness, session.saved_result['readiness'])
        with self.store.recovery_session(request(), replan=True) as session:
            self.assertIsNone(session.saved_result)

    def test_returning_to_old_input_preserves_budget(self):
        changed = replace(request(), concept='changed')
        for item in (request(), changed):
            with self.store.recovery_session(item) as session:
                self.assertTrue(session.reserve('generate'))
        with self.store.recovery_session(request()) as session:
            self.assertFalse(session.reserve('generate'))

    def test_reference_content_change_with_preserved_stat_gets_new_budget(self):
        ref = Path(self.temp.name) / 'ref.png'
        ref.write_bytes(b'aaaa')
        item = replace(request(), segment={'references': {'image_path': str(ref)}})
        with self.store.recovery_session(item) as session:
            self.assertTrue(session.reserve('generate'))
        stat = ref.stat()
        ref.write_bytes(b'bbbb')
        os.utime(ref, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.store.recovery_session(item) as session:
            self.assertTrue(session.reserve('generate'))
