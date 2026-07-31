import datetime
import unittest

from feverslop.domain.prompt_revisions import (
    PromptField,
    PromptHistory,
    PromptRevision,
    build_revision,
    restore_revision,
)


class BuildRevisionTests(unittest.TestCase):
    def test_stable_id_from_same_inputs(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        rev = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=None,
            now=ts,
        )
        self.assertIsInstance(rev, PromptRevision)
        self.assertEqual(rev.scene_number, 3)
        self.assertEqual(rev.field, PromptField.Z_IMAGE_PROMPT)
        self.assertEqual(rev.value, "A singer on stage")
        self.assertIsNone(rev.restored_from)

    def test_different_id_for_different_value(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        rev_a = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=None,
            now=ts,
        )
        rev_b = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A different prompt",
            parent_id=None,
            now=ts,
        )
        self.assertNotEqual(rev_a.id, rev_b.id)

    def test_same_value_different_time_gets_different_id(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 11, 0, 0, tzinfo=datetime.timezone.utc)
        rev1 = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=None,
            now=ts1,
        )
        rev2 = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=rev1.id,
            now=ts2,
        )
        self.assertNotEqual(rev1.id, rev2.id)

    def test_content_hash_identical_for_same_text(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 11, 0, 0, tzinfo=datetime.timezone.utc)
        rev1 = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=None,
            now=ts1,
        )
        rev2 = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=rev1.id,
            now=ts2,
        )
        self.assertEqual(rev1.content_hash, rev2.content_hash)

    def test_parent_id_chain(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        rev1 = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="v1",
            parent_id=None,
            now=ts1,
        )
        rev2 = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="v2",
            parent_id=rev1.id,
            now=ts2,
        )
        self.assertEqual(rev2.parent_id, rev1.id)

    def test_rejects_blank_value(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        with self.assertRaises(ValueError):
            build_revision(
                scene_number=3,
                field=PromptField.Z_IMAGE_PROMPT,
                value="",
                parent_id=None,
                now=ts,
            )

    def test_rejects_whitespace_only_value(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        with self.assertRaises(ValueError):
            build_revision(
                scene_number=3,
                field=PromptField.Z_IMAGE_PROMPT,
                value="   \n  ",
                parent_id=None,
                now=ts,
            )

    def test_rejects_unknown_field(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        with self.assertRaises(ValueError):
            build_revision(
                scene_number=3,
                field="nonexistent_field",
                value="A prompt",
                parent_id=None,
                now=ts,
            )

    def test_rejects_negative_scene_number(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        with self.assertRaises(ValueError):
            build_revision(
                scene_number=-1,
                field=PromptField.Z_IMAGE_PROMPT,
                value="A prompt",
                parent_id=None,
                now=ts,
            )

    def test_revision_is_frozen(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        rev = build_revision(
            scene_number=3,
            field=PromptField.Z_IMAGE_PROMPT,
            value="A singer on stage",
            parent_id=None,
            now=ts,
        )
        with self.assertRaises(Exception):
            rev.value = "mutated"  # type: ignore

    def test_deterministic_id_given_same_inputs(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        rev1 = build_revision(
            scene_number=5,
            field=PromptField.I2V_PROMPT,
            value="deterministic prompt",
            parent_id=None,
            now=ts,
        )
        rev2 = build_revision(
            scene_number=5,
            field=PromptField.I2V_PROMPT,
            value="deterministic prompt",
            parent_id=None,
            now=ts,
        )
        self.assertEqual(rev1.id, rev2.id)


class RestoreRevisionTests(unittest.TestCase):
    def test_restore_creates_new_revision_instead_of_rewriting_history(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        ts3 = datetime.datetime(2026, 7, 21, 10, 2, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v2", parent_id=r1.id, now=ts2)
        r3 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v3", parent_id=r2.id, now=ts3)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1, r2, r3])

        restored = restore_revision(history, revision_id=r1.id, now=ts3.replace(minute=3))

        self.assertEqual(r1.id, restored.restored_from)
        self.assertNotEqual(r1.id, restored.id)
        self.assertEqual(len(history.revisions) + 1, len((*history.revisions, restored)))

    def test_restore_value_matches_original(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="original text", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="changed", parent_id=r1.id, now=ts2)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1, r2])

        restored = restore_revision(history, revision_id=r1.id, now=ts2.replace(minute=2))

        self.assertEqual("original text", restored.value)

    def test_restore_parent_is_current_tip(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        ts3 = datetime.datetime(2026, 7, 21, 10, 2, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v2", parent_id=r1.id, now=ts2)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1, r2])

        restored = restore_revision(history, revision_id=r1.id, now=ts3)

        # The parent should be the current tip (r2), not r1
        self.assertEqual(r2.id, restored.parent_id)

    def test_restore_with_parent_rewrites_parent(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        ts3 = datetime.datetime(2026, 7, 21, 10, 2, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v2", parent_id=r1.id, now=ts2)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1, r2])

        restored = restore_revision(history, revision_id=r1.id, now=ts3, with_parent=True)

        self.assertEqual(r1.id, restored.parent_id)

    def test_restore_invalid_id_raises(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1", parent_id=None, now=ts1)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1])

        with self.assertRaises(ValueError):
            restore_revision(history, revision_id="nonexistent", now=ts1)

    def test_restore_empty_history_raises(self):
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[])

        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        with self.assertRaises(ValueError):
            restore_revision(history, revision_id="r1", now=ts)


class PromptHistoryTests(unittest.TestCase):
    def test_chronological_order(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        ts3 = datetime.datetime(2026, 7, 21, 10, 2, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v2", parent_id=r1.id, now=ts2)
        r3 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v3", parent_id=r2.id, now=ts3)

        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r3, r1, r2])
        self.assertEqual(len(history.revisions), 3)
        self.assertEqual([rev.value for rev in history.revisions], ["v1", "v2", "v3"])

    def test_latest_value(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v1", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="v2", parent_id=r1.id, now=ts2)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1, r2])

        self.assertEqual("v2", history.latest_value)

    def test_empty_history_latest_value_none(self):
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[])
        self.assertIsNone(history.latest_value)

    def test_diff_with_previous(self):
        ts1 = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2026, 7, 21, 10, 1, 0, tzinfo=datetime.timezone.utc)
        ts3 = datetime.datetime(2026, 7, 21, 10, 2, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="hello world", parent_id=None, now=ts1)
        r2 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="hello beautiful world", parent_id=r1.id, now=ts2)
        r3 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="goodbye beautiful world", parent_id=r2.id, now=ts3)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1, r2, r3])

        diff = history.diff_with_previous(r3.id)
        self.assertIsNotNone(diff)
        self.assertIn("hello", diff)
        self.assertIn("goodbye", diff)

    def test_diff_first_revision_is_none(self):
        ts = datetime.datetime(2026, 7, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)
        r1 = build_revision(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, value="first", parent_id=None, now=ts)
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[r1])

        self.assertIsNone(history.diff_with_previous(r1.id))

    def test_diff_invalid_id_raises(self):
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[])
        with self.assertRaises(ValueError):
            history.diff_with_previous("nope")

    def test_history_is_frozen(self):
        history = PromptHistory(scene_number=3, field=PromptField.Z_IMAGE_PROMPT, revisions=[])
        with self.assertRaises(Exception):
            history.revisions = []  # type: ignore


class PromptFieldEnumTests(unittest.TestCase):
    def test_known_fields(self):
        self.assertTrue(hasattr(PromptField, "Z_IMAGE_PROMPT"))
        self.assertTrue(hasattr(PromptField, "I2V_PROMPT"))

    def test_field_value_is_string(self):
        self.assertIsInstance(PromptField.Z_IMAGE_PROMPT.value, str)
        self.assertIsInstance(PromptField.I2V_PROMPT.value, str)


if __name__ == "__main__":
    unittest.main()
