import unittest

from feverslop.domain.ltx25_migration import diagnose_and_migrate_config


class LTX25MigrationTests(unittest.TestCase):
    def test_migration_is_copy_based_and_selects_mode(self):
        source = {"video_pipeline": "ltx_msr", "render_profile": {"quality": "draft"}}
        migrated, report = diagnose_and_migrate_config(source)
        self.assertEqual("ltx25-msr-draft", migrated["render_profile"])
        self.assertEqual({"quality": "draft"}, source["render_profile"])
        self.assertTrue(report.changed)

    def test_unknown_pipeline_fails_closed(self):
        with self.assertRaises(ValueError):
            diagnose_and_migrate_config({"video_pipeline": "ltx_23"})


if __name__ == "__main__":
    unittest.main()
