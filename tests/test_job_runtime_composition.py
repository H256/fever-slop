import unittest
from pathlib import Path

from feverslop.studio.jobs import JobRegistry as LegacyJobRegistry


class JobRuntimeCompositionTests(unittest.TestCase):
    def test_legacy_import_uses_canonical_composition_runtime(self):
        from feverslop.composition.job_runtime import JobRegistry

        self.assertIs(LegacyJobRegistry, JobRegistry)

    def test_canonical_runtime_does_not_import_studio_package(self):
        source = Path("src/feverslop/composition/job_runtime.py").read_text(encoding="utf-8")

        self.assertNotIn("feverslop.studio", source)

