import unittest
from pathlib import Path


class JobSupportCompositionTests(unittest.TestCase):
    def test_log_helper_legacy_import_uses_canonical_module(self):
        from feverslop.composition.logging import render_log_lines
        from feverslop.studio.logging import render_log_lines as legacy_render_log_lines

        self.assertIs(legacy_render_log_lines, render_log_lines)

    def test_pipeline_helper_legacy_import_uses_canonical_module(self):
        from feverslop.composition.pipeline_actions import pipeline_action_availability
        from feverslop.studio.pipeline_actions import pipeline_action_availability as legacy_pipeline_action_availability

        self.assertIs(legacy_pipeline_action_availability, pipeline_action_availability)

    def test_canonical_helpers_do_not_import_studio_package(self):
        for module_name in ("logging", "pipeline_actions"):
            source = Path(f"src/feverslop/composition/{module_name}.py").read_text(encoding="utf-8")
            self.assertNotIn("feverslop.studio", source)

