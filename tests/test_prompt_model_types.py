import unittest
from dataclasses import FrozenInstanceError
from types import MappingProxyType

from feverslop.prompting.dspy_h3_models import PromptMode
from feverslop.prompting.model_types import (
    MODEL_TYPES,
    ModelTypeSpec,
    resolve_model_type,
)


class PromptModelTypesTests(unittest.TestCase):
    def test_supported_model_types_resolve_to_minimax_h3_guides(self):
        expected = {
            "minimax-h3-t2v": (PromptMode.T2V, "minimax-h3-base.md"),
            "minimax-h3-i2v": (PromptMode.I2V, "minimax-h3-base.md"),
            "minimax-h3-fl2v": (PromptMode.FL2V, "minimax-h3-base.md"),
            "minimax-h3-l2v": (PromptMode.L2V, "minimax-h3-base.md"),
            "minimax-h3-r2v": (PromptMode.R2V, "minimax-h3-references.md"),
        }

        for model_type, (mode, guide_filename) in expected.items():
            with self.subTest(model_type=model_type):
                spec = resolve_model_type(model_type)
                self.assertEqual(model_type, spec.model_type)
                self.assertEqual(mode, spec.prompt_mode)
                self.assertTrue(spec.is_minimax_h3)
                self.assertEqual(guide_filename, spec.guide_filename)

    def test_lookup_normalizes_case_and_surrounding_whitespace(self):
        self.assertEqual(
            resolve_model_type("  MINIMAX-H3-R2V "),
            resolve_model_type("minimax-h3-r2v"),
        )

    def test_empty_and_unknown_model_types_are_rejected(self):
        accepted = ", ".join(MODEL_TYPES)
        for value in ("", "   ", "not-a-model"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, accepted):
                    resolve_model_type(value)

    def test_registry_and_specifications_are_immutable(self):
        self.assertIsInstance(MODEL_TYPES, MappingProxyType)
        with self.assertRaises(TypeError):
            MODEL_TYPES["custom"] = ModelTypeSpec(
                model_type="custom",
                prompt_mode=PromptMode.T2V,
                is_minimax_h3=False,
                guide_filename="custom.md",
            )

        with self.assertRaises(FrozenInstanceError):
            resolve_model_type("minimax-h3-t2v").guide_filename = "custom.md"


if __name__ == "__main__":
    unittest.main()
