import unittest


class AuditToolsTests(unittest.TestCase):
    def test_reference_builder_has_public_api(self):
        from feverslop.prompting.h3_prompt_builder import build_references_from_segment

        result = build_references_from_segment({"ref_items": []})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
