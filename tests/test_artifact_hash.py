import hashlib
import json
import unittest

from feverslop.domain.artifact_hash import fingerprint_json


class FingerprintJsonTests(unittest.TestCase):
    def test_defaults_to_utf8_json_without_ascii_escaping(self):
        value = {"text": "café 東京", "nested": {"clé": "値"}}
        expected = hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        ).hexdigest()

        self.assertEqual(expected, fingerprint_json(value))

    def test_explicit_ascii_escaping_changes_json_bytes(self):
        value = {"text": "café 東京", "nested": {"clé": "値"}}
        expected = hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        ).hexdigest()

        self.assertEqual(expected, fingerprint_json(value, ensure_ascii=True))
        self.assertNotEqual(
            fingerprint_json(value),
            fingerprint_json(value, ensure_ascii=True),
        )


if __name__ == "__main__":
    unittest.main()
