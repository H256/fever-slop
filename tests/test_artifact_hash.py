import hashlib
import json
import unittest

from feverslop.domain.artifact_hash import fingerprint_json, is_sha256_hex


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


class Sha256HexTests(unittest.TestCase):
    def test_accepts_exact_lowercase_sha256_hex(self):
        self.assertTrue(is_sha256_hex("0123456789abcdef" * 4))

    def test_rejects_invalid_length(self):
        for value in ("", "a" * 63, "a" * 65):
            with self.subTest(value=value):
                self.assertFalse(is_sha256_hex(value))

    def test_rejects_uppercase(self):
        self.assertFalse(is_sha256_hex("A" + "a" * 63))

    def test_rejects_non_hex(self):
        self.assertFalse(is_sha256_hex("g" + "a" * 63))

    def test_rejects_non_string_values(self):
        self.assertFalse(is_sha256_hex(None))


if __name__ == "__main__":
    unittest.main()
