import unittest

from feverslop.domain.artifact_hash import is_sha256_hex


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
