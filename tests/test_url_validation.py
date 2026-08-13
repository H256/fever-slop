import os
import unittest
from unittest.mock import patch

from feverslop.security.url_validation import APIURLValidationError, validate_api_url


class APIURLValidationTests(unittest.TestCase):
    def test_allows_local_default_services(self):
        self.assertEqual("http://127.0.0.1:8188", validate_api_url("http://127.0.0.1:8188"))
        self.assertEqual("http://localhost:8080/v1", validate_api_url("http://localhost:8080/v1"))

    def test_rejects_unsafe_scheme_and_credentials(self):
        for url in ("file:///etc/passwd", "ftp://example.test", "http://user:secret@example.test"):
            with self.subTest(url=url), self.assertRaises(APIURLValidationError):
                validate_api_url(url)

    def test_rejects_literal_private_and_reserved_addresses(self):
        for address in ("10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.1.1", "0.0.0.0"):
            with self.subTest(address=address), self.assertRaises(APIURLValidationError):
                validate_api_url(f"http://{address}:8080")

    def test_allowlist_can_restrict_hostname(self):
        with patch.dict(os.environ, {"FEVERSLOP_ALLOWED_API_HOSTS": "llm.example, comfy.example"}):
            self.assertEqual("https://llm.example/v1", validate_api_url("https://llm.example/v1"))
            with self.assertRaisesRegex(APIURLValidationError, "allowlist"):
                validate_api_url("https://other.example/v1")

    def test_rejects_query_and_fragment_on_base_url(self):
        for suffix in ("?token=secret", "#fragment"):
            with self.subTest(suffix=suffix), self.assertRaises(APIURLValidationError):
                validate_api_url("https://api.example/v1" + suffix)
