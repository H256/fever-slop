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

    def test_allowlist_can_explicitly_trust_private_local_service(self):
        with patch.dict(os.environ, {"FEVERSLOP_ALLOWED_API_HOSTS": "192.168.1.10"}):
            self.assertEqual(
                "http://192.168.1.10:8188",
                validate_api_url("http://192.168.1.10:8188"),
            )

    def test_rejects_query_and_fragment_on_base_url(self):
        for suffix in ("?token=secret", "#fragment"):
            with self.subTest(suffix=suffix), self.assertRaises(APIURLValidationError):
                validate_api_url("https://api.example/v1" + suffix)

    @patch("feverslop.security.url_validation.socket.getaddrinfo")
    def test_rejects_hostname_resolving_to_private_address(self, getaddrinfo):
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.10", 8080)),
        ]

        with self.assertRaisesRegex(APIURLValidationError, "private"):
            validate_api_url("https://service.example:8080")

    @patch("feverslop.security.url_validation.socket.getaddrinfo")
    def test_allows_hostname_resolving_to_public_address(self, getaddrinfo):
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]

        self.assertEqual(
            "https://service.example",
            validate_api_url("https://service.example"),
        )
