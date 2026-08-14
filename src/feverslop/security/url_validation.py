from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit


class APIURLValidationError(ValueError):
    """Raised when an external API endpoint is unsafe or malformed."""


def validate_api_url(
    url: str,
    *,
    allowed_hosts: set[str] | None = None,
    allow_loopback: bool = True,
) -> str:
    """Validate an HTTP API endpoint before it is used for a request.

    Literal private, link-local, multicast and reserved addresses are rejected
    to prevent accidental SSRF. Local loopback endpoints remain available for
    the default ComfyUI/LLM services. ``FEVERSLOP_ALLOWED_API_HOSTS`` can be
    used to restrict hostnames further (comma-separated, case-insensitive).
    """
    if not isinstance(url, str) or not url.strip():
        raise APIURLValidationError("API URL must be a non-empty string")

    value = url.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise APIURLValidationError("API URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise APIURLValidationError("API URL must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise APIURLValidationError("API base URL must not contain a query or fragment")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise APIURLValidationError("API URL has an invalid host or port") from exc
    if not hostname:
        raise APIURLValidationError("API URL must contain a hostname")
    if port is not None and not 1 <= port <= 65535:
        raise APIURLValidationError("API URL port must be between 1 and 65535")

    normalized_host = hostname.rstrip(".").lower()
    configured_hosts = allowed_hosts
    if configured_hosts is None:
        raw_hosts = os.environ.get("FEVERSLOP_ALLOWED_API_HOSTS", "")
        configured_hosts = {item.strip().lower().rstrip(".") for item in raw_hosts.split(",") if item.strip()}
    if configured_hosts and normalized_host not in configured_hosts:
        raise APIURLValidationError(f"API host is not in the allowlist: {normalized_host}")
    explicitly_allowed = bool(configured_hosts and normalized_host in configured_hosts)

    # A configured host is an explicit trust decision. This is required for
    # local deployments that address ComfyUI or an LLM over a private LAN.
    if explicitly_allowed:
        return value

    if normalized_host == "localhost" and allow_loopback:
        return value
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(normalized_host, port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror:
            # Defer DNS failure handling to the HTTP client; do not turn a
            # temporarily unavailable external service into a config error.
            return value
        for result in resolved:
            resolved_address = ipaddress.ip_address(result[4][0])
            if resolved_address.is_loopback and allow_loopback:
                continue
            if (
                resolved_address.is_private
                or resolved_address.is_link_local
                or resolved_address.is_multicast
                or resolved_address.is_reserved
                or resolved_address.is_unspecified
            ):
                raise APIURLValidationError(
                    f"API URL hostname resolves to a private or reserved address: {normalized_host}"
                )
        return value
    if address.is_loopback and allow_loopback:
        return value
    if address.is_private or address.is_link_local or address.is_multicast or address.is_reserved or address.is_unspecified:
        raise APIURLValidationError(f"API URL targets a private or reserved address: {normalized_host}")
    return value
