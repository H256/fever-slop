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
    allow_private_addresses: bool = True,
) -> tuple[str, str | None]:
    """Validate an HTTP API endpoint before it is used for a request.

    The URL must use ``http`` or ``https`` and may not contain embedded
    credentials, a query string, or a fragment.

    By default (``allow_private_addresses=True``) no address filtering is
    applied: local-network and loopback services are the normal deployment
    model. With ``allow_private_addresses=False`` (strict mode), literal
    private, link-local, multicast, reserved, and unspecified addresses are
    rejected; non-literal hostnames are resolved via DNS, and when DNS
    returns addresses, every resolved address is checked the same way while
    unresolvable hostnames are left to the HTTP client. Loopback stays
    available in both modes while ``allow_loopback`` is true.

    An allowlist (``allowed_hosts`` or the comma-separated, case-insensitive
    ``FEVERSLOP_ALLOWED_API_HOSTS`` environment variable when no explicit
    allowlist is given) restricts hostnames in both modes. A host that is on
    the allowlist is explicitly trusted and skips the address checks
    entirely, including in strict mode.

    Returns ``(normalized_url, pinned_ip)`` where ``pinned_ip`` is the
    first resolved IP address when the host is a non-literal hostname that
    resolves successfully in strict mode, and ``None`` otherwise. Callers
    should use ``pinned_ip`` to pin the TCP connection to the validated
    address, closing the DNS-rebinding TOCTOU window between validation and
    the actual request.

    Transparent-proxying limitation: if the HTTP client is configured to use
    a transparent proxy that re-resolves the hostname independently of the
    pinned IP, the pinning guarantee does not extend to the proxy hop.
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

    # Local-network services are the default deployment model. Strict private
    # address filtering is opt-in for deployments that need SSRF protection.
    # pinned_ip is set only for non-literal hostnames that resolve in strict
    # mode; a literal IP needs no pinning because the client connects to it
    # directly without re-resolving DNS.
    pinned_ip: str | None = None
    if allow_private_addresses or explicitly_allowed:
        return value, None

    if normalized_host == "localhost" and allow_loopback:
        return value, None
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(normalized_host, port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror:
            # Defer DNS failure handling to the HTTP client; do not turn a
            # temporarily unavailable external service into a config error.
            return value, None
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
                    f"API URL hostname resolves to a private or reserved address: {normalized_host}",
                )
        pinned_ip = str(resolved[0][4][0])
        return value, pinned_ip
    if address.is_loopback and allow_loopback:
        return value, None
    if address.is_private or address.is_link_local or address.is_multicast or address.is_reserved or address.is_unspecified:
        raise APIURLValidationError(f"API URL targets a private or reserved address: {normalized_host}")
    return value, None
