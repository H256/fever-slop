"""IP pinning helpers for DNS-rebinding protection.

When ``validate_api_url`` resolves a hostname, the resulting IP is returned
alongside the validated URL.  These helpers pin that IP at connect time,
closing the TOCTOU window between validation and connection.
"""

from __future__ import annotations

import socket
import typing

from httpcore._backends.sync import SyncBackend

if typing.TYPE_CHECKING:
    import httpx
    import requests


class PinnedNetworkBackend(SyncBackend):
    """httpcore NetworkBackend that pins the target IP.

    Replaces the hostname in :meth:`connect_tcp` with the validated IP,
    preventing DNS rebinding between validation and connection.
    """

    def __init__(self, pinned_ip: str) -> None:
        self._pinned_ip = pinned_ip

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[typing.Any] | None = None,
    ) -> typing.Any:
        return super().connect_tcp(
            self._pinned_ip,
            port,
            timeout,
            local_address,
            socket_options,
        )


def create_pinned_httpx_client(
    pinned_ip: str | None,
    **kwargs: typing.Any,
) -> "httpx.Client":
    """Create an ``httpx.Client`` with IP pinning if a pinned IP is provided."""
    import httpx

    client = httpx.Client(**kwargs)
    if pinned_ip:
        transport = client._transport
        assert isinstance(transport, httpx.HTTPTransport)
        transport._pool._network_backend = PinnedNetworkBackend(pinned_ip)
    return client


def create_pinned_requests_session(
    pinned_ip: str | None,
    *,
    pool_connections: int = 10,
    pool_maxsize: int = 10,
    pool_block: bool = False,
) -> "requests.Session":
    """Create a ``requests.Session`` with IP pinning if a pinned IP is provided."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3 import PoolManager
    from urllib3.connection import HTTPConnection, HTTPSConnection

    if pinned_ip:

        class PinnedHTTPConnection(HTTPConnection):
            def __init__(
                self,
                *args: typing.Any,
                _pinned_ip: str | None = None,
                **kwargs: typing.Any,
            ) -> None:
                super().__init__(*args, **kwargs)
                self._pinned_ip = _pinned_ip

            def _new_conn(self) -> socket.socket:
                if self._pinned_ip:
                    original_host = self._dns_host
                    self._dns_host = self._pinned_ip
                    try:
                        return super()._new_conn()
                    finally:
                        self._dns_host = original_host
                return super()._new_conn()

        class PinnedHTTPSConnection(HTTPSConnection):
            def __init__(
                self,
                *args: typing.Any,
                _pinned_ip: str | None = None,
                **kwargs: typing.Any,
            ) -> None:
                super().__init__(*args, **kwargs)
                self._pinned_ip = _pinned_ip

            def _new_conn(self) -> socket.socket:
                if self._pinned_ip:
                    original_host = self._dns_host
                    self._dns_host = self._pinned_ip
                    try:
                        return super()._new_conn()
                    finally:
                        self._dns_host = original_host
                return super()._new_conn()

        class PinnedPoolManager(PoolManager):
            def __init__(self, pinned_ip: str, **kwargs: typing.Any) -> None:
                super().__init__(**kwargs)
                self._pinned_ip = pinned_ip

            def _new_pool(
                self,
                scheme: str,
                host: str,
                port: int,
                request_context: typing.Any = None,
            ) -> typing.Any:
                pool = super()._new_pool(scheme, host, port, request_context)
                pool.conn_kw["_pinned_ip"] = self._pinned_ip
                pool.ConnectionCls = (  # type: ignore[assignment]
                    PinnedHTTPSConnection if scheme == "https" else PinnedHTTPConnection
                )
                return pool

        class PinnedHTTPAdapter(HTTPAdapter):
            def init_poolmanager(
                self,
                connections: int,
                maxsize: int,
                block: bool = False,
                **pool_kwargs: typing.Any,
            ) -> None:
                pool_kwargs.pop("connection_pool_kw", None)
                self.poolmanager = PinnedPoolManager(
                    pinned_ip,
                    num_pools=connections,
                    **pool_kwargs,
                )

        adapter = PinnedHTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            pool_block=pool_block,
        )
    else:
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            pool_block=pool_block,
        )

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
