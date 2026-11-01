"""Standard-library HTTP client for the KW Unix-socket service.

Used by tests and (from T03) by the DMA delivery worker; it never opens a TCP
connection and never logs bodies or credentials.
"""

from __future__ import annotations

import http.client
import json
import socket
from pathlib import Path


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str | Path, timeout: float = 10.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = str(socket_path)

    def connect(self) -> None:  # noqa: D401 - http.client API
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._socket_path)


class ServiceClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        client_id: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.socket_path = str(socket_path)
        self.client_id = client_id
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        encoded = None
        if body is not None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        return self.request_raw(
            method,
            path,
            encoded,
            headers={"Content-Type": "application/json"} if body is not None else None,
        )

    def request_raw(
        self,
        method: str,
        path: str,
        raw: bytes | None = None,
        headers: dict | None = None,
    ) -> tuple[int, dict]:
        connection = UnixHTTPConnection(self.socket_path, self.timeout)
        final_headers = {"Accept": "application/json"}
        if self.client_id:
            final_headers["X-DMA-Client"] = self.client_id
        if headers:
            final_headers.update(headers)
        try:
            connection.request(method, path, body=raw, headers=final_headers)
            response = connection.getresponse()
            raw = response.read()
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {"_raw": raw[:200].decode("utf-8", "replace")}
            return response.status, payload
        finally:
            connection.close()

    def health(self) -> tuple[int, dict]:
        return self.request("GET", "/v1/health")

    def status(self) -> tuple[int, dict]:
        return self.request("GET", "/v1/status")

    def post_batch(self, batch: dict) -> tuple[int, dict]:
        return self.request("POST", "/v1/events:batch", batch)

    def get_source(self, event_id: str, revision: int, part: int = 0) -> tuple[int, dict]:
        return self.request("GET", f"/v1/sources/{event_id}?revision={revision}&part={part}")
