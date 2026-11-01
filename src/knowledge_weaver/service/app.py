"""KW service: HTTP/1.1 over an owner-only Unix domain socket (T02).

* No TCP listener: the default transport is a Unix socket in a 0700 directory
  with 0600 permissions.
* The request shape is validated whole before any part is committed; per-part
  outcomes are durable receipts (see :mod:`knowledge_weaver.store.events`).
* Responses follow the frozen contract schemas; errors use the stable catalog.
* No model call, no network access and no body logging happen in this layer.
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import stat
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from knowledge_weaver import contracts
from knowledge_weaver.store import (
    Store,
    StoreRejection,
    fetch_source,
    health,
    receive_batch,
    status as status_view,
)

from .errors import ApiError, error_payload
from .validation import (
    authorize_batch,
    authorize_scope,
    resolve_binding,
)

MAX_REQUEST_BYTES = contracts.MAX_BATCH_BYTES
_REQUEST_ID_RE = contracts.TOKEN_RE
_EVENT_PATH_PREFIX = "/v1/sources/"


class _ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def _new_request_id() -> str:
    return "req:" + uuid.uuid4().hex[:24]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "knowledge-weaver/2.0"
    sys_version = ""

    # -- plumbing -----------------------------------------------------------
    @property
    def app(self):
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # pragma: no cover - no body/peer logging
        return

    def do_GET(self):  # noqa: N802 - http.server API
        self._handle("GET")

    def do_POST(self):  # noqa: N802 - http.server API
        self._handle("POST")

    # -- request handling ---------------------------------------------------
    def _request_id(self) -> str:
        candidate = self.headers.get("X-Request-Id")
        if isinstance(candidate, str) and _REQUEST_ID_RE.match(candidate) and len(candidate) <= 180:
            return candidate
        return _new_request_id()

    def _handle(self, method: str) -> None:
        request_id = self._request_id()
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            if method == "GET":
                result = self._get(path, parsed.query)
            else:
                result = self._post(path)
            self._send_json(200, result, request_id)
        except ApiError as exc:
            self._send_json(exc.http_status, error_payload(exc.code, exc.message, request_id), request_id)
        except StoreRejection as exc:
            error = ApiError(exc.code, exc.message)
            self._send_json(error.http_status, error_payload(error.code, error.message, request_id), request_id)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client gone
            self.close_connection = True
        except Exception:  # noqa: BLE001 - never leak backend details
            self._send_json(
                500,
                error_payload("INTERNAL_ERROR", "internal error; request not processed", request_id),
                request_id,
            )

    def _get(self, path: str, query: str) -> dict:
        if path == "/v1/health":
            return health(self.app.store)
        if path == "/v1/status":
            return status_view(self.app.store)
        if path.startswith(_EVENT_PATH_PREFIX):
            return self._get_source(path[len(_EVENT_PATH_PREFIX) :], query)
        if path == "/v1/capabilities":
            client_id = self.headers.get("X-DMA-Client")
            binding = resolve_binding(self.app.store, client_id)
            from knowledge_weaver.resource_registry import list_capabilities

            return {
                "items": list_capabilities(
                    self.app.store, workspace_id=binding["workspace_id"]
                )
            }
        if path == "/v1/resources":
            client_id = self.headers.get("X-DMA-Client")
            binding = resolve_binding(self.app.store, client_id)
            from knowledge_weaver.resource_registry import list_resources

            projects = list(binding.get("project_ids") or []) or None
            return {
                "items": list_resources(
                    self.app.store,
                    workspace_id=binding["workspace_id"],
                    project_ids=projects,
                )
            }
        raise ApiError("INVALID_REQUEST", "unknown route or method")

    def _post(self, path: str) -> dict:
        if path == "/v1/events:batch":
            body = self._read_json()
            client_id = self.headers.get("X-DMA-Client")
            binding = resolve_binding(self.app.store, client_id)
            authorize_batch(binding, body)
            return receive_batch(self.app.store, body, secret_check=self.app.secret_check)
        if path == "/v1/corrections":
            body = self._read_json()
            client_id = self.headers.get("X-DMA-Client")
            binding = resolve_binding(self.app.store, client_id)
            claim_id = body.get("claim_id") if isinstance(body, dict) else None
            if isinstance(claim_id, str):
                # F21: authorize the persisted target claim *and* every evidence
                # reference — workspace and project — before applying anything,
                # instead of trusting the request body or the project name alone.
                target = self.app.claim_scope(claim_id)
                if target is not None:
                    authorize_scope(
                        binding,
                        workspace_id=target["workspace_id"],
                        project_ids=[target["project_id"]] if target["project_id"] else [],
                    )
                    for reference in body.get("evidence") or []:
                        if not isinstance(reference, dict):
                            continue
                        try:
                            evidence_revision = int(reference.get("event_revision") or 0)
                        except (TypeError, ValueError):
                            continue  # shape error is reported by apply_correction
                        scope = self.app.source_scope(
                            reference.get("event_id"), evidence_revision
                        )
                        if scope is None:
                            # Unknown/missing references keep the store's
                            # pending_evidence semantics; no scope to check.
                            continue
                        authorize_scope(
                            binding,
                            workspace_id=scope["workspace_id"],
                            project_ids=scope["project_ids"],
                        )
                        if (
                            target["project_id"] is not None
                            and scope["project_ids"]
                            and target["project_id"] not in scope["project_ids"]
                        ):
                            raise ApiError(
                                "SCOPE_DENIED",
                                "correction evidence does not belong to the target project",
                            )
            try:
                return self.app.apply_correction(body)
            except Exception as exc:  # ClaimError is a plain exception in the store layer
                from knowledge_weaver.store import ClaimError

                if isinstance(exc, ClaimError):
                    raise ApiError(exc.code, exc.message) from None
                raise
        if path == "/v1/recall":
            body = self._read_json()
            client_id = self.headers.get("X-DMA-Client")
            binding = resolve_binding(self.app.store, client_id)
            from knowledge_weaver.retrieval import (
                RecallCursorError,
                RecallScopeDenied,
                RecallValidationError,
                execute_recall,
            )

            try:
                return execute_recall(
                    self.app.store, body, binding=binding, embedder=self.app.embedder
                )
            except RecallScopeDenied as exc:
                raise ApiError("SCOPE_DENIED", exc.message) from None
            except RecallCursorError as exc:
                raise ApiError("STALE_CURSOR", exc.message) from None
            except RecallValidationError as exc:
                raise ApiError("INVALID_REQUEST", exc.message) from None
        raise ApiError("INVALID_REQUEST", "unknown route or method")

    def _get_source(self, raw_event_id: str, query: str) -> dict:
        event_id = raw_event_id
        if not contracts.EVENT_ID_RE.match(event_id):
            raise ApiError("INVALID_REQUEST", "source id must be an opaque evt:<sha256> id")
        params = parse_qs(query, keep_blank_values=True)
        revision_values = params.get("revision", [])
        if len(revision_values) != 1 or not revision_values[0].isdigit():
            raise ApiError("INVALID_REQUEST", "source reads require an explicit revision")
        revision = int(revision_values[0])
        if revision < 1:
            raise ApiError("INVALID_REQUEST", "revision must be >= 1")
        part_values = params.get("part", ["0"])
        if len(part_values) != 1 or not part_values[0].isdigit():
            raise ApiError("INVALID_REQUEST", "part must be a non-negative integer")
        part = int(part_values[0])
        result = fetch_source(self.app.store, event_id, revision, part)
        if result is None:
            raise ApiError("SOURCE_NOT_FOUND", "source revision or part not found")
        client_id = self.headers.get("X-DMA-Client")
        binding = resolve_binding(self.app.store, client_id)
        # F21: scope check against the stored source envelope — workspace *and*
        # project — never against body text and never only against the workspace.
        scope = self.app.source_scope(event_id, revision)
        authorize_scope(
            binding,
            workspace_id=scope["workspace_id"] if scope else None,
            project_ids=scope["project_ids"] if scope else [],
        )
        return result

    # -- body/response ------------------------------------------------------
    def _read_json(self) -> dict:
        length_header = self.headers.get("Content-Length")
        if length_header is None or not str(length_header).isdigit():
            raise ApiError("INVALID_REQUEST", "Content-Length is required")
        length = int(length_header)
        if length > MAX_REQUEST_BYTES:
            raise ApiError("PAYLOAD_TOO_LARGE", f"request body exceeds {MAX_REQUEST_BYTES} bytes")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ApiError("INVALID_REQUEST", "incomplete request body")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError("INVALID_REQUEST", "request body must be valid UTF-8 JSON") from None
        if not isinstance(value, dict):
            raise ApiError("INVALID_REQUEST", "request body must be a JSON object")
        return value

    def _send_json(self, status_code: int, payload: dict, request_id: str) -> None:
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):  # pragma: no cover - defensive
            status_code = 500
            encoded = json.dumps(
                error_payload("INTERNAL_ERROR", "internal serialization error", request_id),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("X-Request-Id", request_id)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.close_connection = True


class ServiceApp:
    """Owns a KW store and a Unix-socket HTTP server.

    ``start()`` prepares the socket directory (0700), binds the socket (0600)
    and serves in a background thread. ``stop()`` shuts down and removes the
    socket. ``serve_forever()`` is the blocking variant used by the CLI.
    """

    def __init__(
        self,
        store: Store,
        socket_path: str | Path,
        *,
        secret_check: Callable[[str], bool] | None = None,
        embedder=None,
    ) -> None:
        self.store = store
        self.socket_path = Path(socket_path)
        self.embedder = embedder
        if secret_check is None:
            from .validation.redaction import contains_secret

            secret_check = contains_secret
        self.secret_check = secret_check
        self._server: _ThreadingUnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    def claim_workspace(self, claim_id: str) -> str | None:
        with self.store.read() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            return row["workspace_id"] if row is not None else None

    def apply_correction(self, correction: dict) -> dict:
        from knowledge_weaver.store.claims import apply_correction

        return apply_correction(self.store, correction)

    def workspace_of(self, event_id: str) -> str | None:
        with self.store.read() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return row["workspace_id"] if row is not None else None

    def source_scope(self, event_id, revision) -> dict | None:
        """Workspace + project ids of a stored source revision (for authorization)."""
        if not isinstance(event_id, str) or not isinstance(revision, int):
            return None
        with self.store.read() as conn:
            event = conn.execute(
                "SELECT workspace_id FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if event is None:
                return None
            row = conn.execute(
                "SELECT project_ids_json FROM event_revisions WHERE event_id = ? AND revision = ?",
                (event_id, int(revision)),
            ).fetchone()
        import json as _json

        return {
            "workspace_id": event["workspace_id"],
            "project_ids": list(_json.loads(row["project_ids_json"] or "[]")) if row else [],
        }

    def claim_scope(self, claim_id: str) -> dict | None:
        """Workspace + project of a stored claim (for authorization)."""
        with self.store.read() as conn:
            row = conn.execute(
                "SELECT workspace_id, project_id FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
        if row is None:
            return None
        return {"workspace_id": row["workspace_id"], "project_id": row["project_id"]}

    def claim_workspace(self, claim_id: str) -> str | None:
        scope = self.claim_scope(claim_id)
        return scope["workspace_id"] if scope else None

    def _prepare_socket(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.socket_path.parent, 0o700)
        except OSError:  # pragma: no cover - platform dependent
            pass
        if self.socket_path.exists():
            info = self.socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode):
                raise RuntimeError(f"refusing to replace non-socket path: {self.socket_path}")
            # A stale test socket may be removed only if nothing is listening.
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.5)
                probe.connect(str(self.socket_path))
                raise RuntimeError(f"socket already in use: {self.socket_path}")
            except (ConnectionRefusedError, FileNotFoundError, socket.timeout):
                self.socket_path.unlink(missing_ok=True)
            finally:
                probe.close()

    def _make_server(self) -> _ThreadingUnixStreamServer:
        self._prepare_socket()
        server = _ThreadingUnixStreamServer(str(self.socket_path), _Handler)
        server.app = self  # type: ignore[attr-defined]
        try:
            os.chmod(self.socket_path, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
        return server

    def start(self, ready_timeout: float = 10.0) -> "ServiceApp":
        self._server = self._make_server()
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="kw-service", daemon=True
        )
        self._thread.start()
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if self.socket_path.exists():
                return self
            time.sleep(0.01)
        raise RuntimeError("service socket did not appear")

    def serve_forever(self) -> None:
        self._server = self._make_server()
        try:
            self._server.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.socket_path.unlink(missing_ok=True)

    def __enter__(self) -> "ServiceApp":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()
