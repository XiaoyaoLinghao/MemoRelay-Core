"""Local client bindings (X-DMA-Client) for the KW service.

A binding is a trusted local installation registration, not a cryptographic
identity: it maps an install to a workspace/host/instance and an allow-list of
actors and projects. Body text can never change it (SRC-01). The same-OS-user
limitation is documented in the contract; this is not a hostile-tenant boundary.
"""

from __future__ import annotations

import json
from typing import Iterable

from knowledge_weaver import contracts

from .db import Store, utc_now


class BindingError(ValueError):
    """Raised for malformed binding registration input."""


def _token(value: str, field: str) -> str:
    if not isinstance(value, str) or not contracts.TOKEN_RE.match(value) or len(value) > 180:
        raise BindingError(f"{field} must be a short token ([A-Za-z0-9._:-]+)")
    return value


def register_client(
    store: Store,
    *,
    client_id: str,
    workspace_id: str,
    host: str,
    instance_id: str,
    actor_ids: Iterable[str] = ("*",),
    project_ids: Iterable[str] = (),
    binding_version: int = 1,
) -> dict:
    """Create or replace a local client binding.

    ``project_ids`` empty means "all projects in this workspace"; a non-empty
    list restricts the binding to that subset. ``actor_ids`` containing ``"*"``
    allows every actor originating from the registered instance.
    """
    client_id = _token(client_id, "client_id")
    workspace_id = _token(workspace_id, "workspace_id")
    instance_id = _token(instance_id, "instance_id")
    host = _token(host, "host")
    actor_list = ["*" if actor == "*" else _token(actor, "actor_id") for actor in actor_ids]
    project_list = [_token(project, "project_id") for project in project_ids]
    if not actor_list:
        raise BindingError("actor_ids must not be empty")
    if not isinstance(binding_version, int) or binding_version < 1:
        raise BindingError("binding_version must be a positive integer")
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO clients "
            "(client_id, workspace_id, host, instance_id, actor_ids_json, project_ids_json, binding_version, registered_at, revoked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL) "
            "ON CONFLICT (client_id) DO UPDATE SET "
            "workspace_id=excluded.workspace_id, host=excluded.host, instance_id=excluded.instance_id, "
            "actor_ids_json=excluded.actor_ids_json, project_ids_json=excluded.project_ids_json, "
            "binding_version=excluded.binding_version, registered_at=excluded.registered_at, revoked_at=NULL",
            (
                client_id,
                workspace_id,
                host,
                instance_id,
                json.dumps(actor_list, ensure_ascii=False),
                json.dumps(project_list, ensure_ascii=False),
                binding_version,
                utc_now(),
            ),
        )
    return get_client(store, client_id)  # type: ignore[return-value]


def get_client(store: Store, client_id: str) -> dict | None:
    with store.read() as conn:
        row = conn.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,)).fetchone()
    if row is None:
        return None
    return {
        "client_id": row["client_id"],
        "workspace_id": row["workspace_id"],
        "host": row["host"],
        "instance_id": row["instance_id"],
        "actor_ids": json.loads(row["actor_ids_json"]),
        "project_ids": json.loads(row["project_ids_json"]),
        "binding_version": row["binding_version"],
        "revoked": row["revoked_at"] is not None,
    }


def revoke_client(store: Store, client_id: str) -> bool:
    with store.transaction() as conn:
        cursor = conn.execute(
            "UPDATE clients SET revoked_at = ? WHERE client_id = ? AND revoked_at IS NULL",
            (utc_now(), client_id),
        )
        return cursor.rowcount > 0
