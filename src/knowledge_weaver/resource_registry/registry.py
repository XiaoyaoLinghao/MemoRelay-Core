"""Registry of capability sources and resource connection references (T15).

Boundaries (PRD §7, engineering §4):

* only non-secret references are stored: capability source/version/hash/locator/
  dependencies and resource endpoint/account label/``credential_ref``/verified
  recipe id/last verification state;
* credential values, private keys, cookies and tokens never enter this table;
  values that still match the shared secret policy are refused;
* there is **no** arbitrary execution or keychain-reading API: the host tool /
  OS authorization performs the actual access;
* recipe ids must reference a registered capability (verified reference, not a
  free-form command);
* entries are scoped by workspace (and project for resources) like everything
  else; hosts install and execute capabilities themselves.
"""

from __future__ import annotations

import json
import re

from knowledge_weaver import contracts
from knowledge_weaver.store.db import Store, utc_now

CAPABILITY_STATES = {"available", "missing", "incompatible", "unknown"}
RESOURCE_STATES = {"available", "needs_os_authorization", "unverified", "unavailable"}
_FORBIDDEN_KEYS = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|private[_-]?key|"
    r"cookie|authorization|credential_value|master[_-]?key)"
)
_HASH_RE = re.compile(r"^[a-f0-9]{64}$")


class RegistryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _token(value, field: str) -> str:
    if not isinstance(value, str) or not contracts.TOKEN_RE.match(value) or len(value) > 180:
        raise RegistryError("INVALID_REQUEST", f"{field} must be a short token")
    return value


def _assert_non_secret(value: str, field: str) -> None:
    if contracts.secret_kinds(value):
        raise RegistryError(
            "REDACTION_REQUIRED", f"{field} still contains secret-bearing content; store a reference instead"
        )


def _assert_contract(name: str, payload: dict) -> None:
    issues = contracts.validate_instance(name, payload)
    if issues:
        first = issues[0]
        raise RegistryError("INVALID_REQUEST", f"invalid {name} at {first.path}: {first.message}")


def register_capability(
    store: Store,
    *,
    workspace_id: str,
    actor_id: str,
    name: str,
    source_uri: str,
    version: str,
    content_hash: str,
    install_locator: str,
    dependencies=(),
    state: str = "unknown",
    verified_at: str | None = None,
) -> dict:
    workspace_id = _token(workspace_id, "workspace_id")
    actor_id = _token(actor_id, "actor_id")
    if state not in CAPABILITY_STATES:
        raise RegistryError("INVALID_REQUEST", f"unsupported capability state {state!r}")
    if not isinstance(content_hash, str) or not _HASH_RE.match(content_hash):
        raise RegistryError("INVALID_REQUEST", "content_hash must be a hex SHA-256")
    for field, value in (("source_uri", source_uri), ("install_locator", install_locator)):
        _assert_non_secret(value, field)
    dependency_list = []
    for dependency in dependencies:
        _assert_non_secret(str(dependency), "dependency")
        dependency_list.append(str(dependency))
    capability = {
        "capability_id": "cap:" + contracts.sha256_hex(
            contracts.compact_json([workspace_id, actor_id, name, source_uri, version, content_hash])
        )[:40],
        "actor_id": actor_id,
        "name": name,
        "source_uri": source_uri,
        "version": version,
        "content_hash": content_hash,
        "install_locator": install_locator,
        "dependencies": dependency_list,
        "state": state,
        "verified_at": verified_at,
    }
    _assert_contract("Capability", capability)
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO capabilities "
            "(capability_id, workspace_id, actor_id, name, source_uri, version, content_hash, "
            "install_locator, dependencies_json, state, verified_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                capability["capability_id"],
                workspace_id,
                actor_id,
                name,
                source_uri,
                version,
                content_hash,
                install_locator,
                json.dumps(dependency_list, ensure_ascii=False),
                state,
                verified_at,
                utc_now(),
            ),
        )
    return {"workspace_id": workspace_id, **capability}


def register_resource(
    store: Store,
    *,
    workspace_id: str,
    project_id: str | None,
    name: str,
    endpoint: str,
    account_label: str,
    credential_ref: str,
    recipe_id: str,
    last_verified_at: str | None = None,
    state: str = "unverified",
) -> dict:
    workspace_id = _token(workspace_id, "workspace_id")
    if project_id is not None:
        project_id = _token(project_id, "project_id")
    credential_ref = _token(credential_ref, "credential_ref")
    recipe_id = _token(recipe_id, "recipe_id")
    if state not in RESOURCE_STATES:
        raise RegistryError("INVALID_REQUEST", f"unsupported resource state {state!r}")
    _assert_non_secret(endpoint, "endpoint")
    _assert_non_secret(account_label, "account_label")
    with store.read() as conn:
        recipe = conn.execute(
            "SELECT 1 FROM capabilities WHERE capability_id = ? AND workspace_id = ?",
            (recipe_id, workspace_id),
        ).fetchone()
    if recipe is None:
        raise RegistryError("INVALID_REQUEST", "recipe_id must reference a registered capability")
    resource = {
        "resource_id": "res:" + contracts.sha256_hex(
            contracts.compact_json([workspace_id, project_id, name, endpoint, account_label])
        )[:40],
        "project_id": project_id,
        "name": name,
        "endpoint": endpoint,
        "account_label": account_label,
        "credential_ref": credential_ref,
        "recipe_id": recipe_id,
        "last_verified_at": last_verified_at,
        "state": state,
    }
    _assert_contract("Resource", resource)
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO resources "
            "(resource_id, workspace_id, project_id, name, endpoint, account_label, credential_ref, "
            "recipe_id, last_verified_at, state, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                resource["resource_id"],
                workspace_id,
                project_id,
                name,
                endpoint,
                account_label,
                credential_ref,
                recipe_id,
                last_verified_at,
                state,
                utc_now(),
            ),
        )
    return {"workspace_id": workspace_id, **resource}


def _capability_row(row) -> dict:
    return {
        "capability_id": row["capability_id"],
        "actor_id": row["actor_id"],
        "name": row["name"],
        "source_uri": row["source_uri"],
        "version": row["version"],
        "content_hash": row["content_hash"],
        "install_locator": row["install_locator"],
        "dependencies": json.loads(row["dependencies_json"]),
        "state": row["state"],
        "verified_at": row["verified_at"],
    }


def _resource_row(row) -> dict:
    return {
        "resource_id": row["resource_id"],
        "project_id": row["project_id"],
        "name": row["name"],
        "endpoint": row["endpoint"],
        "account_label": row["account_label"],
        "credential_ref": row["credential_ref"],
        "recipe_id": row["recipe_id"],
        "last_verified_at": row["last_verified_at"],
        "state": row["state"],
    }


def list_capabilities(store: Store, *, workspace_id: str, actor_id: str | None = None) -> list[dict]:
    with store.read() as conn:
        if actor_id is not None:
            rows = conn.execute(
                "SELECT * FROM capabilities WHERE workspace_id = ? AND actor_id = ? "
                "ORDER BY name, capability_id",
                (workspace_id, actor_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM capabilities WHERE workspace_id = ? ORDER BY name, capability_id",
                (workspace_id,),
            ).fetchall()
    items = [_capability_row(row) for row in rows]
    for item in items:
        _assert_contract("Capability", item)
    return items


def list_resources(
    store: Store,
    *,
    workspace_id: str,
    project_ids: list[str] | None = None,
) -> list[dict]:
    with store.read() as conn:
        rows = conn.execute(
            "SELECT * FROM resources WHERE workspace_id = ? ORDER BY name, resource_id",
            (workspace_id,),
        ).fetchall()
    items = [_resource_row(row) for row in rows if project_ids is None or row["project_id"] in project_ids]
    for item in items:
        _assert_contract("Resource", item)
    return items


def get_capability(store: Store, capability_id: str, *, workspace_id: str) -> dict | None:
    with store.read() as conn:
        row = conn.execute(
            "SELECT * FROM capabilities WHERE capability_id = ? AND workspace_id = ?",
            (capability_id, workspace_id),
        ).fetchone()
    return _capability_row(row) if row else None


def get_resource(store: Store, resource_id: str, *, workspace_id: str) -> dict | None:
    with store.read() as conn:
        row = conn.execute(
            "SELECT * FROM resources WHERE resource_id = ? AND workspace_id = ?",
            (resource_id, workspace_id),
        ).fetchone()
    return _resource_row(row) if row else None


def update_capability_state(store: Store, capability_id: str, *, workspace_id: str, state: str, verified_at: str | None = None) -> dict:
    if state not in CAPABILITY_STATES:
        raise RegistryError("INVALID_REQUEST", f"unsupported capability state {state!r}")
    with store.transaction() as conn:
        cursor = conn.execute(
            "UPDATE capabilities SET state = ?, verified_at = COALESCE(?, verified_at), updated_at = ? "
            "WHERE capability_id = ? AND workspace_id = ?",
            (state, verified_at, utc_now(), capability_id, workspace_id),
        )
        if cursor.rowcount == 0:
            raise RegistryError("SOURCE_NOT_FOUND", "capability not found")
    return get_capability(store, capability_id, workspace_id=workspace_id)  # type: ignore[return-value]


def update_resource_state(store: Store, resource_id: str, *, workspace_id: str, state: str, last_verified_at: str | None = None) -> dict:
    if state not in RESOURCE_STATES:
        raise RegistryError("INVALID_REQUEST", f"unsupported resource state {state!r}")
    with store.transaction() as conn:
        cursor = conn.execute(
            "UPDATE resources SET state = ?, last_verified_at = COALESCE(?, last_verified_at), updated_at = ? "
            "WHERE resource_id = ? AND workspace_id = ?",
            (state, last_verified_at, utc_now(), resource_id, workspace_id),
        )
        if cursor.rowcount == 0:
            raise RegistryError("SOURCE_NOT_FOUND", "resource not found")
    return get_resource(store, resource_id, workspace_id=workspace_id)  # type: ignore[return-value]
