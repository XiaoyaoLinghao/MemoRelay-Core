"""Client-binding and scope authorization for the KW service (SRC-01 / AC-42).

Authorization comes from the installed local client binding and the native
envelope, never from body text and never from model-selected roles. The binding
is a trusted installation registration, not a cryptographic identity or a
hostile same-OS-user boundary.
"""

from __future__ import annotations

from knowledge_weaver.store import Store, get_client

from ..errors import ApiError


def resolve_binding(store: Store, client_id: str | None) -> dict:
    if not client_id:
        raise ApiError("UNREGISTERED_CLIENT", "missing X-DMA-Client binding")
    binding = get_client(store, client_id)
    if binding is None or binding["revoked"]:
        raise ApiError("UNREGISTERED_CLIENT", "unknown or revoked local client binding")
    return binding


def _actor_allowed(binding: dict, actor_id: str) -> bool:
    allowed = binding["actor_ids"]
    return "*" in allowed or actor_id in allowed


def _projects_allowed(binding: dict, project_ids: list[str]) -> bool:
    allowed = binding["project_ids"]
    if not allowed:  # empty allow-list means "all projects in this workspace"
        return True
    return set(project_ids) <= set(allowed)


def authorize_event(binding: dict, event: dict) -> None:
    workspace_id = event.get("workspace_id")
    if workspace_id != binding["workspace_id"]:
        raise ApiError("SCOPE_DENIED", "event workspace is not authorized for this binding")
    source = event.get("source") or {}
    if source.get("instance_id") != binding["instance_id"]:
        raise ApiError("SCOPE_DENIED", "event instance is not authorized for this binding")
    if source.get("host") != binding["host"]:
        raise ApiError("SCOPE_DENIED", "event host is not authorized for this binding")
    if not _actor_allowed(binding, str(source.get("actor_id"))):
        raise ApiError("SCOPE_DENIED", "event actor is not authorized for this binding")
    if not _projects_allowed(binding, list(event.get("project_ids") or [])):
        raise ApiError("SCOPE_DENIED", "event project scope is not authorized for this binding")


def authorize_batch(binding: dict, batch: dict) -> None:
    events = batch.get("events") if isinstance(batch, dict) else None
    if not isinstance(events, list):
        return  # shape errors are reported by the store layer
    for event in events:
        if isinstance(event, dict):
            authorize_event(binding, event)


def authorize_workspace(binding: dict, workspace_id: str | None) -> None:
    if workspace_id is not None and workspace_id != binding["workspace_id"]:
        raise ApiError("SCOPE_DENIED", "workspace is not authorized for this binding")


def authorize_project(binding: dict, project_ids: list[str] | None) -> None:
    """Authorize a persisted target object's project scope (F21/AC-42).

    A non-empty binding allow-list restricts every read/write entry point, not
    only batch ingestion. An object with no project (workspace-level) is allowed;
    an object carrying a project outside the allow-list is denied before any
    body content is returned or any projection changes.
    """
    allowed = list(binding.get("project_ids") or [])
    if not allowed:
        return
    targets = [str(project) for project in (project_ids or [])]
    if not set(targets) <= set(allowed):
        raise ApiError("SCOPE_DENIED", "target project is not authorized for this binding")


def authorize_scope(
    binding: dict, *, workspace_id: str | None, project_ids: list[str] | None
) -> None:
    """Authorize a stored object's full scope: workspace *and* project (F21/AC-42).

    Every entry point that resolves persisted objects (source reads, correction
    targets, every correction evidence reference) uses this one helper so a
    foreign-workspace object can never pass on a matching project name alone.
    A workspace-level object (no project) stays allowed while the workspace
    matches, and an empty binding project allow-list still means "all projects in
    this workspace".
    """
    authorize_workspace(binding, workspace_id)
    authorize_project(binding, project_ids)
