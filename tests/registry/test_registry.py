"""T15: non-secret capability/resource registry and references."""
from __future__ import annotations

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.resource_registry import (
    RegistryError,
    list_capabilities,
    list_resources,
    register_capability,
    register_resource,
    update_capability_state,
    update_resource_state,
)

HASH = "a" * 64


def _capability(store, *, workspace="owner:demo", name="kw-consolidate-skill"):
    return register_capability(
        store,
        workspace_id=workspace,
        actor_id="actor:hermes",
        name=name,
        source_uri="https://github.com/example/skill",
        version="1.2.0",
        content_hash=HASH,
        install_locator="~/.hermes/skills/kw-consolidate",
        dependencies=["python>=3.11"],
        state="available",
        verified_at="2026-10-30T00:00:00Z",
    )


def test_capability_roundtrip_and_contract_shape(store):
    registered = _capability(store)
    items = list_capabilities(store, workspace_id="owner:demo")
    assert len(items) == 1
    assert contracts.validate_instance("Capability", items[0]) == []
    assert items[0]["name"] == "kw-consolidate-skill"
    assert items[0]["dependencies"] == ["python>=3.11"]
    assert not hasattr(registered, "credential_value")


def test_resource_requires_registered_recipe_and_reference_only(store):
    recipe = _capability(store)
    resource = register_resource(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        name="nas-share",
        endpoint="smb://nas.example.invalid/songta",
        account_label="songta-readonly",
        credential_ref="credref:nas:readonly",
        recipe_id=recipe["capability_id"],
        state="needs_os_authorization",
    )
    items = list_resources(store, workspace_id="owner:demo")
    assert len(items) == 1
    assert contracts.validate_instance("Resource", items[0]) == []
    assert resource["credential_ref"] == "credref:nas:readonly"
    with pytest.raises(RegistryError):
        register_resource(
            store,
            workspace_id="owner:demo",
            project_id=None,
            name="unknown-recipe",
            endpoint="smb://nas.example.invalid/other",
            account_label="x",
            credential_ref="credref:x",
            recipe_id="cap:missing",
        )


def test_secret_bearing_metadata_is_refused(store):
    with pytest.raises(RegistryError) as excinfo:
        register_capability(
            store,
            workspace_id="owner:demo",
            actor_id="actor:hermes",
            name="leaky",
            source_uri="https://example.invalid/?token=CANARY-TOKEN-abc123",
            version="1.0",
            content_hash=HASH,
            install_locator="/tmp/x",
        )
    assert excinfo.value.code == "REDACTION_REQUIRED"
    recipe = _capability(store)
    with pytest.raises(RegistryError):
        register_resource(
            store,
            workspace_id="owner:demo",
            project_id=None,
            name="leaky",
            endpoint="smb://user:CANARY-PW-1@nas.example.invalid/share",
            account_label="x",
            credential_ref="credref:x",
            recipe_id=recipe["capability_id"],
        )


def test_invalid_hash_state_and_unknown_update(store):
    with pytest.raises(RegistryError):
        register_capability(
            store,
            workspace_id="owner:demo",
            actor_id="actor:hermes",
            name="bad-hash",
            source_uri="https://example.invalid/x",
            version="1.0",
            content_hash="not-a-hash",
            install_locator="/tmp/x",
        )
    capability = _capability(store)
    with pytest.raises(RegistryError):
        update_capability_state(store, capability["capability_id"], workspace_id="owner:demo", state="running")
    updated = update_capability_state(
        store, capability["capability_id"], workspace_id="owner:demo", state="incompatible"
    )
    assert updated["state"] == "incompatible"
    with pytest.raises(RegistryError):
        update_capability_state(store, "cap:missing", workspace_id="owner:demo", state="missing")


def test_workspace_isolation(store):
    _capability(store, workspace="owner:demo")
    assert list_capabilities(store, workspace_id="owner:other") == []


def test_resource_state_update_and_filtering(store):
    recipe = _capability(store)
    register_resource(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        name="nas",
        endpoint="smb://nas.example.invalid/songta",
        account_label="readonly",
        credential_ref="credref:nas",
        recipe_id=recipe["capability_id"],
    )
    register_resource(
        store,
        workspace_id="owner:demo",
        project_id="project:other",
        name="nas-other",
        endpoint="smb://nas.example.invalid/other",
        account_label="readonly",
        credential_ref="credref:nas",
        recipe_id=recipe["capability_id"],
    )
    assert len(list_resources(store, workspace_id="owner:demo", project_ids=["project:songta"])) == 1
    resource = list_resources(store, workspace_id="owner:demo", project_ids=["project:songta"])[0]
    updated = update_resource_state(
        store, resource["resource_id"], workspace_id="owner:demo", state="available",
        last_verified_at="2026-10-30T00:00:00Z",
    )
    assert updated["state"] == "available"
    assert updated["last_verified_at"] == "2026-10-30T00:00:00Z"
