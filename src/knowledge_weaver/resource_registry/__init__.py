"""KW 2.0 capability/resource reference registry (non-secret metadata only)."""

from .registry import (
    RegistryError,
    get_capability,
    get_resource,
    list_capabilities,
    list_resources,
    register_capability,
    register_resource,
    update_capability_state,
    update_resource_state,
)

__all__ = [
    "RegistryError",
    "get_capability",
    "get_resource",
    "list_capabilities",
    "list_resources",
    "register_capability",
    "register_resource",
    "update_capability_state",
    "update_resource_state",
]
