"""Knowledge Weaver — MCP server for structured knowledge retrieval."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Single source of truth is pyproject.toml [project].version; read it from
    # installed metadata so the version is never hardcoded twice (see VERSIONING.md).
    __version__ = _pkg_version("memorelay-core")
except PackageNotFoundError:
    try:  # compatible with existing Knowledge Weaver installations
        __version__ = _pkg_version("knowledge-weaver")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

