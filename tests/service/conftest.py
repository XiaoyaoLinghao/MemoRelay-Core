"""UDS service fixtures: synthetic data only, runtime under pytest tmp_path."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from knowledge_weaver.service import ServiceApp
from knowledge_weaver.service.client import ServiceClient
from knowledge_weaver.store import Store, register_client

from ..store.conftest import make_batch, make_event  # noqa: F401 - re-exported builders


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


@pytest.fixture
def socket_path():
    # macOS AF_UNIX paths are limited to ~104 bytes; pytest's tmp_path is too
    # long, so the socket lives in a short-lived isolated /tmp directory while
    # databases remain under tmp_path.
    directory = Path(tempfile.mkdtemp(prefix="umkw-", dir="/tmp"))
    try:
        yield directory / "kw.sock"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def binding(store):
    return register_client(
        store,
        client_id="client:hermes-main",
        workspace_id="owner:demo",
        host="hermes",
        instance_id="instance:hermes-test",
        actor_ids=["jun-test"],
        project_ids=["project:songta"],
    )


@pytest.fixture
def service(store, socket_path):
    app = ServiceApp(store, socket_path)
    app.start()
    yield app
    app.stop()


@pytest.fixture
def client(service, binding):
    return ServiceClient(service.socket_path, client_id=binding["client_id"])
