"""AC-41 coverage gap: optional-dependency failure isolation + build provenance.

Codex round-2: "可选依赖失败隔离未独立执行完整用例". This module runs the **real
core** in a child interpreter whose import path makes the optional capabilities
(``mcp``, ``httpx``, ``jieba``, ``pypinyin``, ``sqlite_vec``, ``cryptography``,
``numpy``, ...) unimportable, then proves that core, FTS keyword recall, the
container-free ``service.app``/``ServiceClient`` API (Unix socket), and logical
import/export all still work, while the optional failures stay isolated.

No real network/model is used: the only transport is a loopback Unix socket in
``tmp_path`` and every secret/credential is a synthetic canary.

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-41):
"核心、FTS、API、导入导出在空环境运行；可选依赖失败不拖垮核心；记录锁文件与
wheel来源。"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import knowledge_weaver
from knowledge_weaver import contracts

_CHILD_SOURCE = r'''
"""Child interpreter: optional deps unavailable, real core exercises."""
import importlib.abc
import importlib.metadata
import json
import sys
import time
import os
from pathlib import Path

BLOCKED = {
    "mcp", "httpx", "jieba", "pypinyin", "sqlite_vec", "cryptography",
    "numpy", "faiss", "torch", "onnxruntime", "sentence_transformers",
}


class BlockOptional(importlib.abc.MetaPathFinder):
    """Raise ImportError for every optional capability (deterministic failure)."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(
                "optional dependency %r is unavailable (AC-41 isolation test)" % root
            )
        return None


sys.meta_path.insert(0, BlockOptional())

report = {"optional_imports_blocked": [], "core": {}, "api": {}, "portability": {}}
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)

# 1) Core + FTS: import the 2.0 core packages and write/retrieve through FTS.
import knowledge_weaver
from knowledge_weaver import contracts
from knowledge_weaver.store import Store, receive_batch, record_claim_revision, register_client
from knowledge_weaver.retrieval import execute_recall

report["core"]["version"] = knowledge_weaver.__version__

def make_event(native_id, text):
    return {
        "schema_version": "1.0",
        "event_id": contracts.canonical_event_id("owner:demo", "instance:hermes-test", native_id),
        "event_revision": 1,
        "workspace_id": "owner:demo",
        "operation": "create",
        "source": {
            "instance_id": "instance:hermes-test",
            "host": "hermes",
            "actor_id": "jun-test",
            "session_id": "session:test",
            "native_event_id": native_id,
            "native_schema": "hermes-snapshot-20261030",
            "role": "user",
            "origin_kind": "user_input",
            "locator": "hermes://session:test/" + native_id,
        },
        "event_time": {"value": "2026-10-30T09:00:00Z", "precision": "instant", "timezone": "UTC"},
        "body": {
            "text": text,
            "sha256": contracts.body_digest(text),
            "part_index": 0,
            "part_count": 1,
            "content_revision": "native:1",
            "format": "text/plain",
        },
        "redaction": {"policy_version": "redact:1", "status": "passed", "replacements": 0, "credential_refs": []},
        "project_ids": ["project:songta"],
        "parent_revision": None,
        "derived_from": [],
    }

def make_batch(events):
    return {"schema_version": "1.0", "batch_id": "batch:ac41", "events": events}

store = Store(root / "kw" / "knowledge.db")
register_client(
    store,
    client_id="client:test",
    workspace_id="owner:demo",
    host="hermes",
    instance_id="instance:hermes-test",
    actor_ids=["*"],
    project_ids=["project:songta"],
)
event = make_event("message:ac41", "松塔项目端口为4200")
receipt = receive_batch(store, make_batch([event]))["receipts"][0]
assert receipt["disposition"] == "accepted", receipt
text = event["body"]["text"]
claim = record_claim_revision(
    store,
    workspace_id="owner:demo",
    project_id="project:songta",
    subject_id="subject:songta-server",
    predicate="server.port",
    facet="desired",
    qualifiers={},
    value=4200,
    display_text="端口4200",
    authority="user_confirmed",
    assertion_kind="explicit_statement",
    event_time=event["event_time"],
    evidence=[{
        "event_id": event["event_id"],
        "event_revision": 1,
        "part_index": 0,
        "start_byte": 0,
        "end_byte": len(text.encode("utf-8")),
        "body_sha256": event["body"]["sha256"],
        "availability": "full",
    }],
)
request = {
    "schema_version": "1.0",
    "request_id": "req:test:1",
    "session_id": "session:test",
    "turn_id": "turn:1",
    "query": "端口",
    "project_ids": ["project:songta"],
    "intent": "current",
    "budget_tokens": 2000,
    "limit": 8,
    "deadline_ms": 7000,
    "cursor": None,
}
binding = {"client_id": "client:test", "workspace_id": "owner:demo",
           "project_ids": ["project:songta"], "actor_ids": ["*"]}
recall = execute_recall(store, request, binding=binding)
assert recall["items"] and recall["items"][0]["value"] == 4200, recall
assert recall["incomplete_reasons"] == ["embedding_unavailable"], recall
report["core"]["fts_recall_items"] = len(recall["items"])
report["core"]["claim_revision"] = claim["revision_id"]

# 2) API: the real Unix-socket service app + stdlib client.
from knowledge_weaver.service import ServiceApp
from knowledge_weaver.service.client import ServiceClient

sock = Path(sys.argv[2]) / "kw.sock"
app = ServiceApp(store, sock).start()
try:
    client = ServiceClient(app.socket_path, client_id="client:test")
    status, health = client.health()
    assert status == 200 and health["status"] in ("ready", "restoring"), (status, health)
    status, body = client.get_source(event["event_id"], 1, 0)
    assert status == 200 and body["availability"] == "full", (status, body)
    status, body = client.post_batch(make_batch([make_event("message:ac41-2", "备份每周")]))
    assert status == 200 and body["receipts"][0]["disposition"] == "accepted", (status, body)
    report["api"] = {"health": health["status"], "socket": str(sock)}
finally:
    app.stop()

# 3) Portability: logical export/import works without optional packages.
from knowledge_weaver.portability import export_snapshot, import_snapshot

snapshot = root / "snapshot"
exported = export_snapshot(store, snapshot)
target = Path(os.path.realpath(root / "imported"))
imported = import_snapshot(snapshot, target)
assert imported["counts"]["claim_revisions"] >= 1, imported
assert imported["rebuild_required"]["vectors"] is True, imported
store2 = Store(imported["target_db"])
recall2 = execute_recall(store2, request, binding=binding)
assert recall2["items"] and recall2["items"][0]["value"] == 4200, recall2
report["portability"] = {"files": exported["files"], "claims": imported["counts"]["claims"]}

# 4) Optional failures are isolated and explicit, not a core crash.
try:
    import knowledge_weaver.server  # noqa: F401 - requires the optional mcp extra
except ImportError as exc:
    report["optional_imports_blocked"].append({"module": "knowledge_weaver.server", "error": str(exc)})
else:
    raise AssertionError("knowledge_weaver.server must require the optional mcp extra")
try:
    import knowledge_weaver.embedder  # noqa: F401 - legacy v1 adapter needs httpx
except ImportError as exc:
    report["optional_imports_blocked"].append({"module": "knowledge_weaver.embedder", "error": str(exc)})
else:
    raise AssertionError("knowledge_weaver.embedder must require the optional httpx extra")

# Core still works after the optional imports failed.
assert execute_recall(store, request, binding=binding)["items"]
report["ok"] = True
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
'''


@pytest.fixture
def child_env():
    kw_src = Path(knowledge_weaver.__file__).resolve().parents[1]
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join([str(kw_src)]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return env


def test_core_fts_api_import_export_survive_optional_dependency_failure(tmp_path, child_env):
    """Clauses: 核心、FTS、API、导入导出在空环境运行; 可选依赖失败不拖垮核心."""
    script = tmp_path / "ac41_child.py"
    script.write_text(_CHILD_SOURCE, encoding="utf-8")
    workdir = tmp_path / "child-work"
    # Unix domain socket paths are limited to ~103 bytes; the pytest basetemp
    # under the workspace is too deep, so keep only the socket in /tmp (the same
    # loopback-short-path pattern the workspace E2E runner uses).
    socket_dir = tempfile.mkdtemp(prefix="umac41-", dir="/tmp")
    try:
        completed = subprocess.run(
            [sys.executable, str(script), str(workdir), socket_dir],
            capture_output=True,
            text=True,
            env=child_env,
            cwd=str(tmp_path),
            timeout=120,
        )
    finally:
        import shutil

        shutil.rmtree(socket_dir, ignore_errors=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True, payload
    assert payload["core"]["fts_recall_items"] >= 1, payload
    assert payload["core"]["version"], payload
    assert payload["api"]["health"] in ("ready", "restoring"), payload
    assert payload["portability"]["files"] >= 1 and payload["portability"]["claims"] >= 1, payload
    blocked = {entry["module"] for entry in payload["optional_imports_blocked"]}
    assert blocked == {"knowledge_weaver.server", "knowledge_weaver.embedder"}, payload


def _repo_root() -> Path:
    """Locate the source checkout from this test file, not from the install path.

    The Phase-2 evidence run installs the products as wheels, so
    ``knowledge_weaver.__file__`` lives under ``.../site-packages``; the lock
    files and pyproject live in the repository checkout that this test ships
    with. Using ``__file__`` keeps the provenance assertions valid for both the
    editable dev venv and the sealed-wheel gate environment.
    """
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "requirements").is_dir():
        return checkout
    installed = Path(knowledge_weaver.__file__).resolve().parents[2]
    if (installed / "pyproject.toml").is_file():
        return installed
    pytest.skip(f"repository checkout not found from {__file__!r}")


def test_locked_build_inputs_and_wheel_provenance_are_recorded():
    """Clause: 记录锁文件与wheel来源."""
    kw_root = _repo_root()
    pyproject = kw_root / "pyproject.toml"

    import tomllib

    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    # The core install must stay standard-library only; optional capabilities
    # live in extras so a missing one can never break the core.
    assert project["dependencies"] == [], project["dependencies"]
    extras = project.get("optional-dependencies", {})
    assert "mcp" in extras, extras
    assert any("httpx" in dependency for dependency in extras["llm"]), extras

    requirements = kw_root / "requirements"
    test_lock = requirements / "test-lock-macos13-x86_64-py311.txt"
    build_lock = requirements / "build-lock-macos13-x86_64-py311.txt"
    for lock in (test_lock, build_lock):
        text = lock.read_text(encoding="utf-8")
        assert "--hash=sha256:" in text, lock
        assert "macOS 13" in text or "macos13" in text or "Target:" in text, lock
    test_lock_text = test_lock.read_text(encoding="utf-8")
    assert "pytest==" in test_lock_text and "--hash=sha256:" in test_lock_text
    build_lock_text = build_lock.read_text(encoding="utf-8")
    assert "setuptools==" in build_lock_text and "wheel==" in build_lock_text

    # The installed distribution and its provenance are recorded (editable
    # install in the dev venv; the Phase-2 evidence run will be a wheel install).
    distribution = importlib_metadata_distribution("memorelay-core")
    assert distribution.version == knowledge_weaver.__version__
    assert distribution.metadata["Name"].lower().replace("_", "-") == "memorelay-core"
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:  # editable/direct installs record their source
        recorded = json.loads(direct_url)
        assert "url" in recorded, recorded
        assert recorded["url"].startswith("file://"), recorded


def importlib_metadata_distribution(name):
    from importlib import metadata

    return metadata.distribution(name)
