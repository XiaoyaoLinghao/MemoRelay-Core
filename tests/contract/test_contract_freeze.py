"""T01: the contract package is frozen, complete and checksum-verified."""
from __future__ import annotations

import json
from pathlib import Path

from knowledge_weaver import contracts as c


def test_contract_directory_found():
    directory = c.contracts_dir()
    assert directory.name == "v1"
    assert (directory / "openapi.json").is_file()


def test_all_frozen_files_present():
    directory = c.contracts_dir()
    for name in c.CONTRACT_FILES:
        assert (directory / name).is_file(), name
    assert (directory / "SHA256SUMS").is_file()
    assert (directory / "README.md").is_file()


def test_checksums_pass():
    assert c.verify_checksums() == []


def test_checksum_tamper_is_detected(tmp_path, monkeypatch):
    source = c.contracts_dir()
    target = tmp_path / "contracts"
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            (target / path.name).write_bytes(path.read_bytes())
    data = json.loads((target / "openapi.json").read_text(encoding="utf-8"))
    data["info"]["version"] = "tampered"
    (target / "openapi.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(c, "contracts_dir", lambda: target)
    monkeypatch.setattr(c, "_load_cache", {})
    issues = c.verify_checksums()
    assert any("checksum mismatch" in issue.message for issue in issues)


def test_undeclared_json_file_is_detected(tmp_path, monkeypatch):
    source = c.contracts_dir()
    target = tmp_path / "contracts"
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            (target / path.name).write_bytes(path.read_bytes())
    (target / "extra.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(c, "contracts_dir", lambda: target)
    issues = c.verify_checksums()
    assert any("not covered" in issue.message for issue in issues)


def test_version_consistency():
    assert c.version_consistency() == []
    info = c.version_info()
    assert info["contract_version"] == c.CONTRACT_VERSION
    assert info["supported_schema_versions"] == ["1.0"]
    assert info["redaction_policy_versions"] == ["redact:1"]
    assert info["compat"]["client_emits"] == "1.0"
    # T20 is Codex-owned: the freeze must not claim independent review passed.
    assert "Codex" in info["frozen_by"] and "pending" in info["frozen_by"]


def test_schema_structure_health():
    assert c.check_schema_structure() == []


def test_openapi_surface_health():
    assert c.check_openapi_surface() == []


def test_semantic_invariants_complete():
    assert c.check_invariants() == []
    ids = {invariant["id"] for invariant in c.invariants()}
    assert ids == set(c.REQUIRED_INVARIANT_IDS)


def test_error_catalog_complete():
    assert c.check_error_catalog() == []
    codes = {entry["code"] for entry in c.error_catalog()["entries"]}
    assert c.REQUIRED_ERROR_CODES <= codes
    assert codes == c.REQUIRED_ERROR_CODES | {"INTERNAL_ERROR"}


def test_contract_health_aggregate_passes():
    result = c.contract_health()
    assert result["passed"], [check for check in result["checks"] if not check.get("passed")]
    # 25 schema checks + 9 valid + 9 invalid + structural checks
    by_check = {check["check"]: check for check in result["checks"]}
    assert by_check["shared-examples"]["passed"]
    assert sum(1 for name in by_check if name.startswith("schema:")) == 25
    assert sum(1 for name in by_check if name.startswith("valid:")) == 9
    assert sum(1 for name in by_check if name.startswith("invalid:")) == 9


def test_contract_check_script_runs_outside_installed_package():
    import subprocess
    import sys

    repo = c.contracts_dir().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "contract_check.py"), "--quiet"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "PASS" in completed.stdout
