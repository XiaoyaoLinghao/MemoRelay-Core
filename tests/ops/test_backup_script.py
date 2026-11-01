"""T18: deterministic external weekly backup (local+NAS, 4 generations, verify-then-rotate)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from knowledge_weaver.store import Store

from ..state.conftest import record

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "backup" / "weekly_backup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("weekly_backup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(db_path: Path) -> None:
    from knowledge_weaver.store import Store as _Store

    store = _Store(db_path)
    record(store, display_text="端口4200", value=4200)


def _generations(root: Path) -> list[str]:
    directory = root / "generations"
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def test_backup_verifies_both_copies_and_rotates_only_successful(tmp_path):
    module = _load_module()
    db = tmp_path / "kw" / "knowledge.db"
    _seed(db)
    local = tmp_path / "local"
    nas = tmp_path / "nas"
    for index in range(1, 6):
        report = module.run_backup(
            db_path=db,
            local_dir=local,
            nas_dir=nas,
            keep=4,
            now=f"2026-0{index}-01T00:00:00Z" if index < 5 else "2026-05-01T00:00:00Z",
        )
        assert report["ok"] is True
        assert report["verified"] == {"local": True, "nas": True}
    assert len(_generations(local)) == 4
    assert len(_generations(nas)) == 4
    assert _generations(local) == _generations(nas)
    assert _generations(local)[0] == "gen-20260201T000000Z"
    for generation in _generations(local):
        marker = json.loads((local / "generations" / generation / "SUCCESS.json").read_text())
        assert marker["verified"] == {"local": True, "nas": True}
        assert marker["manifest_sha256"]


def test_failed_nas_copy_does_not_prune_existing_generations(tmp_path):
    module = _load_module()
    db = tmp_path / "kw" / "knowledge.db"
    _seed(db)
    local = tmp_path / "local"
    nas = tmp_path / "nas"
    for index in (1, 2):
        module.run_backup(
            db_path=db, local_dir=local, nas_dir=nas, keep=2, now=f"2026-0{index}-01T00:00:00Z"
        )
    before_local = _generations(local)
    # Make the NAS path unusable: a file where the generations directory should be.
    broken_nas = tmp_path / "broken-nas"
    broken_nas.mkdir()
    (broken_nas / "generations").write_text("not a directory", encoding="utf-8")
    with pytest.raises(Exception):
        module.run_backup(
            db_path=db, local_dir=local, nas_dir=broken_nas, keep=1, now="2026-03-01T00:00:00Z"
        )
    # No successful generation was pruned after the failed copy; the failed
    # generation has no success marker and will never be rotated as successful.
    assert set(before_local) <= set(_generations(local))
    assert _generations(nas) == before_local
    assert not (local / "generations" / "gen-20260301T000000Z" / "SUCCESS.json").exists()
    assert not (local / ".staging-gen-20260301T000000Z").exists()


def test_same_generation_is_idempotent(tmp_path):
    module = _load_module()
    db = tmp_path / "kw" / "knowledge.db"
    _seed(db)
    local = tmp_path / "local"
    nas = tmp_path / "nas"
    first = module.run_backup(db_path=db, local_dir=local, nas_dir=nas, now="2026-01-01T00:00:00Z")
    second = module.run_backup(db_path=db, local_dir=local, nas_dir=nas, now="2026-01-01T00:00:00Z")
    assert first["generation"] == second["generation"]
    assert second.get("existing") is True
    assert len(_generations(local)) == 1


def test_script_does_not_install_a_scheduler(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    assert "launchctl" not in text
    assert "crontab" not in text
    assert "subprocess" not in text
