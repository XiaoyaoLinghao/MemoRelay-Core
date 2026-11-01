#!/usr/bin/env python3
"""External logical backup: local only, or local plus an optional NAS copy.

No scheduler or model calls. A generation is successful only after every
requested destination verifies. Rotation follows verification, never a failed
export/copy. Overlapping invocations sharing the local root fail immediately.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from knowledge_weaver.portability import (  # noqa: E402
    export_snapshot,
    import_snapshot,  # noqa: F401 - re-exported for operators/tests
    validate_snapshot,
)
from knowledge_weaver.store.db import Store  # noqa: E402


def _generation(now: str | None) -> str:
    moment = datetime.fromisoformat(now.replace("Z", "+00:00")) if now else datetime.now(timezone.utc)
    return "gen-" + moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _successful_generations(root: Path) -> list[Path]:
    generations = root / "generations"
    if not generations.is_dir():
        return []
    return sorted(p for p in generations.iterdir() if p.is_dir() and (p / "SUCCESS.json").is_file())


def _verify_generation(path: Path) -> None:
    validate_snapshot(path / "snapshot")
    marker = json.loads((path / "SUCCESS.json").read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != _sha256(path / "snapshot" / "manifest.json"):
        raise ValueError(f"success marker checksum mismatch: {path.name}")


def _publish(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (target / "SUCCESS.json").exists():
            raise ValueError(f"refusing to replace a successful generation: {target.name}")
        shutil.rmtree(target)  # Incomplete generation from an interrupted run.
    source.rename(target)


def _copy_verified(source: Path, staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)
    validate_snapshot(staging)


def _rotate(root: Path, keep: int, report: dict) -> list[str]:
    generations = _successful_generations(root)
    removed = []
    for path in generations[:-keep] if len(generations) > keep else []:
        shutil.rmtree(path)
        removed.append(path.name)
    report.setdefault("removed", {})[str(root)] = removed
    return removed


def run_backup(
    *, db_path: str | Path, local_dir: str | Path,
    nas_dir: str | Path | None = None, keep: int = 4,
    now: str | None = None, dry_run: bool = False,
) -> dict:
    if keep < 1:
        raise ValueError("keep must be >= 1")
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise ValueError("source database must already exist")
    local_root = Path(local_dir).resolve()
    roots = {"local": local_root}
    if nas_dir is not None:
        nas_root = Path(nas_dir).resolve()
        if local_root == nas_root or local_root in nas_root.parents or nas_root in local_root.parents:
            raise ValueError("local and NAS directories must be separate, non-nested roots")
        roots["nas"] = nas_root
    # A mistyped backup root must not rotate the source database away.
    if any(root == db_path or root in db_path.parents for root in roots.values()):
        raise ValueError("source database must be outside backup roots")
    generation = _generation(now)
    report = {
        "generation": generation, "mode": "local_and_nas" if nas_dir is not None else "local_only",
        "local_dir": str(local_root), "nas_dir": str(roots["nas"]) if "nas" in roots else None,
        "keep": keep, "dry_run": dry_run,
        "verified": {name: False for name in roots}, "removed": {}, "ok": False,
    }
    if dry_run:
        report["planned_rotation"] = {
            name: [p.name for p in _successful_generations(root)[:-keep]]
            for name, root in roots.items()
        }
        report["ok"] = True
        return report  # Inspection only, not a successful backup/verification.
    local_root.mkdir(parents=True, exist_ok=True)
    # Keep this file in place: unlinking a live lock would allow a second inode
    # to be locked by another invocation. The OS releases flock on process exit.
    with (local_root / ".backup.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another backup is using this local directory") from None
        stages = {name: root / (".staging-" + generation) for name, root in roots.items()}
        destinations = {name: root / "generations" / generation for name, root in roots.items()}
        try:
            for root in roots.values():
                root.mkdir(parents=True, exist_ok=True)
            local_generation = destinations["local"]
            if (local_generation / "SUCCESS.json").is_file():
                _verify_generation(local_generation)
                report["existing"] = True
            else:
                shutil.rmtree(stages["local"], ignore_errors=True)
                stages["local"].mkdir()
                export_snapshot(Store(db_path), stages["local"] / "snapshot")
                validate_snapshot(stages["local"] / "snapshot")
                _publish(stages["local"], local_generation)
            report["verified"]["local"] = True
            digest = _sha256(local_generation / "snapshot" / "manifest.json")
            if "nas" in roots:
                nas_generation = destinations["nas"]
                if (nas_generation / "SUCCESS.json").is_file():
                    _verify_generation(nas_generation)
                else:
                    _copy_verified(local_generation / "snapshot", stages["nas"] / "snapshot")
                    _publish(stages["nas"], nas_generation)
                if _sha256(nas_generation / "snapshot" / "manifest.json") != digest:
                    raise ValueError("local and NAS generation snapshots differ")
                report["verified"]["nas"] = True
            marker = {
                "generation": generation, "created_at": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": digest, "verified": dict(report["verified"]),
            }
            for destination in destinations.values():
                if not (destination / "SUCCESS.json").exists():
                    pending = destination / ".SUCCESS.pending"
                    pending.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    pending.replace(destination / "SUCCESS.json")
            # Check retained backups too: a corrupt success marker/copy must not
            # count towards the number of usable recovery points.
            for root in roots.values():
                for path in _successful_generations(root):
                    _verify_generation(path)
            for root in roots.values():
                _rotate(root, keep, report)
            report["successful_generations"] = {
                name: [p.name for p in _successful_generations(root)] for name, root in roots.items()
            }
            report["ok"] = True
            return report
        finally:
            for staging in stages.values():
                shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="existing KW database; export records an operation-ledger entry")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--nas-dir", default=None, help="optional second destination; omit for local-only backup")
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--now", default=None, help="ISO timestamp for a deterministic generation id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_backup(db_path=args.db, local_dir=args.local_dir, nas_dir=args.nas_dir,
                            keep=args.keep, now=args.now, dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
