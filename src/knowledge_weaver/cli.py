"""kw-memory CLI: install/doctor/status/export/import/cleanup/workers/serve (T17).

All commands print machine-readable JSON. No command edits launchd/cron, no
command reads keychain secrets, and the service only starts in the foreground
(an operator reviews and installs any user-level service themselves).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from knowledge_weaver import contracts


def _emit(payload, *, code: int = 0) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return code


def _store(db_path: str, *, create: bool = False):
    from knowledge_weaver.store import Store

    path = Path(db_path)
    if not create and not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    return Store(path, initialize=create)


def cmd_versions(_args) -> int:
    return _emit(
        {
            "contract_version": contracts.CONTRACT_VERSION,
            "event_schema_version": contracts.EVENT_SCHEMA_VERSION,
            "extraction_schema_version": contracts.EXTRACTION_SCHEMA_VERSION,
            "redaction_policy_version": contracts.redaction_policy_version(),
            "contracts_dir": str(contracts.contracts_dir()),
        }
    )


def cmd_doctor(args) -> int:
    checks = []
    health = contracts.contract_health()
    checks.append({"check": "contracts", "passed": health["passed"], "detail": None if health["passed"] else "contract health failed"})
    checks.append({"check": "python", "passed": sys.version_info >= (3, 11), "detail": sys.version.split()[0]})
    db = Path(args.db)
    if db.is_file():
        store = _store(args.db)
        integrity = store.quick_check()
        meta = store.meta()
        checks.append({"check": "database", "passed": integrity == "ok", "detail": {"integrity": integrity, "watermark": meta["watermark"], "restore_epoch": meta["restore_epoch"]}})
    else:
        checks.append({"check": "database", "passed": False, "detail": "not found (run init)"})
    if args.socket:
        socket_path = Path(args.socket)
        if socket_path.exists():
            checks.append({"check": "socket", "passed": socket_path.stat().st_mode & 0o777 == 0o600, "detail": str(socket_path)})
        else:
            checks.append({"check": "socket", "passed": False, "detail": "socket not running (start serve)"})
    failed = [check for check in checks if not check["passed"]]
    return _emit({"ok": not failed, "checks": checks, "failed": failed}, code=0 if not failed else 1)


def cmd_init(args) -> int:
    from knowledge_weaver.store import register_client

    store = _store(args.db, create=True)
    if args.client:
        register_client(
            store,
            client_id=args.client,
            workspace_id=args.workspace,
            host=args.host,
            instance_id=args.instance,
            actor_ids=args.actor or ["*"],
            project_ids=args.project or [],
        )
    return _emit({"ok": True, "db": str(store.db_path), "meta": store.meta(), "client": args.client})


def cmd_status(args) -> int:
    from knowledge_weaver.jobs import status_counts
    from knowledge_weaver.lifecycle import lifecycle_status
    from knowledge_weaver.store import status as status_view

    store = _store(args.db)
    return _emit({"ok": True, "status": status_view(store), "jobs": status_counts(store), "lifecycle": lifecycle_status(store)})


def cmd_export(args) -> int:
    from knowledge_weaver.portability import export_snapshot

    store = _store(args.db)
    return _emit({"ok": True, **export_snapshot(store, args.out)})


def cmd_import(args) -> int:
    from knowledge_weaver.portability import import_snapshot

    report = import_snapshot(args.snapshot, args.target, bootstrap_import=args.bootstrap)
    return _emit({"ok": True, **report})


def cmd_validate_snapshot(args) -> int:
    from knowledge_weaver.portability import validate_snapshot

    validated = validate_snapshot(args.snapshot)
    return _emit(
        {
            "ok": True,
            "manifest": validated["manifest"],
            "files": len(validated["manifest"]["files"]) + 2,
        }
    )


def cmd_cleanup(args) -> int:
    from knowledge_weaver.lifecycle import lifecycle_status, sweep
    from knowledge_weaver.lifecycle.retention import classify_claim_revisions, normalize_clock

    if args.now is not None:
        # F09b: refuse an invalid/ambiguous clock before touching the database;
        # the sweep/classify entries still normalize it (keeping the exact
        # instant) so an offset clock means the same instant as its Z form.
        normalize_clock(args.now)
    store = _store(args.db)
    if args.dry_run:
        classify_claim_revisions(store, now=args.now)
        due = lifecycle_status(store, now=args.now)["due_now"]
        return _emit({"ok": True, "dry_run": True, "due_now": due, "note": "classification records refreshed; nothing expired or purged"})
    result = sweep(store, now=args.now)
    return _emit({"ok": True, "dry_run": False, **result, "status": lifecycle_status(store, now=args.now)})


def cmd_tasks(args) -> int:
    from knowledge_weaver.store import list_task_views

    store = _store(args.db)
    views = list_task_views(store, workspace_id=args.workspace, project_ids=args.project or None)
    return _emit({"ok": True, "tasks": views})


def cmd_merge_entity(args) -> int:
    from knowledge_weaver.store import merge_entities

    store = _store(args.db)
    alias = merge_entities(
        store,
        workspace_id=args.workspace,
        project_id=args.entity_project,
        source_subject_id=args.source,
        canonical_subject_id=args.canonical,
        reason=args.reason,
    )
    return _emit({"ok": True, "alias": alias})


def cmd_undo_merge(args) -> int:
    from knowledge_weaver.store import undo_merge

    store = _store(args.db)
    result = undo_merge(
        store,
        workspace_id=args.workspace,
        project_id=args.entity_project,
        source_subject_id=args.source,
        canonical_subject_id=args.canonical,
    )
    return _emit({"ok": True, "undo": result})


def cmd_serve(args) -> int:
    from knowledge_weaver.embeddings.index import HashEmbeddingProvider, configure_model
    from knowledge_weaver.service import ServiceApp

    store = _store(args.db)
    embedder = None
    if args.embedding_provider == "hash":
        embedder = HashEmbeddingProvider(dimension=args.dimension, model_id="kw:hash-embedding")
    elif args.embedding_provider == "http":
        from knowledge_weaver.embeddings.http_provider import HttpEmbeddingProvider

        if not args.embedding_url or not args.embedding_model:
            raise ValueError("--embedding-url and --embedding-model are required for the http provider")
        embedder = HttpEmbeddingProvider(
            args.embedding_url,
            model_id=args.embedding_model,
            dimension=args.dimension,
            model_revision=args.embedding_model_revision,
            api_key_env=args.embedding_key_env,
        )
    if embedder is not None:
        configure_model(
            embedder.model_id,
            model_revision=embedder.model_revision,
            dimension=embedder.dimension,
        )
    app = ServiceApp(store, args.socket, embedder=embedder)
    print(json.dumps({"ok": True, "db": str(store.db_path), "socket": str(args.socket), "embedder": args.embedding_provider}))
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


def cmd_extract_worker(args) -> int:
    from knowledge_weaver.extraction import CommandExtractor, HttpExtractor, run_worker_once
    from knowledge_weaver.jobs import status_counts

    store = _store(args.db)
    if args.command:
        extractor = CommandExtractor(args.command, timeout=args.timeout)
    else:
        if not args.url or not args.model:
            raise ValueError("--url and --model are required when --command is not used")
        extractor = HttpExtractor(
            args.url,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            thinking=None if args.thinking == "provider-default" else args.thinking == "on",
        )
    total = {"claimed": 0, "succeeded": 0, "failed": 0, "retry_wait": 0, "stale": 0}
    try:
        while True:
            result = run_worker_once(store, extractor, worker_id=args.worker, lease_seconds=max(60, int(args.timeout) + 30))
            total["claimed"] += result["claimed"]
            if result["claimed"]:
                total[result["state"]] = total.get(result["state"], 0) + 1
            else:
                break
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    return _emit({"ok": True, "processed": total, "jobs": status_counts(store)})


def cmd_embed_worker(args) -> int:
    from knowledge_weaver.embeddings.index import HashEmbeddingProvider, run_embed_worker_once
    from knowledge_weaver.jobs import status_counts

    store = _store(args.db)
    if args.provider == "http":
        from knowledge_weaver.embeddings.http_provider import HttpEmbeddingProvider

        if not args.url or not args.model:
            raise ValueError("--url and --model are required for --provider http")
        provider = HttpEmbeddingProvider(
            args.url,
            model_id=args.model,
            dimension=args.dimension,
            model_revision=args.model_revision,
            api_key_env=args.api_key_env,
        )
    else:
        provider = HashEmbeddingProvider(dimension=args.dimension, model_id="kw:hash-embedding")
    total = {"claimed": 0, "succeeded": 0, "failed": 0}
    try:
        while True:
            result = run_embed_worker_once(store, provider, worker_id=args.worker)
            total["claimed"] += result["claimed"]
            if result["claimed"]:
                total[result["state"]] = total.get(result["state"], 0) + 1
            else:
                break
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    return _emit({"ok": True, "processed": total, "jobs": status_counts(store)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memorelay-core", description="Unified Memory KW 2.0 operations CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("versions", help="print contract/schema versions")
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("doctor", help="read-only environment/database checks")
    p.add_argument("--db", default=str(Path.home() / "Library/Application Support/UnifiedMemory/kw/knowledge.db"))
    p.add_argument("--socket", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("init", help="create the local database and register a client binding")
    p.add_argument("--db", required=True)
    p.add_argument("--workspace", default="owner:local")
    p.add_argument("--host", default="local")
    p.add_argument("--instance", default="instance:local")
    p.add_argument("--client", default=None)
    p.add_argument("--actor", action="append", default=None)
    p.add_argument("--project", action="append", default=None)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("status", help="health, pipelines, jobs and retention status")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("export", help="write a portable logical snapshot")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="validate and import a snapshot into a new directory")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--bootstrap", action="store_true")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("validate-snapshot", help="validate snapshot versions/hashes/paths")
    p.add_argument("--snapshot", required=True)
    p.set_defaults(func=cmd_validate_snapshot)

    p = sub.add_parser("cleanup", help="run retention sweep (or a dry-run report)")
    p.add_argument("--db", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--now", default=None)
    p.set_defaults(func=cmd_cleanup)

    p = sub.add_parser("tasks", help="list task verification views")
    p.add_argument("--db", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", action="append", default=None)
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("merge-entity", help="manually merge a subject into a canonical subject (reversible alias)")
    p.add_argument("--db", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--entity-project", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--canonical", required=True)
    p.add_argument("--reason", default="manual merge")
    p.set_defaults(func=cmd_merge_entity)

    p = sub.add_parser("undo-merge", help="undo a manual entity merge")
    p.add_argument("--db", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--entity-project", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--canonical", required=True)
    p.set_defaults(func=cmd_undo_merge)

    p = sub.add_parser("serve", help="run the KW service in the foreground (no launchd changes)")
    p.add_argument("--db", required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--embedding-provider", choices=["none", "hash", "http"], default="none")
    p.add_argument("--embedding-url", default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-model-revision", default="1")
    p.add_argument("--embedding-key-env", default=None)
    p.add_argument("--dimension", type=int, default=64)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("extract-worker", help="process extraction jobs via an external model command")
    p.add_argument("--db", required=True)
    p.add_argument("--command", nargs="+", default=None, help="external command; JSON on stdin/stdout; credentials stay in its environment")
    p.add_argument("--url", default=None, help="OpenAI-compatible /v1/chat/completions endpoint")
    p.add_argument("--model", default=None)
    p.add_argument("--thinking", choices=["provider-default", "on", "off"], default="provider-default", help="provider-specific thinking toggle; use off for DeepSeek extraction")
    p.add_argument("--api-key-env", default=None, help="env var name holding the API key (never the value)")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--worker", default="worker:extract")
    p.set_defaults(func=cmd_extract_worker)

    p = sub.add_parser("embed-worker", help="process embedding jobs (offline hash provider for smoke/testing)")
    p.add_argument("--db", required=True)
    p.add_argument("--provider", choices=["hash", "http"], default="hash")
    p.add_argument("--url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--model-revision", default="1")
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--dimension", type=int, default=64)
    p.add_argument("--once", action="store_true")
    p.add_argument("--worker", default="worker:embed")
    p.set_defaults(func=cmd_embed_worker)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary, no secret content in messages
        code = getattr(exc, "code", type(exc).__name__)
        message = getattr(exc, "message", None) or str(exc)
        return _emit({"ok": False, "error": {"code": code, "message": message[:300]}}, code=2)


if __name__ == "__main__":
    raise SystemExit(main())
