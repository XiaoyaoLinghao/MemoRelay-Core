"""KW 2.0 local evidence store (SQLite, standard library only)."""

from .claims import (
    AUTHORITY_RANK,
    ClaimError,
    claim_id_for,
    current_items,
    get_current,
    recompute_claim,
    record_claim_revision,
    record_claim_revision_in_transaction,
    rebuild_projection,
    resolve_evidence,
)
from .clients import BindingError, get_client, register_client, revoke_client
from .db import DEFAULT_RESTORE_EPOCH, SCHEMA_VERSION, Store, open_store, utc_now
from .entities import EntityError, list_aliases, merge_entities, resolve_canonical_subject, undo_merge
from .events import StoreRejection, event_payload_fingerprint, receive_batch, receipt_id_for
from .sources import fetch_source
from .tasks import list_task_views, recompute_task
from .status import health, status

__all__ = [
    "AUTHORITY_RANK",
    "ClaimError",
    "claim_id_for",
    "current_items",
    "get_current",
    "recompute_claim",
    "record_claim_revision",
    "record_claim_revision_in_transaction",
    "rebuild_projection",
    "resolve_evidence",
    "DEFAULT_RESTORE_EPOCH",
    "SCHEMA_VERSION",
    "BindingError",
    "Store",
    "StoreRejection",
    "event_payload_fingerprint",
    "EntityError",
    "list_aliases",
    "merge_entities",
    "resolve_canonical_subject",
    "undo_merge",
    "fetch_source",
    "list_task_views",
    "recompute_task",
    "get_client",
    "health",
    "open_store",
    "receive_batch",
    "receipt_id_for",
    "register_client",
    "revoke_client",
    "status",
    "utc_now",
]
