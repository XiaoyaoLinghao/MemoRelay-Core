"""KW 2.0 single semantic extractor (untrusted proposal validation + jobs)."""

from .external import CommandExtractor
from .http_client import HttpExtractor
from .pipeline import (
    MODEL_VERSION,
    PROMPT_VERSION,
    enqueue_extraction,
    enqueue_extraction_in_transaction,
    extraction_input_key,
    reconcile_extraction_jobs,
    run_extraction_job,
    run_worker_once,
)
from .proposals import (
    ExtractionError,
    ResolvedClaim,
    SanitizedPart,
    apply_resolved_claims,
    derive_authority,
    load_event_group,
    resolve_proposal,
    sanitized_parts,
    subject_id_for,
)

__all__ = [
    "MODEL_VERSION",
    "PROMPT_VERSION",
    "CommandExtractor",
    "ExtractionError",
    "HttpExtractor",
    "ResolvedClaim",
    "SanitizedPart",
    "apply_resolved_claims",
    "derive_authority",
    "enqueue_extraction",
    "enqueue_extraction_in_transaction",
    "extraction_input_key",
    "reconcile_extraction_jobs",
    "load_event_group",
    "resolve_proposal",
    "run_extraction_job",
    "run_worker_once",
    "sanitized_parts",
    "subject_id_for",
]
