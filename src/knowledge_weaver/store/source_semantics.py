"""Operator-reviewed source scope corrections, preserving source bytes/history.

This is a local maintenance interface, never a model proposal or public inbox
field. The operator must first verify native provenance; this function fences
the approved decision against the exact stored sanitized part fingerprint.
"""
from __future__ import annotations

import re
from knowledge_weaver.store.claims import ClaimError, recompute_claim
from knowledge_weaver.store.db import utc_now


def record_source_exclusion(store, *, event_id: str, event_revision: int, part_index: int,
                            body_sha256: str, reason: str, proof_sha256: str,
                            reviewer: str) -> dict:
    if reason not in ('clock_reading', 'historical_recall'):
        raise ClaimError('INVALID_REQUEST', 'unsupported non-assertive source scope')
    if not all(isinstance(x, str) and re.fullmatch(r'[a-f0-9]{64}', x) for x in (body_sha256, proof_sha256)):
        raise ClaimError('INVALID_REQUEST', 'source and native proof fingerprints required')
    if not isinstance(reviewer, str) or not re.fullmatch(r'operator:[A-Za-z0-9._:-]{1,120}', reviewer):
        raise ClaimError('INVALID_REQUEST', 'explicit local operator reviewer required')
    with store.transaction() as conn:
        part = conn.execute('''SELECT p.body_sha256,p.availability,p.withheld,e.role,e.origin_kind
            FROM event_parts p JOIN events e USING(event_id)
            WHERE p.event_id=? AND p.revision=? AND p.part_index=?''',
                            (event_id, event_revision, part_index)).fetchone()
        if part is None or part['body_sha256'] != body_sha256:
            raise ClaimError('STALE_CURSOR', 'approved source fingerprint no longer matches')
        if part['role'] != 'tool' or part['origin_kind'] != 'tool_execution' or part['withheld'] or part['availability'] != 'full':
            raise ClaimError('INVALID_REQUEST', 'only a verified full native tool source may be scope-corrected')
        old = conn.execute('SELECT * FROM source_semantic_exclusions WHERE event_id=? AND event_revision=? AND part_index=?',
                           (event_id, event_revision, part_index)).fetchone()
        if old is not None and (old['body_sha256'] != body_sha256 or old['reason'] != reason or old['proof_sha256'] != proof_sha256):
            raise ClaimError('EVENT_CONFLICT', 'existing reviewed source decision differs')
        inserted = conn.execute('''INSERT OR IGNORE INTO source_semantic_exclusions
            (event_id,event_revision,part_index,body_sha256,reason,proof_sha256,reviewer,reviewed_at)
            VALUES(?,?,?,?,?,?,?,?)''',
                     (event_id, event_revision, part_index, body_sha256, reason, proof_sha256, reviewer, utc_now())).rowcount
        affected = [x[0] for x in conn.execute('''SELECT DISTINCT r.claim_id FROM evidence_links l
            JOIN claim_revisions r USING(revision_id)
            WHERE l.event_id=? AND l.event_revision=? AND l.part_index=? AND l.body_sha256=?''',
                                             (event_id, event_revision, part_index, body_sha256))]
        if inserted:
            store.bump_watermark(conn, 'source_semantic_review', f'{event_id}@{event_revision}#{part_index}')
            for claim_id in affected:
                recompute_claim(conn, claim_id)
    return {'event_id': event_id, 'event_revision': event_revision, 'part_index': part_index,
            'reason': reason, 'affected_claim_ids': affected, 'source_bytes_preserved': True}
