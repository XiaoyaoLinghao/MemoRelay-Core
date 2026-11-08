"""Explicit local review of already-processed native compression copies.

This is neither a model endpoint nor an automatic content deduplicator. The
operator first verifies native UID, exact payload and compression lineage and
keeps that proof by its digest. Here the same transaction checks the immutable
stored scope/parts against the successful extraction of the native original. No source, fact,
failure, attempt, time or recovery link is rewritten, and no model is called.

Duplicate disposition is distinct from causal recovery: the canonical work
belongs to the earlier native original, even if processed later. Both status and snapshot import verify the
stored association, so an unrelated successful job cannot hide a failure.
"""
from __future__ import annotations

import json
import re

from .claims import ClaimError
from .db import utc_now


# Fixed aliases: j=failed job, d=review row, t=canonical successful job.
# Comparing the ordered part ledger also works after ordinary body retention;
# expiry does not turn a correctly reviewed duplicate back into unprocessed work.
_PARTS = """(SELECT json_group_array(json_array(part_index,body_sha256,body_bytes))
 FROM (SELECT part_index,body_sha256,body_bytes FROM event_parts
       WHERE event_id={event}.event_id AND revision={revision}.revision
       ORDER BY part_index))"""

VALID_DUPLICATE_RESOLUTION_SQL = """j.kind='extract' AND j.state='failed' AND j.last_error_code IS NOT NULL
 AND EXISTS (
 SELECT 1 FROM source_duplicate_resolutions d
 JOIN jobs t ON t.job_id=d.canonical_job_id
 JOIN event_revisions sr ON j.input_key=sr.event_id||'@'||CAST(sr.revision AS TEXT)
 JOIN events s ON s.event_id=sr.event_id
 JOIN event_revisions cr ON t.input_key=cr.event_id||'@'||CAST(cr.revision AS TEXT)
 JOIN events c ON c.event_id=cr.event_id
 WHERE d.failed_job_id=j.job_id AND t.job_id<>j.job_id
 AND t.kind='extract' AND t.state='succeeded'
 AND s.native_event_id GLOB 'msg:[0-9]*' AND c.native_event_id GLOB 'msg:[0-9]*'
 AND substr(s.native_event_id,5) NOT GLOB '*[^0-9]*'
 AND substr(c.native_event_id,5) NOT GLOB '*[^0-9]*'
 AND CAST(substr(c.native_event_id,5) AS INTEGER)<CAST(substr(s.native_event_id,5) AS INTEGER)
 AND s.event_id<>c.event_id AND s.host='hermes' AND c.host=s.host
 AND s.workspace_id=c.workspace_id AND s.instance_id=c.instance_id
 AND s.actor_id=c.actor_id AND s.session_id=c.session_id
 AND s.role=c.role AND s.origin_kind=c.origin_kind
 AND s.origin_kind IN ('user_input','assistant_reply','tool_execution')
 AND sr.project_ids_json=cr.project_ids_json
 AND sr.redaction_policy=cr.redaction_policy
 AND sr.part_count=cr.part_count AND sr.part_count>0
 AND sr.group_state='complete' AND cr.group_state='complete'
 AND length(d.proof_sha256)=64 AND d.proof_sha256 NOT GLOB '*[^0-9a-f]*'
 AND d.reviewer GLOB 'operator:*' AND length(d.reviewed_at)>0
 AND d.source_parts_json=SOURCE_PARTS AND d.source_parts_json=TARGET_PARTS
 AND (SELECT COUNT(*) FROM event_parts p WHERE p.event_id=s.event_id AND p.revision=sr.revision)=sr.part_count
 AND (SELECT COUNT(*) FROM event_parts p WHERE p.event_id=c.event_id AND p.revision=cr.revision)=cr.part_count
 )
""".replace('SOURCE_PARTS', _PARTS.format(event='s',revision='sr')).replace(
    'TARGET_PARTS', _PARTS.format(event='c',revision='cr'))


def _parts(conn, input_key):
    event_id, revision = input_key.rsplit('@', 1)
    return conn.execute(
        'SELECT part_index,body_sha256,body_bytes,availability,withheld '
        'FROM event_parts WHERE event_id=? AND revision=? ORDER BY part_index',
        (event_id, int(revision))).fetchall()


def record_native_duplicate(store, *, failed_job_id, canonical_job_id,
                            proof_sha256, reviewer):
    """Fence an explicit, source-verified operator disposition against the DB.

Caller MUST supply a previously persisted native UID/payload/summary proof.
There is no API or model input that calls this local maintenance interface.
"""
    if not isinstance(proof_sha256,str) or not re.fullmatch(r'[a-f0-9]{64}',proof_sha256):
        raise ClaimError('INVALID_REQUEST','native lineage proof fingerprint required')
    if not isinstance(reviewer,str) or not re.fullmatch(r'operator:[A-Za-z0-9._:-]{1,120}',reviewer):
        raise ClaimError('INVALID_REQUEST','explicit local operator reviewer required')
    with store.transaction() as conn:
        failed=conn.execute('SELECT * FROM jobs WHERE job_id=?',(failed_job_id,)).fetchone()
        target=conn.execute('SELECT * FROM jobs WHERE job_id=?',(canonical_job_id,)).fetchone()
        if failed is None or target is None or failed_job_id==canonical_job_id:
            raise ClaimError('INVALID_REQUEST','distinct existing source jobs required')
        if failed['kind']!='extract' or failed['state']!='failed' or target['kind']!='extract' or target['state']!='succeeded':
            raise ClaimError('INVALID_REQUEST','failed copy and successful canonical extraction required')
        source_parts=_parts(conn,failed['input_key']); target_parts=_parts(conn,target['input_key'])
        if not source_parts or any(p['availability']!='full' or p['withheld'] for p in [*source_parts,*target_parts]):
            raise ClaimError('STALE_CURSOR','full native source evidence required during review')
        parts_json=json.dumps([[p['part_index'],p['body_sha256'],p['body_bytes']] for p in source_parts],separators=(',',':'))
        other=json.dumps([[p['part_index'],p['body_sha256'],p['body_bytes']] for p in target_parts],separators=(',',':'))
        if parts_json!=other:
            raise ClaimError('STALE_CURSOR','source parts differ from reviewed canonical source')
        old=conn.execute('SELECT * FROM source_duplicate_resolutions WHERE failed_job_id=?',(failed_job_id,)).fetchone()
        if old and (old['canonical_job_id']!=canonical_job_id or old['source_parts_json']!=parts_json or old['proof_sha256']!=proof_sha256):
            raise ClaimError('EVENT_CONFLICT','existing native duplicate decision differs')
        inserted=conn.execute('INSERT OR IGNORE INTO source_duplicate_resolutions '
            '(failed_job_id,canonical_job_id,source_parts_json,proof_sha256,reviewer,reviewed_at) VALUES(?,?,?,?,?,?)',
            (failed_job_id,canonical_job_id,parts_json,proof_sha256,reviewer,utc_now())).rowcount
        valid=conn.execute('SELECT 1 FROM jobs j WHERE j.job_id=? AND ('+VALID_DUPLICATE_RESOLUTION_SQL+')',
                           (failed_job_id,)).fetchone()
        if valid is None:
            raise ClaimError('INVALID_REQUEST','native duplicate scope, part ledger or processing order does not match')
        if inserted:
            store.bump_watermark(conn,'native_duplicate_review',failed_job_id)
    return {'failed_job_id':failed_job_id,'canonical_job_id':canonical_job_id,
            'disposition':'already_processed_native_duplicate','inserted':bool(inserted),
            'source_and_failure_history_preserved':True,'model_calls':0}
