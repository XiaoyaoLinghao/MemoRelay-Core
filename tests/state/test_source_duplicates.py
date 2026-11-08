"""Native duplicate maintenance preserves failed history and canonical facts."""
import hashlib
import json

import pytest

from knowledge_weaver.store import Store, receive_batch, current_items, fetch_source
from knowledge_weaver.store.claims import ClaimError
from knowledge_weaver.store.source_duplicates import record_native_duplicate
from knowledge_weaver.store.status import _job_pipeline
from knowledge_weaver.extraction import run_worker_once, load_event_group, sanitized_parts
from knowledge_weaver.extraction.proposals import ExtractionError
from knowledge_weaver.portability import export_snapshot, import_snapshot
from ..store.conftest import make_event, make_batch
from ..extraction.test_extraction import _claim, _proposal, MockExtractor


class Fail:
    def extract(self, parts, context):
        raise ExtractionError('MODEL_UNAVAILABLE','synthetic invalid evidence',retryable=False,category='evidence_invalid')


def pair(store, *, changed=None):
    source=make_event(native_id='msg:1',text='The measured file line count is 42.',role='tool',origin_kind='tool_execution')
    receive_batch(store,make_batch([source],batch_id='batch:canonical'))
    part=sanitized_parts(load_event_group(store,source['event_id'],1))[0]
    first=run_worker_once(store,MockExtractor(_proposal(_claim(part,subject_hint='file:x',predicate='file.line_count',value=42,display_text=part.text,facet='observed',assertion_kind='observation'))))
    assert first['state']=='succeeded'
    options=dict(native_id='msg:3',text=source['body']['text'],role='tool',origin_kind='tool_execution')
    options.update(changed or {})
    copy=make_event(**options)
    receive_batch(store,make_batch([copy],batch_id='batch:copy'))
    second=run_worker_once(store,Fail())
    assert second['state']=='failed'
    return source,copy,first['job_id'],second['job_id']


def review(store,target,failed,**kw):
    args=dict(failed_job_id=failed,canonical_job_id=target,
              proof_sha256=hashlib.sha256(b'synthetic native UID plus payload plus summary proof').hexdigest(),
              reviewer='operator:test')
    args.update(kw)
    return record_native_duplicate(store,**args)


def history(store):
    with store.read() as conn:
        return {name:[tuple(x) for x in conn.execute('SELECT * FROM '+name+' ORDER BY rowid')]
                for name in ('jobs','job_attempts','events','event_revisions','event_parts','claim_revisions')}


def test_review_reuses_completed_work_without_altering_source_or_failure(store):
    original,copy,target,failed=pair(store)
    before=history(store);facts=current_items(store,workspace_id='owner:demo')
    assert _job_pipeline(store,'extract')['failed']==1
    result=review(store,target,failed)
    assert result['inserted'] and result['model_calls']==0
    assert history(store)==before
    assert current_items(store,workspace_id='owner:demo')==facts
    assert fetch_source(store,copy['event_id'],1,0)['text']==original['body']['text']
    assert _job_pipeline(store,'extract')['failed']==0
    with store.read() as conn:
        row=conn.execute('SELECT state,recovered_by_job_id,last_error_code FROM jobs WHERE job_id=?',(failed,)).fetchone()
        assert tuple(row)==('failed',None,'MODEL_UNAVAILABLE')
    watermark=store.watermark()
    assert not review(store,target,failed)['inserted']
    assert store.watermark()==watermark


@pytest.mark.parametrize('changed',[
    {'text':'The measured file line count is 43.'},
    {'actor_id':'other-agent'},
    {'instance_id':'instance:other'},
    {'session_id':'session:unrelated'},
    {'role':'assistant','origin_kind':'assistant_reply'},
    {'project_ids':('project:other',)},
])
def test_different_source_or_identity_is_not_a_processed_duplicate(store,changed):
    _,_,target,failed=pair(store,changed=changed)
    before=history(store)
    with pytest.raises(ClaimError):review(store,target,failed)
    assert history(store)==before
    assert _job_pipeline(store,'extract')['failed']==1
    with store.read() as conn:assert conn.execute('SELECT COUNT(*) FROM source_duplicate_resolutions').fetchone()[0]==0


def test_operator_proof_and_unchanged_review_required(store):
    _,_,target,failed=pair(store)
    with pytest.raises(ClaimError):review(store,target,failed,proof_sha256='')
    with pytest.raises(ClaimError):review(store,target,failed,reviewer='model:test')
    with pytest.raises(ClaimError):review(store,failed,failed)
    review(store,target,failed)
    with pytest.raises(ClaimError):review(store,target,failed,proof_sha256='0'*64)


def test_snapshot_restores_duplicate_disposition_separately_from_causal_recovery(store,tmp_path):
    _,copy,target,failed=pair(store);review(store,target,failed)
    export_snapshot(store,tmp_path/'snapshot')
    assert json.loads((tmp_path/'snapshot/manifest.json').read_text())['snapshot_schema_version']==5
    import_snapshot(tmp_path/'snapshot',tmp_path/'restored')
    restored=Store(tmp_path/'restored/knowledge.db')
    assert _job_pipeline(restored,'extract')['failed']==0
    with restored.read() as conn:
        assert conn.execute('SELECT COUNT(*) FROM source_duplicate_resolutions').fetchone()[0]==1
        assert conn.execute('SELECT state FROM jobs WHERE job_id=?',(failed,)).fetchone()[0]=='failed'
    assert fetch_source(restored,copy['event_id'],1,0)['text']==copy['body']['text']


def test_invalidated_canonical_success_cannot_hide_failure(store):
    _,_,target,failed=pair(store);review(store,target,failed)
    with store.transaction() as conn:
        conn.execute("UPDATE jobs SET state='failed',last_error_code='SYNTHETIC_FAILURE' WHERE job_id=?",(target,))
    assert _job_pipeline(store,'extract')['failed']==2


def test_changed_part_fingerprint_reopens_duplicate_as_unresolved(store):
    _,copy,target,failed=pair(store);review(store,target,failed)
    with store.transaction() as conn:
        conn.execute('UPDATE event_parts SET body_sha256=? WHERE event_id=?',('0'*64,copy['event_id']))
    assert _job_pipeline(store,'extract')['failed']==1
