import json
from types import SimpleNamespace
import pytest
from knowledge_weaver import contracts
from knowledge_weaver.store import receive_batch, current_items
from knowledge_weaver.extraction import run_worker_once, load_event_group, sanitized_parts, ExtractionError
from knowledge_weaver.extraction.page_validation import schema_details
from knowledge_weaver.extraction.diagnostics import failure_detail
from ..store.conftest import make_event, make_batch
from .test_extraction import _claim, _proposal


def test_schema_repair_uses_same_source_without_echoing_unknown_key(store):
    event=make_event(text='端口4200');receive_batch(store,make_batch([event]))
    calls=[]
    class Model:
        def extract(self,parts,context):
            calls.append((parts,context))
            p=_proposal(_claim(parts[0]))
            if len(calls)==1:p['CANARY_PRIVATE_KEY']='do-not-copy'
            return p
    assert run_worker_once(store,Model())['state']=='succeeded'
    assert len(calls)==2 and calls[0][0]==calls[1][0]
    feedback=calls[1][1]['page_validation_feedback']
    assert feedback['category']=='schema_invalid'
    assert 'CANARY_PRIVATE_KEY' not in json.dumps(feedback)
    assert any(x['code']=='unknown_property' for x in feedback['issues'])
    assert len(current_items(store,workspace_id='owner:demo'))==1


def test_unresolved_schema_is_bounded_and_retains_diagnostics_no_partial_write(store):
    event=make_event();receive_batch(store,make_batch([event]))
    calls=[]
    class Model:
        def extract(self,parts,context):calls.append(1);return {'CANARY_PRIVATE_KEY':'never-copy'}
    outcome=run_worker_once(store,Model())
    assert outcome['state']=='retry_wait' and outcome['failure_category']=='schema_invalid'
    assert len(calls)==2 and current_items(store,workspace_id='owner:demo')==[]
    with store.read() as c:
        meta=c.execute('SELECT metadata_json FROM extraction_failure_diagnostics').fetchone()[0]
    assert 'CANARY_PRIVATE_KEY' not in meta and 'never-copy' not in meta
    assert json.loads(meta)['validation']['total_issue_count']>0


def test_real_model_warning_is_not_suppressed(store):
    event=make_event();receive_batch(store,make_batch([event]))
    class Model:
        calls=0
        def extract(self,parts,context):
            self.calls+=1;p=_proposal(_claim(parts[0]));p['warnings']=['Ambiguous contradictory source'];return p
    m=Model();outcome=run_worker_once(store,m)
    assert outcome['state']=='retry_wait' and outcome['error']=='MODEL_REVIEW_REQUIRED'
    assert m.calls==2 and current_items(store,workspace_id='owner:demo')==[]


def test_long_retry_keeps_successful_siblings_despite_changed_feedback(store):
    text='HEAD端口4317。'+('没有新信息。'*1800)+'TAIL保留47天。'
    event=make_event(text=text);receive_batch(store,make_batch([event]))
    head_calls=0;tail_calls=0
    class Model:
        def extract(self,parts,context):
            nonlocal head_calls,tail_calls
            body=parts[0].text
            if 'HEAD端口4317' in body:
                head_calls+=1
                return _proposal(_claim(parts[0],quote='HEAD端口4317',predicate='fixture.head',value=4317,assertion_kind='explicit_statement'))
            if 'TAIL保留47天' in body:
                tail_calls+=1
                p=_proposal(_claim(parts[0],quote='TAIL保留47天',predicate='fixture.tail',value=47,assertion_kind='explicit_statement'))
                if tail_calls<=2:p['warnings']=['Unresolved ambiguity']
                return p
            return {'schema_version':'1.0','claims':[],'excluded_event_ids':[event['event_id']],'warnings':[]}
    first=run_worker_once(store,Model());assert first['state']=='retry_wait'
    assert current_items(store,workspace_id='owner:demo')==[]
    with store.read() as c:due=c.execute('SELECT run_after FROM jobs WHERE job_id=?',(first['job_id'],)).fetchone()[0]
    second=run_worker_once(store,Model(),now=due)
    assert second['state']=='succeeded' and head_calls==1 and tail_calls==3
    assert {i['value'] for i in current_items(store,workspace_id='owner:demo')}=={4317,47}
    with store.read() as c:
        receipt=json.loads(c.execute('SELECT coverage_json FROM extraction_segment_receipts').fetchone()[0])
    assert receipt['complete'] and receipt['covered_bytes']==len(text.encode())


def test_schema_metadata_bounds_and_secret_classification():
    items=[SimpleNamespace(path='$.claims[4].value.TOP_SECRET',message="unknown property 'TOP_SECRET'") for _ in range(20)]
    d=schema_details(items)
    assert len(d['issues'])==8 and d['total_issue_count']==20
    assert 'TOP_SECRET' not in json.dumps(d)
    stored=json.loads(failure_detail('secret_content',validation=d,segment_ranges=[{'start_byte':1,'raw':'TOP_SECRET'}]))
    assert stored['category']=='secret_content' and stored['segment_ranges']==[{'start_byte':1}]
    assert 'TOP_SECRET' not in json.dumps(stored)


def test_schema_feedback_reports_actual_length_bound_and_json_type():
    issues=[SimpleNamespace(path='$.claims[1].evidence[0].support_quote',message='length must be <= 1200'),SimpleNamespace(path='$.claims',message="expected type ['array'], got str")]
    details=schema_details(issues)
    assert details['issues'][0]['code']=='upper_bound' and details['issues'][0]['bound']==1200
    assert details['issues'][1]['expected_types']==['array']
    persisted=json.loads(failure_detail('schema_invalid',validation=details))
    assert persisted['validation']['issues'][0]['comparison']=='<='
    assert persisted['validation']['issues'][1]['expected_types']==['array']
