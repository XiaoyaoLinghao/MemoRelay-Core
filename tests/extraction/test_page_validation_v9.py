import copy
import json
import pytest
from knowledge_weaver.store import receive_batch, current_items
from knowledge_weaver.extraction import run_worker_once
from knowledge_weaver.extraction import load_event_group, sanitized_parts
from knowledge_weaver.extraction.page_validation import evidence_audit, review_notes
from ..store.conftest import make_event, make_batch
from .test_extraction import _claim, _proposal
from .test_extraction import MockExtractor


def test_v9_wrong_source_reference_is_repaired_before_any_write(store):
    event=make_event(text='端口4200');receive_batch(store,make_batch([event]))
    calls=[]
    class Model:
        def extract(self,parts,context):
            calls.append(copy.deepcopy(context));p=_proposal(_claim(parts[0]))
            if len(calls)==1:p['claims'][0]['evidence'][0]['event_id']='evt:'+'b'*64
            return p
    result=run_worker_once(store,Model())
    assert result['state']=='succeeded' and result['claims_written']==1
    feedback=calls[1]['page_validation_feedback']
    assert feedback['category']=='evidence_invalid'
    assert feedback['evidence_audit']==[{'claim_index':0,'reference_index':0,'source_exists':False,'exact_substring':False,'over_quote_limit':False}]
    assert len(current_items(store,workspace_id='owner:demo'))==1


def test_v9_nonverbatim_reference_stays_blocked_no_partial_success(store):
    event=make_event(text='端口4200');receive_batch(store,make_batch([event]))
    class Model:
        def extract(self,parts,context):
            return _proposal(_claim(parts[0]),_claim(parts[0],quote='不存在的证据',predicate='other'))
    out=run_worker_once(store,Model())
    assert out['state']=='retry_wait' and out['failure_category']=='evidence_invalid'
    assert current_items(store,workspace_id='owner:demo')==[]


def test_reference_audit_checks_every_reference_but_bounds_feedback(store):
    event=make_event(text='端口4200');receive_batch(store,make_batch([event]))
    parts=sanitized_parts(load_event_group(store,event['event_id'],1))
    proposal=_proposal(_claim(parts[0]))
    valid=proposal['claims'][0]['evidence'][0]
    proposal['claims'][0]['evidence']=[copy.deepcopy(valid) for _ in range(100)]+[{**valid,'support_quote':'wrong'}]
    assert evidence_audit(proposal,parts)==[{'claim_index':0,'reference_index':100,'source_exists':True,'exact_substring':False,'over_quote_limit':False}]


def test_v9_specific_critique_is_bounded_ephemeral_and_not_suppressed(store):
    event=make_event();receive_batch(store,make_batch([event]))
    calls=[]
    class Model:
        def extract(self,parts,context):
            calls.append(copy.deepcopy(context));p=_proposal(_claim(parts[0]))
            p['warnings']=['Same business time has conflicting ports: operator review required.']
            return p
    out=run_worker_once(store,Model())
    assert out['state']=='retry_wait' and out['error']=='MODEL_REVIEW_REQUIRED'
    assert calls[1]['page_validation_feedback']['untrusted_review_notes']==['Same business time has conflicting ports: operator review required.']
    assert current_items(store,workspace_id='owner:demo')==[]
    with store.read() as c:
        persisted=c.execute('SELECT metadata_json FROM extraction_failure_diagnostics').fetchone()[0]
    assert 'conflicting ports' not in persisted and 'operator review' not in persisted
    assert len(json.dumps(review_notes(['文字'*200]*20),ensure_ascii=False).encode())<2200
    assert review_notes(['password=CANARY_MUST_NOT_LEAVE'])==[]


def test_v9_oversize_wrong_quote_reports_both_errors_without_editing_input(store):
    event=make_event(text='端口4200');receive_batch(store,make_batch([event]))
    feedback=[]
    class Model:
        def extract(self,parts,context):
            if 'page_validation_feedback' in context:feedback.append(context['page_validation_feedback'])
            return _proposal(_claim(parts[0],quote='not-verbatim-'*200))
    out=run_worker_once(store,Model())
    assert out['failure_category']=='schema_invalid'
    audit=feedback[0]['evidence_audit'][0]
    assert audit['source_exists'] and not audit['exact_substring'] and audit['over_quote_limit']
    assert current_items(store,workspace_id='owner:demo')==[]


def test_rejected_target_is_repaired_instead_of_zero_write_success(store):
    old=make_event(native_id='old-anchor',text='端口4200');receive_batch(store,make_batch([old]))
    part=sanitized_parts(load_event_group(store,old['event_id'],1))[0]
    assert run_worker_once(store,MockExtractor(_proposal(_claim(part))))['state']=='succeeded'
    anchor=current_items(store,workspace_id='owner:demo')[0]['claim_id']
    new=make_event(native_id='new-anchor',text='端口改为4300');receive_batch(store,make_batch([new],batch_id='new-anchor'))
    calls=[]
    class Model:
        def extract(self,parts,context):
            calls.append(copy.deepcopy(context))
            return _proposal(_claim(parts[0],target_claim_hint=anchor,value=4300,
                                    predicate='server.address' if len(calls)==1 else 'server.port',display_text='端口改为4300'))
    outcome=run_worker_once(store,Model())
    assert outcome['state']=='succeeded' and outcome['claims_written']==1
    assert calls[1]['page_validation_feedback']['resolver_issues']==[{'claim_index':0,'code':'target_incompatible'}]
    fragment=calls[1]['page_validation_feedback']['untrusted_previous_claims'][0]
    assert fragment['claim_index']==0 and fragment['fields']['predicate']=='server.address'
    assert fragment['fields']['target_claim_hint']==anchor


def test_model_schema_exposes_exact_scoped_identifiers_without_mutating_contract(store):
    from knowledge_weaver import contracts
    from knowledge_weaver.extraction import HttpExtractor
    event=make_event(text='端口4200');receive_batch(store,make_batch([event]))
    parts=sanitized_parts(load_event_group(store,event['event_id'],1))
    context={'prompt_version':'prompt:9','project_ids':['project:songta'],'known_claims':[{'claim_id':'claim:fixture'}]}
    client=HttpExtractor('http://fixture.invalid',model='fixture')
    schema=client._request_schema(parts,context)
    fields=schema['properties']['claims']['items']['properties']
    assert fields['project_hint']['enum']==[None,'project:songta']
    assert fields['target_claim_hint']['enum']==[None,'claim:fixture']
    assert fields['evidence']['items']['properties']['event_id']['enum']==[event['event_id']]
    assert 'enum' not in contracts.extraction_schema()['properties']['claims']['items']['properties']['project_hint']
    assert client._request_schema(parts,{'prompt_version':'prompt:8'})==contracts.extraction_schema()


def test_previous_claim_fragments_are_bounded_and_exclude_secret_fields():
    from knowledge_weaver.extraction.page_validation import previous_claim_fragments
    proposal={'claims':[{'predicate':'test','qualifiers':{'password':'SYNTHETIC_CANARY_ONLY'},'evidence':['x'*10000],'subject_hint':'x'*8000} for _ in range(20)]}
    feedback={'resolver_issues':[{'claim_index':x,'code':'target_incompatible'} for x in range(20)]}
    result=previous_claim_fragments(proposal,feedback)
    encoded=json.dumps(result,ensure_ascii=False)
    assert len(encoded.encode())<8192 and len(result)<=8
    assert 'SYNTHETIC_CANARY_ONLY' not in encoded
    assert all(x['fields']=={'predicate':'test'} for x in result)
