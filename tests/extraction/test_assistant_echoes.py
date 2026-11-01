"""A recalled answer is not a fresh confirmation; genuine new sources survive."""
import json
import pytest
from knowledge_weaver.store import receive_batch,fetch_source,current_items
from knowledge_weaver.extraction import load_event_group,sanitized_parts,run_worker_once
from knowledge_weaver.lifecycle import sweep
from .test_extraction import MockExtractor,_claim,_proposal
from ..store.conftest import make_event,make_batch
OLD={'value':'2020-01-01T00:00:00Z','precision':'instant','timezone':'UTC'}
NEW={'value':'2026-10-31T00:00:00Z','precision':'instant','timezone':'UTC'}

def ingest(store,name,*,role='user',origin='user_input',text='端口4200',when=OLD,**claim):
 event=make_event(native_id=name,role=role,origin_kind=origin,text=text,event_time=when)
 receive_batch(store,make_batch([event],batch_id='batch:'+name))
 part=sanitized_parts(load_event_group(store,event['event_id'],1))[0]
 model=MockExtractor(_proposal(_claim(part,assertion_kind='explicit_statement',**claim)))
 return event,model

def revisions(store):
 with store.read() as c:return [dict(v) for v in c.execute('SELECT revision_id,claim_id,value_json,authority,event_time_value FROM claim_revisions ORDER BY revision_id')]

def source_policy(store,event):
 with store.read() as c:return dict(c.execute("SELECT * FROM lifecycle_records WHERE subject_kind='event_part' AND subject_id=?",(event['event_id']+'#1@0',)).fetchone())

@pytest.mark.parametrize('assertion',['explicit_statement','inference'])
def test_same_confirmed_field_value_is_reused_not_reaged(store,assertion):
 original,model=ingest(store,'original');assert run_worker_once(store,model)['claims_written']==1;before=revisions(store)
 event,model=ingest(store,'assistant',role='assistant',origin='assistant_reply',text='根据统一记忆，端口4200。',when=NEW)
 model.proposal['claims'][0]['assertion_kind']=assertion
 result=run_worker_once(store,model);assert result['claims_written']==0 and result['claims_reused']==1
 assert revisions(store)==before
 policy=source_policy(store,event);assert policy['class']=='temporary_detail' and policy['business_anchor']==OLD['value'] and policy['expires_at']=='2020-01-31T00:00:00Z'
 assert fetch_source(store,event['event_id'],1,0)['event_time']==NEW
 with store.read() as c:detail=json.loads(c.execute('SELECT detail FROM job_attempts WHERE job_id=?',(result['job_id'],)).fetchone()[0])
 assert detail['reused_existing_claims'][0]['revision_id']==before[0]['revision_id']
 calls=len(model.calls);assert not run_worker_once(store,model)['claimed'] and len(model.calls)==calls
 assert sweep(store,now='2026-11-01T00:00:00Z')['event_parts']==2
 assert fetch_source(store,event['event_id'],1,0)['text'] is None
 assert fetch_source(store,original['event_id'],1,0)['text'] is None
 assert any(x['value']==4200 for x in current_items(store,workspace_id='owner:demo'))

@pytest.mark.parametrize('role,origin',[('user','user_input'),('tool','tool_execution'),('assistant','derived_import')])
def test_independent_confirmation_tool_observation_and_legacy_source_not_suppressed(store,role,origin):
 _,model=ingest(store,'original');run_worker_once(store,model)
 event,model=ingest(store,'new-source',role=role,origin=origin,text='端口4200',when=NEW)
 result=run_worker_once(store,model);assert result['claims_written']==1 and result.get('claims_reused',0)==0
 assert len(revisions(store))==2

@pytest.mark.parametrize('changes,text',[(dict(value=4300,display_text='端口4300'),'端口4300'),(dict(qualifiers={'rack':'B'}),'机架B的端口4200'),(dict(subject_hint='另一台服务器'),'另一台服务器的端口4200')])
def test_new_value_qualifier_or_entity_remains_new_information(store,changes,text):
 _,model=ingest(store,'original');run_worker_once(store,model)
 event,model=ingest(store,'new-information',role='assistant',origin='assistant_reply',text=text,when=NEW,**changes)
 result=run_worker_once(store,model);assert result['claims_written']==1 and result.get('claims_reused',0)==0
 assert len(revisions(store))==2


@pytest.mark.parametrize('unknown',[{'value':None,'precision':'unknown','timezone':'UTC'},{'value':'2099-01-01T00:00:00Z','precision':'instant','timezone':'UTC'}])
@pytest.mark.parametrize('origin',['user_input','derived_import'])
def test_unknown_original_clock_not_replaced_with_reply_time(store,unknown,origin):
 _,model=ingest(store,'original',when=unknown,origin=origin);run_worker_once(store,model)
 event,model=ingest(store,'reply',role='assistant',origin='assistant_reply',when=NEW)
 result=run_worker_once(store,model);assert result['claims_reused']==1
 policy=source_policy(store,event);assert policy['business_anchor'] is None and policy['expires_at'] is None and policy['class']=='unclassified'
 with store.read() as c:
  current=c.execute('SELECT state,selected_revision_ids_json FROM current_claims').fetchone();assert current['state']=='uncertain' and json.loads(current['selected_revision_ids_json'])==[]


def test_reuse_audit_and_lifecycle_share_the_fenced_transaction(store,monkeypatch):
 _,model=ingest(store,'original');run_worker_once(store,model);before=revisions(store)
 event,model=ingest(store,'reply',role='assistant',origin='assistant_reply',when=NEW)
 import knowledge_weaver.lifecycle.bootstrap as module
 original=module.finalize_sources_in_transaction
 def broken(conn,**kwargs):original(conn,**kwargs);raise OSError('synthetic transaction failure')
 monkeypatch.setattr(module,'finalize_sources_in_transaction',broken)
 with pytest.raises(OSError):run_worker_once(store,model)
 assert revisions(store)==before
 with store.read() as c:
  rows=list(c.execute('SELECT detail FROM job_attempts WHERE job_id IN (SELECT job_id FROM jobs WHERE input_key LIKE ?)',(event['event_id']+'@%',)))
  assert rows and all(x[0] is None for x in rows)
 assert fetch_source(store,event['event_id'],1,0)['text']=='端口4200'


def test_reply_cannot_reconfirm_an_existing_tool_verified_fact(store):
 _,model=ingest(store,'tool-original',role='tool',origin='tool_execution');run_worker_once(store,model);before=revisions(store);assert before[0]['authority']=='tool_verified'
 _,model=ingest(store,'reply',role='assistant',origin='assistant_reply',when=NEW)
 result=run_worker_once(store,model);assert result['claims_written']==0 and result['claims_reused']==1 and revisions(store)==before


@pytest.mark.parametrize('kind,origin,expected',[('inference','assistant_reply','inferred'),('explicit_statement','assistant_reply','reported'),('explicit_statement','derived_import','legacy_unknown')])
def test_repeating_weak_remembered_information_does_not_upgrade_its_grade(store,kind,origin,expected):
 event,model=ingest(store,'weak-original',role='assistant',origin=origin)
 model.proposal['claims'][0]['assertion_kind']=kind;run_worker_once(store,model);before=revisions(store);assert before[0]['authority']==expected
 _,model=ingest(store,'reply',role='assistant',origin='assistant_reply',when=NEW)
 result=run_worker_once(store,model);assert result['claims_written']==0 and result['claims_reused']==1 and revisions(store)==before


def test_new_completion_report_after_a_task_hypothesis_is_preserved(store):
 value={'status':'reported_done','conditions':[],'satisfied':[]}
 _,model=ingest(store,'task-guess',role='assistant',origin='assistant_reply',text='猜测恢复演练可能完成',predicate='task.status',subject_hint='恢复演练',facet='reported',value=value,display_text='恢复演练可能完成')
 model.proposal['claims'][0]['assertion_kind']='inference';run_worker_once(store,model)
 _,model=ingest(store,'task-report',role='assistant',origin='assistant_reply',when=NEW,text='我报告恢复演练已经完成，但尚未独立核验',predicate='task.status',subject_hint='恢复演练',facet='reported',value=value,display_text='恢复演练已报告完成，未核验')
 model.proposal['claims'][0]['assertion_kind']='completion_report';result=run_worker_once(store,model)
 assert result['claims_written']==1 and result.get('claims_reused',0)==0
 assert {v['authority'] for v in revisions(store)}=={'inferred','reported'}
