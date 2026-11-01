"""Fact lifetime does not make full old raw logs permanent (AC18/19/21)."""
import datetime as dt
import json
import pytest
from knowledge_weaver.extraction import run_worker_once,load_event_group,sanitized_parts
from knowledge_weaver.lifecycle import sweep,register_lifecycle
from knowledge_weaver.store import receive_batch,current_items,fetch_source
from .test_extraction import MockExtractor,_claim,_proposal
from ..store.conftest import make_event,make_batch
OLD={'value':'2020-01-01T00:00:00Z','precision':'instant','timezone':'UTC'}
NEW={'value':'2026-10-31T00:00:00Z','precision':'instant','timezone':'UTC'}
def source(store,name,text,when=OLD,role='user',origin='user_input'):
 e=make_event(native_id=name,text=text,event_time=when,role=role,origin_kind=origin);receive_batch(store,make_batch([e],batch_id='batch:'+name));return e,sanitized_parts(load_event_group(store,e['event_id'],1))[0]
def policy(store,event):
 with store.read() as c:return dict(c.execute("SELECT * FROM lifecycle_records WHERE subject_kind='event_part' AND subject_id=?",(event['event_id']+'#1@0',)).fetchone())
@pytest.mark.parametrize('days,expired',[(29,False),(30,True),(31,True)])
def test_valid_fact_survives_original_full_detail_expiry(store,days,expired):
 e,p=source(store,'raw','端口4200；无价值诊断日志UNIMPORTANT-OLD-DETAIL')
 model=MockExtractor(_proposal(_claim(p,quote='端口4200',assertion_kind='explicit_statement')));assert run_worker_once(store,model)['claims_written']==1
 before=policy(store,e);assert before['class']=='temporary_detail' and before['expires_at']=='2020-01-31T00:00:00Z'
 now=(dt.datetime(2020,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=days)).isoformat()
 sweep(store,now=now);raw=fetch_source(store,e['event_id'],1,0);assert (raw['text'] is None)==expired
 items=current_items(store,workspace_id='owner:demo');assert len(items)==1 and items[0]['value']==4200 and items[0]['source_detail_expired']==expired and items[0]['state']=='current'
 assert items[0]['event_time']==OLD and raw['event_time']==OLD
 assert not run_worker_once(store,model)['claimed'] and len(model.calls)==1

def test_mixed_new_information_and_recalled_fact_do_not_reage_old_raw_detail(store):
 original,p=source(store,'original','端口4200');run_worker_once(store,MockExtractor(_proposal(_claim(p,assertion_kind='explicit_statement'))))
 reply,p=source(store,'reply','记忆中的端口4200；新的备份周期每周',NEW,'assistant','assistant_reply')
 proposal=_proposal(_claim(p,quote='端口4200',assertion_kind='explicit_statement'),_claim(p,quote='新的备份周期每周',predicate='backup.schedule',value='weekly',display_text='新的备份周期每周',assertion_kind='explicit_statement'))
 result=run_worker_once(store,MockExtractor(proposal));assert result['claims_written']==1 and result['claims_reused']==1
 row=policy(store,reply);assert row['business_anchor']==OLD['value'] and row['expires_at']=='2020-01-31T00:00:00Z'
 sweep(store,now='2026-11-01T00:00:00Z');assert fetch_source(store,reply['event_id'],1,0)['text'] is None
 values={x['predicate']:x for x in current_items(store,workspace_id='owner:demo')}
 assert values['server.port']['value']==4200 and values['server.port']['event_time']==OLD
 assert values['backup.schedule']['value']=='weekly' and values['backup.schedule']['event_time']==NEW
 assert all(x['source_detail_expired'] for x in values.values())

@pytest.mark.parametrize('days,expired',[(364,False),(365,True),(366,True)])
def test_superseded_fact_history_365_days_does_not_expire_current_configuration(store,days,expired):
 e,p=source(store,'old-port','端口4100');run_worker_once(store,MockExtractor(_proposal(_claim(p,value=4100,display_text='端口4100',assertion_kind='explicit_statement'))))
 old=current_items(store,workspace_id='owner:demo')[0]
 replacement={'value':'2020-02-01T00:00:00Z','precision':'instant','timezone':'UTC'}
 e,p=source(store,'correction','端口改为4200',replacement);run_worker_once(store,MockExtractor(_proposal(_claim(p,target_claim_hint=old['claim_id'],assertion_kind='explicit_correction'))))
 now=(dt.datetime(2020,2,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=days)).isoformat();sweep(store,now=now)
 with store.read() as c:
  row=dict(c.execute('SELECT status,display_text,value_json FROM claim_revisions WHERE revision_id=?',(old['revision_id'],)).fetchone())
  index_count=c.execute('SELECT COUNT(*) FROM claim_fts WHERE revision_id=?',(old['revision_id'],)).fetchone()[0]
 assert (row['status']=='expired')==expired
 assert (row['value_json']=='null')==expired and (index_count==0)==expired
 assert any(x['value']==4200 and x['event_time']==replacement for x in current_items(store,workspace_id='owner:demo'))

def test_explicit_source_detail_policy_is_preserved(store):
 e,p=source(store,'explicit','端口4200')
 register_lifecycle(store,subject_kind='event_part',subject_id=e['event_id']+'#1@0',class_='superseded_history',business_anchor=OLD['value'],state='bootstrap_pending',reason='explicit source policy')
 run_worker_once(store,MockExtractor(_proposal(_claim(p,assertion_kind='explicit_statement'))))
 row=policy(store,e);assert row['class']=='superseded_history' and row['expires_at']=='2020-12-31T00:00:00Z'


def test_echo_of_retained_fact_after_source_expiry_keeps_known_original_age(store):
 original,p=source(store,'aged-source','端口4200');run_worker_once(store,MockExtractor(_proposal(_claim(p,assertion_kind='explicit_statement'))))
 sweep(store,now='2026-10-30T00:00:00Z');assert current_items(store,workspace_id='owner:demo')[0]['state']=='current'
 reply,p=source(store,'aged-reply','记忆中端口4200',NEW,'assistant','assistant_reply')
 result=run_worker_once(store,MockExtractor(_proposal(_claim(p,quote='端口4200',assertion_kind='explicit_statement'))))
 assert result['claims_written']==0 and result['claims_reused']==1
 row=policy(store,reply);assert row['business_anchor']==OLD['value'] and row['expires_at']=='2020-01-31T00:00:00Z'
 sweep(store,now='2026-11-01T00:00:00Z');assert fetch_source(store,reply['event_id'],1,0)['text'] is None
 assert any(x['value']==4200 and x['event_time']==OLD for x in current_items(store,workspace_id='owner:demo'))
