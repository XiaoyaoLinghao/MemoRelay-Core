"""A model-disclaimed proposal is not a confirmed fact; retries stay bounded."""
import json
from knowledge_weaver.extraction import run_worker_once
from knowledge_weaver.store import current_items,fetch_source
from .test_source_lifecycle import setup,lifecycle
from .test_extraction import MockExtractor,_claim,_proposal

def test_warning_withholds_values_then_retry_repairs_from_original_evidence(store):
 event,subject,part=setup(store)
 bad=_proposal(_claim(part,assertion_kind='explicit_statement'));bad['warnings']=['CANARY-MODEL-WARNING-NOT-FOR-LOGS']
 model=MockExtractor(bad);result=run_worker_once(store,model)
 assert result['state']=='retry_wait' and result['error']=='MODEL_REVIEW_REQUIRED'
 assert not current_items(store,workspace_id='owner:demo') and lifecycle(store,subject)['state']=='bootstrap_pending'
 with store.read() as c:
  job=dict(c.execute("SELECT * FROM jobs WHERE kind='extract'").fetchone());attempts=[dict(x) for x in c.execute('SELECT * FROM job_attempts')]
 assert 'CANARY-MODEL-WARNING' not in json.dumps(attempts)
 good=MockExtractor(_proposal(_claim(part,assertion_kind='explicit_statement')))
 fixed=run_worker_once(store,good,now=job['run_after']);assert fixed['claims_written']==1
 assert 'MODEL_REVIEW_REQUIRED' in good.calls[0][1]['previous_validation_error']
 assert good.calls[0][0][0].text==part.text
 assert lifecycle(store,subject)['business_anchor']==event['event_time']['value']
 calls=len(good.calls);assert not run_worker_once(store,good)['claimed'] and len(good.calls)==calls

def test_persistent_model_warning_stops_after_three_attempts_without_loss(store):
 event,subject,part=setup(store);bad=_proposal(_claim(part,assertion_kind='explicit_statement'));bad['warnings']=['still uncertain'];model=MockExtractor(bad);now=None
 for i in range(3):
  result=run_worker_once(store,model,now=now)
  with store.read() as c:job=dict(c.execute("SELECT * FROM jobs WHERE kind='extract'").fetchone())
  now=job['run_after']
 assert result['state']=='failed' and len(model.calls)==3
 assert not run_worker_once(store,model)['claimed'] and not current_items(store,workspace_id='owner:demo')
 assert fetch_source(store,event['event_id'],1,0)['text']==part.text
 assert lifecycle(store,subject)['state']=='bootstrap_pending'


def test_zero_claim_warning_is_a_one_call_noop_and_keeps_source_protected(store):
 event,subject,part=setup(store);proposal=_proposal();proposal['excluded_event_ids']=[event['event_id']];proposal['warnings']=['No new durable information']
 model=MockExtractor(proposal);result=run_worker_once(store,model)
 assert result['state']=='succeeded' and result['claims_written']==0
 assert not current_items(store,workspace_id='owner:demo') and lifecycle(store,subject)['state']=='bootstrap_pending'
 assert not run_worker_once(store,model)['claimed'] and len(model.calls)==1
 assert fetch_source(store,event['event_id'],1,0)['text']==part.text

def test_existing_fact_reuse_with_warning_has_no_new_write_or_retry(store):
 from .test_assistant_echoes import ingest,revisions
 _,model=ingest(store,'original');run_worker_once(store,model);before=revisions(store)
 event,model=ingest(store,'repeat',role='assistant',origin='assistant_reply');model.proposal['warnings']=['Recalled existing value; no fresh confirmation']
 result=run_worker_once(store,model);assert result['state']=='succeeded' and result['claims_written']==0 and result['claims_reused']==1
 assert revisions(store)==before and not run_worker_once(store,model)['claimed'] and len(model.calls)==1
