"""Normal entity continuity through actual inbox->model->validated commit."""
import json
import pytest
from knowledge_weaver.store import receive_batch,current_items
from knowledge_weaver.extraction import load_event_group,sanitized_parts,run_extraction_job,ExtractionError
from knowledge_weaver.jobs import claim as claim_jobs
from knowledge_weaver.store.claims import record_claim_revision
from ..state.conftest import evidence_ref
from ..store.conftest import make_event,make_batch
from ..state.conftest import record
from .test_extraction import MockExtractor,_claim,_proposal

def propose(store,event,processor=None,**overrides):
 receive_batch(store,make_batch([event],batch_id='batch:'+event['source']['native_event_id']));part=sanitized_parts(load_event_group(store,event['event_id'],1))[0];c=_claim(part,**({'assertion_kind':'explicit_statement'}|overrides));m=MockExtractor(_proposal(c));
 if processor:
  with store.transaction() as conn:conn.execute("UPDATE jobs SET prompt_version=? WHERE input_key=?",(processor,event['event_id']+'@1'))
 jobs=claim_jobs(store,worker_id='pilot:test',kinds=('extract',),limit=1000);j=next(x for x in jobs if x['input_key']==event['event_id']+'@1');r=run_extraction_job(store,j,m);return r,m

def test_same_entity_reuses_backend_identity_despite_different_subject_words(store):
 first=record(store,subject_id='subject:canonical',display_text='云杉项目服务端口4387',value=4387)
 e=make_event(native_id='message:next',text='云杉服务器服务端口更正为4399',event_time={'value':'2026-10-31T02:00:00Z','precision':'instant','timezone':'UTC'})
 r,m=propose(store,e,subject_hint='云杉项目服务器',target_claim_hint=first['claim_id'],value=4399)
 items=[x for x in current_items(store,workspace_id='owner:demo') if x['predicate']=='server.port' and x['facet']=='desired']
 assert len(items)==1 and items[0]['subject_id']=='subject:canonical' and items[0]['value']==4399,items
 with store.read() as c:assert c.execute('SELECT COUNT(*) FROM claim_revisions WHERE claim_id=?',(first['claim_id'],)).fetchone()[0]==2

def test_observed_and_desired_share_entity_but_keep_separate_properties(store):
 old=record(store,subject_id='subject:canonical',display_text='云杉端口期望4399',value=4399)
 e=make_event(native_id='message:measurement',text='云杉实测端口4398',role='tool',origin_kind='tool_execution')
 propose(store,e,subject_hint='云杉服务器',target_claim_hint=old['claim_id'],value=4398,facet='observed')
 items=[x for x in current_items(store,workspace_id='owner:demo') if x['predicate']=='server.port'];assert len(items)==2
 assert {(x['subject_id'],x['facet'],x['value']) for x in items}=={('subject:canonical','desired',4399),('subject:canonical','observed',4398)}

@pytest.mark.parametrize('processor',['prompt:8','prompt:9'])
@pytest.mark.parametrize('case',['other-workspace','other-project','other-property','other-qualifiers','unknown-id'])
def test_target_must_be_actual_compatible_scoped_field(store,case,processor):
 kwargs={}
 if case=='other-workspace':kwargs['workspace_id']='owner:other'
 if case=='other-project':kwargs['project_id']='project:other'
 if case=='other-property':kwargs['predicate']='backup.schedule'
 if case=='other-qualifiers':kwargs['qualifiers']={'protocol':'https'}
 if case=='other-workspace':
  e0=make_event(native_id='message:foreign',workspace_id='owner:other',text='existing endpoint');receive_batch(store,make_batch([e0]));old=record_claim_revision(store,workspace_id='owner:other',project_id='project:songta',subject_id='subject:unrelated',predicate='server.port',facet='desired',qualifiers={},value=4387,display_text='existing endpoint',authority='user_confirmed',assertion_kind='explicit_statement',event_time=e0['event_time'],evidence=[evidence_ref(e0)])
 else:old=record(store,subject_id='subject:unrelated',display_text='existing endpoint',value=4387,**kwargs)
 target=old['claim_id'] if case!='unknown-id' else 'claim:nonexistent'
 e=make_event(native_id='message:next',text='云杉服务端口4399')
 if processor=='prompt:8':
  r,m=propose(store,e,target_claim_hint=target,value=4399,processor=processor)
  assert r['claims_written']==0
 else:
  with pytest.raises(ExtractionError) as exc:propose(store,e,target_claim_hint=target,value=4399,processor=processor)
  assert exc.value.category=='proposal_invalid'
 with store.read() as c:
  assert c.execute('SELECT COUNT(*) FROM claim_revisions').fetchone()[0]==1
  assert c.execute('SELECT COUNT(*) FROM event_parts WHERE event_id=?',(e['event_id'],)).fetchone()[0]==1

def test_anchors_are_scoped_bounded_and_include_nonsecret_qualifiers(store):
 a=record(store,display_text='端口4387',value=4387,qualifiers={'operator_note':'local-only-note'})
 record(store,subject_id='subject:foreign',display_text='不能出站的外部项目',value=5,project_id='project:foreign')
 e=make_event(native_id='message:next',text='服务端口4399');r,m=propose(store,e,value=4399)
 context=m.calls[0][1];anchors=context['known_claims'];assert a['claim_id'] in {x['claim_id'] for x in anchors}
 assert 'local-only-note' in json.dumps(context) and '不能出站的外部项目' not in json.dumps(context,ensure_ascii=False)
 assert all(set(x)=={'claim_id','project_id','predicate','facet','label','qualifiers'} for x in anchors) and len(anchors)<=48

def test_existing_secret_label_is_omitted_from_model_context(store):
 canary='sk-'+'PILOT_SECRET_LABEL_8m2KqZ'*3
 old=record(store,subject_id='subject:legacy',display_text='Legacy connection '+canary,value=3)
 e=make_event(native_id='message:next',text='端口4399');r,m=propose(store,e,value=4399)
 assert canary not in json.dumps(m.calls[0][1]);assert old['claim_id'] not in {x['claim_id'] for x in m.calls[0][1]['known_claims']}

def test_old_relevant_field_not_starved_by_many_recent_unrelated_fields(store):
 old=record(store,subject_id='subject:old',display_text='罕见水杉端口4387',value=4387)
 for i in range(110):record(store,subject_id='subject:noise:'+str(i),predicate='noise.'+str(i),display_text='无关工单归档 '+str(i),value=i)
 e=make_event(native_id='message:next',text='罕见水杉端口4399');r,m=propose(store,e,target_claim_hint=old['claim_id'],value=4399)
 assert old['claim_id'] in {x['claim_id'] for x in m.calls[0][1]['known_claims']}
 assert r['claims_written']==1

def test_completion_report_cannot_bypass_task_rules_with_freeform_status_predicate(store):
 e=make_event(native_id="message:completion-alias",text="恢复演练已完成，尚未复核",role="assistant",origin_kind="assistant_reply")
 r,m=propose(store,e,predicate="recovery_drill.status",assertion_kind="completion_report",facet="reported",value={"status":"reported_done","conditions":[],"satisfied":[]})
 items=current_items(store,workspace_id="owner:demo")
 assert len(items)==1 and items[0]["predicate"]=="task.status"
 assert items[0]["value"]["status"]=="reported_done" and items[0]["evidence_grade"]=="reported"

def test_secret_in_anchor_qualifiers_is_omitted_before_model_call(store):
 canary="sk-"+"PILOT_QUALIFIER_SECRET_8m2KqZ"*3
 old=record(store,subject_id="subject:legacyqual",value=3,display_text="legacy scoped field",qualifiers={"operator_note":canary})
 e=make_event(native_id="message:qual-safe",text="端口4399");r,m=propose(store,e,value=4399)
 assert canary not in json.dumps(m.calls[0][1])
 assert old["claim_id"] not in {x["claim_id"] for x in m.calls[0][1]["known_claims"]}

def test_unbound_new_field_collision_keeps_existing_value_and_native_source(store):
 from knowledge_weaver.extraction import ExtractionError
 text="项目验收标记为晨雾"
 from knowledge_weaver.extraction.proposals import subject_id_for
 old2=record(store,subject_id=subject_id_for("project:songta","松塔服务器"),display_text=text,value="晨雾")
 e=make_event(native_id="message:collision",text="自动采集验收标记为竹影",event_time={"value":"2026-10-31T10:00:00Z","precision":"instant","timezone":"UTC"})
 # The fixture proposal's default subject_hint is read directly here so no
 # backend SQL or projection rewrite is needed for the actual assertion below.
 receive_batch(store,make_batch([e]));part=sanitized_parts(load_event_group(store,e["event_id"],1))[0]
 c=_claim(part,subject_hint="松塔服务器",value="竹影",assertion_kind="explicit_statement");m=MockExtractor(_proposal(c));j=next(x for x in claim_jobs(store,worker_id="collision",kinds=("extract",),limit=100) if x["input_key"]==e["event_id"]+"@1")
 with pytest.raises(ExtractionError,match="FIELD_IDENTITY_COLLISION"):run_extraction_job(store,j,m)
 assert any(x["value"]=="晨雾" for x in current_items(store,workspace_id="owner:demo"))
 with store.read() as c:assert c.execute("SELECT count(*) FROM event_parts WHERE event_id=?",(e["event_id"],)).fetchone()[0]==1

def test_identity_collision_retry_feedback_repairs_new_field_without_overwrite(store):
 from knowledge_weaver.extraction import run_worker_once
 from knowledge_weaver.extraction.proposals import subject_id_for
 old=record(store,subject_id=subject_id_for("project:songta","松塔服务器"),display_text="项目标记为晨雾",value="晨雾")
 while run_worker_once(store,MockExtractor(_proposal()))["claimed"]:pass
 e=make_event(native_id="message:repair-new-field",text="自动采集标记为竹影",event_time={"value":"2026-10-31T10:00:00Z","precision":"instant","timezone":"UTC"})
 receive_batch(store,make_batch([e]));part=sanitized_parts(load_event_group(store,e["event_id"],1))[0]
 bad=_claim(part,subject_hint="松塔服务器",value="竹影",assertion_kind="explicit_statement")
 first=run_worker_once(store,MockExtractor(_proposal(bad)))
 assert first["state"]=="retry_wait"
 assert {x["value"] for x in current_items(store,workspace_id="owner:demo")}=={"晨雾"}
 with store.read() as c:
  due=c.execute("SELECT run_after FROM jobs WHERE job_id=?",(first["job_id"],)).fetchone()[0]
  detail=c.execute("SELECT detail FROM job_attempts WHERE job_id=?",(first["job_id"],)).fetchone()[0]
 assert detail.startswith("FIELD_IDENTITY_COLLISION:")
 good=MockExtractor(_proposal(dict(bad,qualifiers={"scope":"automatic_capture"})))
 resumed=run_worker_once(store,good,now=due)
 assert resumed["state"]=="succeeded"
 assert good.calls[0][1]["previous_validation_error"].startswith("FIELD_IDENTITY_COLLISION:")
 assert {x["value"] for x in current_items(store,workspace_id="owner:demo")}=={"晨雾","竹影"}

def test_unrelated_correction_elsewhere_in_part_cannot_bypass_new_field_collision(store):
 from knowledge_weaver.extraction import ExtractionError
 from knowledge_weaver.extraction.proposals import subject_id_for
 record(store,subject_id=subject_id_for("project:songta","松塔服务器"),display_text="项目标记为晨雾",value="晨雾")
 e=make_event(native_id="message:separate-correction",text="更正另一个网页服务器端口。自动采集标记为竹影。",event_time={"value":"2026-10-31T10:00:00Z","precision":"instant","timezone":"UTC"})
 with pytest.raises(ExtractionError,match="FIELD_IDENTITY_COLLISION"):
  propose(store,e,subject_hint="松塔服务器",value="竹影",quote="自动采集标记为竹影")
