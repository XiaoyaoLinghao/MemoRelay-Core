"""Real extraction/job commit -> source retention; no model calls or host data."""
import pytest
from knowledge_weaver.extraction import run_worker_once, load_event_group, sanitized_parts, ExtractionError
from knowledge_weaver.lifecycle import register_lifecycle, sweep
from knowledge_weaver.portability import export_snapshot, import_snapshot
from knowledge_weaver.store import Store, receive_batch, current_items, fetch_source
from .test_extraction import MockExtractor, _claim, _proposal
from ..store.conftest import make_event, make_batch

ANCHOR={"value":"2020-01-01T00:00:00Z","precision":"instant","timezone":"UTC"}


def setup(store, *, event_time=ANCHOR):
    event=make_event(native_id="preprod-bootstrap",text="端口改为4200",event_time=event_time)
    receive_batch(store,make_batch([event]))
    subject=f"{event['event_id']}#1@0"
    register_lifecycle(store,subject_kind="event_part",subject_id=subject,class_="temporary_detail",business_anchor=None,state="bootstrap_pending",reason="imported-history-bootstrap")
    part=sanitized_parts(load_event_group(store,event["event_id"],1))[0]
    return event,subject,part


def lifecycle(store,subject):
    with store.read() as conn:
        return dict(conn.execute("SELECT * FROM lifecycle_records WHERE subject_kind='event_part' AND subject_id=?",(subject,)).fetchone())


def test_failed_then_valid_extraction_unlocks_old_detail_but_keeps_knowledge(store):
    event,subject,part=setup(store)
    assert sweep(store,now="2026-11-01T00:00:00Z")["event_parts"]==0
    class Offline:
        def extract(self,*args,**kwargs):raise ExtractionError("MODEL_UNAVAILABLE","synthetic outage",retryable=True)
    assert run_worker_once(store,Offline())["state"]=="retry_wait"
    assert lifecycle(store,subject)["state"]=="bootstrap_pending"
    with store.read() as conn:retry=conn.execute("SELECT run_after FROM jobs WHERE kind='extract'").fetchone()[0]
    model=MockExtractor(_proposal(_claim(part,assertion_kind="explicit_statement")))
    assert run_worker_once(store,model,now=retry)["state"]=="succeeded"
    row=lifecycle(store,subject)
    assert row["state"]=="active" and row["business_anchor"]==ANCHOR["value"]
    assert row["expires_at"]=="2020-01-31T00:00:00Z"
    assert row["reason"].startswith("extraction-complete:job:")
    assert sweep(store,now="2026-11-01T00:00:00Z")["event_parts"]==1
    assert fetch_source(store,event["event_id"],1,0)["text"] is None
    items=current_items(store,workspace_id="owner:demo")
    assert any(i["value"]==4200 and i["source_detail_expired"] for i in items)


@pytest.mark.parametrize("mode",["invalid_quote","unaccounted","model_warning"])
def test_success_without_complete_valid_disposition_does_not_unlock(store,mode):
    event,subject,part=setup(store)
    proposal=_proposal(_claim(part,assertion_kind="explicit_statement"))
    if mode=="invalid_quote":proposal["claims"][0]["evidence"][0]["support_quote"]="not in source"
    elif mode=="unaccounted":proposal["claims"]=[]
    else:proposal["warnings"]=["source remains ambiguous"]
    result=run_worker_once(store,MockExtractor(proposal))
    assert result["state"]==("retry_wait" if mode=="model_warning" else "succeeded")
    if mode=="model_warning":assert current_items(store,workspace_id="owner:demo")==[]
    assert lifecycle(store,subject)["state"]=="bootstrap_pending"
    assert sweep(store,now="2030-01-01T00:00:00Z")["event_parts"]==0
    assert fetch_source(store,event["event_id"],1,0)["text"]==part.text


def test_explicit_no_durable_knowledge_disposition_finishes_source(store):
    event,subject,part=setup(store)
    proposal=_proposal();proposal["excluded_event_ids"]=[event["event_id"]]
    assert run_worker_once(store,MockExtractor(proposal))["state"]=="succeeded"
    assert lifecycle(store,subject)["state"]=="active"
    assert sweep(store,now="2026-11-01T00:00:00Z")["event_parts"]==1
    assert current_items(store,workspace_id="owner:demo")==[]


def test_unknown_business_age_is_not_replaced_by_processing_time(store):
    event,subject,part=setup(store,event_time={"value":None,"precision":"unknown","timezone":"UTC"})
    run_worker_once(store,MockExtractor(_proposal(_claim(part,assertion_kind="explicit_statement"))))
    row=lifecycle(store,subject)
    assert row["state"]=="active" and row["class"]=="unclassified"
    assert row["business_anchor"] is None and row["expires_at"] is None
    assert sweep(store,now="2035-01-01T00:00:00Z")["event_parts"]==0


def test_same_completed_job_is_not_recomputed_or_renewed(store):
    event,subject,part=setup(store);model=MockExtractor(_proposal(_claim(part,assertion_kind="explicit_statement")))
    run_worker_once(store,model);before=lifecycle(store,subject)
    assert run_worker_once(store,model)["claimed"]==0
    assert len(model.calls)==1 and lifecycle(store,subject)==before


def test_stale_source_fence_keeps_protection_and_rolls_back_claims(store):
    event,subject,part=setup(store)
    class Changed:
        def extract(self,*args,**kwargs):
            with store.transaction() as conn:conn.execute("UPDATE event_revisions SET group_state='incomplete' WHERE event_id=?",(event["event_id"],))
            return _proposal(_claim(part,assertion_kind="explicit_statement"))
    assert run_worker_once(store,Changed())["state"]=="stale"
    assert lifecycle(store,subject)["state"]=="bootstrap_pending"
    assert current_items(store,workspace_id="owner:demo")==[]


def test_source_classification_failure_rolls_back_both_claim_and_unlock(store,monkeypatch):
    event,subject,part=setup(store)
    import knowledge_weaver.lifecycle.bootstrap as module
    real=module.finalize_sources_in_transaction
    def broken(conn,**kwargs):
        real(conn,**kwargs)
        raise OSError("synthetic transaction write failure")
    monkeypatch.setattr(module,"finalize_sources_in_transaction",broken)
    with pytest.raises(OSError):run_worker_once(store,MockExtractor(_proposal(_claim(part,assertion_kind="explicit_statement"))))
    assert lifecycle(store,subject)["state"]=="bootstrap_pending"
    assert current_items(store,workspace_id="owner:demo")==[]


def test_imported_pending_history_finishes_using_original_business_age(store,tmp_path):
    event,subject,part=setup(store)
    export_snapshot(store,tmp_path/"snapshot")
    restored=Store(import_snapshot(tmp_path/"snapshot",tmp_path/"restored",bootstrap_import=True)["target_db"])
    assert sweep(restored,now="2026-11-01T00:00:00Z")["event_parts"]==0
    part=sanitized_parts(load_event_group(restored,event["event_id"],1))[0]
    run_worker_once(restored,MockExtractor(_proposal(_claim(part,assertion_kind="explicit_statement"))))
    assert lifecycle(restored,subject)["expires_at"]=="2020-01-31T00:00:00Z"
    assert sweep(restored,now="2026-11-01T00:00:00Z")["event_parts"]==1
    assert any(i["value"]==4200 for i in current_items(restored,workspace_id="owner:demo"))


def test_only_accounted_part_of_multi_part_history_is_unlocked(store):
    first=make_event(native_id="preprod-parts",text="端口4200",event_time=ANCHOR)
    second=make_event(native_id="preprod-parts",text="待确认的复杂历史",event_time=ANCHOR)
    first["body"]["part_count"]=second["body"]["part_count"]=2
    second["body"]["part_index"]=1
    receive_batch(store,make_batch([first,second]))
    subjects=[f"{first['event_id']}#1@{i}" for i in (0,1)]
    for subject in subjects:register_lifecycle(store,subject_kind="event_part",subject_id=subject,class_="temporary_detail",business_anchor=None,state="bootstrap_pending",reason="imported-history-bootstrap")
    part=sanitized_parts(load_event_group(store,first["event_id"],1))[0]
    run_worker_once(store,MockExtractor(_proposal(_claim(part,assertion_kind="explicit_statement"))))
    assert lifecycle(store,subjects[0])["state"]=="active"
    assert lifecycle(store,subjects[1])["state"]=="bootstrap_pending"
    assert sweep(store,now="2026-11-01T00:00:00Z")["event_parts"]==1
    assert fetch_source(store,first["event_id"],1,1)["text"]=="待确认的复杂历史"


def test_protected_unresolved_history_is_visible_and_scope_bound(store):
    from knowledge_weaver.retrieval import execute_recall
    from ..retrieval.conftest import recall_request
    event,subject,part=setup(store)
    run_worker_once(store,MockExtractor(_proposal()))
    result=execute_recall(store,recall_request(query="端口"),binding={"workspace_id":"owner:demo","project_ids":["project:songta"]})
    assert "processing_pending" in result["incomplete_reasons"]
    assert result["status"]=="degraded"
    other=execute_recall(store,recall_request(query="端口",project_ids=["project:other"]),binding={"workspace_id":"owner:demo","project_ids":["project:other"]})
    assert "processing_pending" not in other["incomplete_reasons"] and other["items"]==[]


def test_independent_parts_keep_distinct_30_and_180_day_classes(store):
    first=make_event(native_id="preprod-part-classes",text="诊断记录：端口4200",event_time=ANCHOR)
    second=make_event(native_id="preprod-part-classes",text="恢复演练仅自述已完成，尚未独立核验",event_time=ANCHOR)
    first["body"]["part_count"]=second["body"]["part_count"]=2
    second["body"]["part_index"]=1
    receive_batch(store,make_batch([first,second]))
    register_lifecycle(store,subject_kind="event_part",subject_id=f"{first['event_id']}#1@0",class_="temporary_detail",business_anchor=ANCHOR["value"],state="bootstrap_pending",reason="explicit temporary diagnostic detail")
    parts=sanitized_parts(load_event_group(store,first["event_id"],1))
    proposals=[_claim(parts[0],assertion_kind="explicit_statement"),_claim(parts[1],subject_hint="恢复演练",predicate="task.status",facet="reported",value={"status":"reported_done","conditions":[],"satisfied":[]},display_text=parts[1].text,assertion_kind="completion_report")]
    result=run_worker_once(store,MockExtractor(_proposal(*proposals)))
    assert result["state"]=="succeeded" and result["claims_written"]==2
    rows=[lifecycle(store,f"{first['event_id']}#1@{i}") for i in (0,1)]
    assert [r["class"] for r in rows]==["temporary_detail","completed_task"]
    assert [r["expires_at"] for r in rows]==["2020-01-31T00:00:00Z","2020-06-29T00:00:00Z"]
