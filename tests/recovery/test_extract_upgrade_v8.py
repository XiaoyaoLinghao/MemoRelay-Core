import pytest
from knowledge_weaver.jobs import enqueue_extraction_repair, JobError
from knowledge_weaver.extraction import run_worker_once
from knowledge_weaver.store.status import _job_pipeline
from knowledge_weaver.store import Store
from knowledge_weaver.portability import export_snapshot, import_snapshot, SnapshotError
from .test_extract_upgrade import failed, old_history, Fail, NoFacts, store


def test_failed_v7_child_can_upgrade_once_and_clear_both_causally(store,tmp_path):
    old,_=failed(store);old_before=old_history(store,old)
    middle=enqueue_extraction_repair(store,old)
    assert run_worker_once(store,Fail())['state']=='failed'
    middle_before=old_history(store,middle['job_id'])
    new=enqueue_extraction_repair(store,middle['job_id'],prompt_version='prompt:8')
    assert _job_pipeline(store,'extract')['failed']==2
    assert run_worker_once(store,NoFacts())['state']=='succeeded'
    assert _job_pipeline(store,'extract')['failed']==0
    assert old_history(store,old)==old_before and old_history(store,middle['job_id'])==middle_before
    export_snapshot(store,tmp_path/'saved');import_snapshot(tmp_path/'saved',tmp_path/'restored')
    restored=Store(tmp_path/'restored/knowledge.db')
    assert _job_pipeline(restored,'extract')['failed']==0
    with restored.transaction() as c:
        c.execute('UPDATE jobs SET repair_of_job_id=NULL WHERE job_id=?',(middle['job_id'],))
    assert _job_pipeline(restored,'extract')['failed']==1
    export_snapshot(restored,tmp_path/'broken')
    with pytest.raises(SnapshotError):import_snapshot(tmp_path/'broken',tmp_path/'reject')


def test_v8_failed_is_not_eligible_for_same_processor_replay(store):
    old,_=failed(store,prompt='prompt:8')
    with pytest.raises(JobError):enqueue_extraction_repair(store,old,prompt_version='prompt:8')


def test_direct_v6_to_v8_preserves_failed_record(store):
    old,_=failed(store);before=old_history(store,old)
    new=enqueue_extraction_repair(store,old,prompt_version='prompt:8')
    assert run_worker_once(store,NoFacts())['job_id']==new['job_id']
    assert old_history(store,old)==before and _job_pipeline(store,'extract')['failed']==0
