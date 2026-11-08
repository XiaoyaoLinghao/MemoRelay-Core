import pytest
from knowledge_weaver.jobs import enqueue_extraction_repair, JobError
from knowledge_weaver.extraction import run_worker_once
from knowledge_weaver.store.status import _job_pipeline
from knowledge_weaver.store import Store
from knowledge_weaver.portability import export_snapshot, import_snapshot, SnapshotError
from .test_extract_upgrade import failed, old_history, Fail, NoFacts, store


def test_three_upgrades_recover_only_exact_failed_chain_with_restorable_history(store,tmp_path):
    root,_=failed(store);chain=[root]
    for version in ('prompt:7','prompt:8'):
        child=enqueue_extraction_repair(store,chain[-1],prompt_version=version)
        assert run_worker_once(store,Fail())['state']=='failed'
        chain.append(child['job_id'])
    before={jid:old_history(store,jid) for jid in chain}
    last=enqueue_extraction_repair(store,chain[-1],prompt_version='prompt:9')
    export_snapshot(store,tmp_path/'pending');import_snapshot(tmp_path/'pending',tmp_path/'pending-restored')
    restored=Store(tmp_path/'pending-restored/knowledge.db')
    assert _job_pipeline(restored,'extract')['failed']==3
    assert run_worker_once(restored,NoFacts())['job_id']==last['job_id']
    assert _job_pipeline(restored,'extract')['failed']==0
    for jid in chain:assert old_history(restored,jid)==before[jid]
    export_snapshot(restored,tmp_path/'done');import_snapshot(tmp_path/'done',tmp_path/'done-restored')
    final=Store(tmp_path/'done-restored/knowledge.db')
    assert _job_pipeline(final,'extract')['failed']==0
    with final.transaction() as c:c.execute('UPDATE jobs SET repair_of_job_id=NULL WHERE job_id=?',(chain[1],))
    assert _job_pipeline(final,'extract')['failed']==1
    export_snapshot(final,tmp_path/'broken')
    with pytest.raises(SnapshotError):import_snapshot(tmp_path/'broken',tmp_path/'rejected')


def test_v9_replay_and_downgrade_still_rejected(store):
    old,_=failed(store,prompt='prompt:9')
    for version in ('prompt:7','prompt:8','prompt:9'):
        with pytest.raises(JobError):enqueue_extraction_repair(store,old,prompt_version=version)


@pytest.mark.parametrize('field,value',[('model_version','model:other'),('policy_version','policy:other'),('input_key','other@1')])
def test_v9_mismatched_chain_cannot_commit_or_hide_failure(store,field,value):
    old,_=failed(store);child=enqueue_extraction_repair(store,old,prompt_version='prompt:9')
    before=old_history(store,old)
    with store.transaction() as c:c.execute('UPDATE jobs SET '+field+'=? WHERE job_id=?',(value,child['job_id']))
    result=run_worker_once(store,NoFacts())
    assert result['state']!='succeeded'
    assert _job_pipeline(store,'extract')['failed']>=1
    assert old_history(store,old)==before
    with store.read() as c:
        assert c.execute('SELECT recovered_by_job_id FROM jobs WHERE job_id=?',(old,)).fetchone()[0] is None
