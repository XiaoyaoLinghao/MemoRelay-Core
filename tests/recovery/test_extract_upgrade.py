"""Explicit same-source processor upgrades preserve failures and clear real gaps."""
import pytest
from knowledge_weaver.store import receive_batch, Store, status
from knowledge_weaver.store.status import _job_pipeline
from knowledge_weaver.extraction import run_worker_once, ExtractionError
from knowledge_weaver.jobs import enqueue_extraction_repair, JobError
from knowledge_weaver.portability import export_snapshot, import_snapshot, SnapshotError
from ..store.conftest import make_event, make_batch


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / 'kw/knowledge.db')


class Fail:
    def extract(self, parts, context):
        raise ExtractionError('MODEL_UNAVAILABLE', 'synthetic unavailable', retryable=False, category='transport')


class NoFacts:
    def extract(self, parts, context):
        return {'schema_version': '1.0', 'claims': [], 'excluded_event_ids': [p.event_id for p in parts], 'warnings': []}


def failed(store, *, name='old', prompt='prompt:6'):
    event = make_event(native_id='upgrade:' + name, text='合成临时过程，无新增业务事实。')
    receive_batch(store, make_batch([event], batch_id='batch:' + name))
    with store.transaction() as conn:
        conn.execute('UPDATE jobs SET prompt_version=? WHERE input_key=?', (prompt, event['event_id'] + '@1'))
    outcome = run_worker_once(store, Fail())
    assert outcome['state'] == 'failed'
    return outcome['job_id'], event


def old_history(store, job_id):
    with store.read() as conn:
        job = dict(conn.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone())
        for key in ('recovered_by_job_id', 'recovered_at', 'completed_seq'):
            job.pop(key)
        attempts = [dict(x) for x in conn.execute('SELECT * FROM job_attempts WHERE job_id=?', (job_id,))]
    return job, attempts


def test_upgrade_links_only_after_its_successful_commit_and_keeps_failed_history(store):
    old, _ = failed(store)
    before = old_history(store, old)
    new = enqueue_extraction_repair(store, old)
    assert new['input_version'] == 2 and new['repair_of_job_id'] == old
    assert _job_pipeline(store, 'extract')['failed'] == 1
    with store.read() as conn:
        assert conn.execute('SELECT recovered_by_job_id FROM jobs WHERE job_id=?', (old,)).fetchone()[0] is None
    result = run_worker_once(store, NoFacts())
    assert result['state'] == 'succeeded'
    assert _job_pipeline(store, 'extract')['failed'] == 0
    assert old_history(store, old) == before
    with store.read() as conn:
        parent = conn.execute('SELECT * FROM jobs WHERE job_id=?', (old,)).fetchone()
        target = conn.execute('SELECT * FROM jobs WHERE job_id=?', (new['job_id'],)).fetchone()
        assert parent['state'] == 'failed' and parent['last_error_code'] == 'MODEL_UNAVAILABLE'
        assert parent['recovered_by_job_id'] == target['job_id']
        assert target['completed_seq'] > parent['completed_seq']
    assert enqueue_extraction_repair(store, old)['job_id'] == new['job_id']
    assert run_worker_once(store, NoFacts())['claimed'] == 0


def test_failed_upgrade_cannot_hide_the_old_failure_or_spawn_more_retries(store):
    old, _ = failed(store)
    new = enqueue_extraction_repair(store, old)
    assert run_worker_once(store, Fail())['state'] == 'failed'
    assert _job_pipeline(store, 'extract')['failed'] == 2
    assert enqueue_extraction_repair(store, old)['job_id'] == new['job_id']
    with store.read() as conn:
        assert conn.execute('SELECT recovered_by_job_id FROM jobs WHERE job_id=?', (old,)).fetchone()[0] is None


def test_same_processor_failure_is_not_eligible_for_an_upgrade(store):
    old, _ = failed(store, prompt='prompt:7')
    with pytest.raises(JobError):
        enqueue_extraction_repair(store, old)
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE kind='extract'").fetchone()[0] == 1


def test_another_sources_success_still_does_not_clear_a_failed_unit(store, tmp_path):
    old, _ = failed(store)
    other = make_event(native_id='upgrade:other', text='另一个来源，无新增业务事实。')
    receive_batch(store, make_batch([other], batch_id='batch:other'))
    success = run_worker_once(store, NoFacts())
    assert success['state'] == 'succeeded'
    assert _job_pipeline(store, 'extract')['failed'] == 1
    with store.transaction() as conn:
        conn.execute('UPDATE jobs SET recovered_by_job_id=?,recovered_at=?,completed_seq=0 WHERE job_id=?',
                     (success['job_id'], '2026-11-06T00:00:00Z', old))
        conn.execute('UPDATE jobs SET completed_seq=1 WHERE job_id=?', (success['job_id'],))
    assert _job_pipeline(store, 'extract')['failed'] == 1
    export_snapshot(store, tmp_path / 'invalid-link')
    with pytest.raises(SnapshotError):
        import_snapshot(tmp_path / 'invalid-link', tmp_path / 'bad-restore')


def test_pending_upgrade_survives_export_restore_then_records_exact_recovery(store, tmp_path):
    old, _ = failed(store)
    new = enqueue_extraction_repair(store, old)
    export_snapshot(store, tmp_path / 'pending')
    import_snapshot(tmp_path / 'pending', tmp_path / 'restored')
    restored = Store(tmp_path / 'restored/knowledge.db')
    assert _job_pipeline(restored, 'extract')['failed'] == 1
    outcome = run_worker_once(restored, NoFacts())
    assert outcome['job_id'] == new['job_id'] and outcome['state'] == 'succeeded'
    assert _job_pipeline(restored, 'extract')['failed'] == 0
    export_snapshot(restored, tmp_path / 'done')
    import_snapshot(tmp_path / 'done', tmp_path / 'done-restored')
    final = Store(tmp_path / 'done-restored/knowledge.db')
    assert _job_pipeline(final, 'extract')['failed'] == 0
