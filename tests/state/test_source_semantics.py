"""A non-assertive source can be scope-corrected without deleting its history."""
import hashlib
import json
import pytest

from knowledge_weaver.store import receive_batch, current_items, fetch_source
from knowledge_weaver.store.claims import ClaimError, record_claim_revision, rebuild_projection
from knowledge_weaver.store.source_semantics import record_source_exclusion
from knowledge_weaver.extraction import run_worker_once
from knowledge_weaver.portability import export_snapshot, import_snapshot
from .conftest import evidence_ref
from ..store.conftest import make_event, make_batch


def _source(store, name='recall', role='tool', origin='tool_execution', value=4100):
    event = make_event(native_id='scope:' + name, text=f'该次工具记录给出的服务端口为{value}。', role=role, origin_kind=origin)
    receive_batch(store, make_batch([event], batch_id='batch:' + name))
    return event


def _record(store, event, value=4100):
    return record_claim_revision(store, workspace_id='owner:demo', project_id='project:songta',
                                 subject_id='subject:service', predicate='server.port', facet='observed',
                                 qualifiers={}, value=value, display_text='历史工具检索返回的端口',
                                 authority='tool_verified', assertion_kind='observation',
                                 event_time=event['event_time'], evidence=[evidence_ref(event)])


def _exclude(store, event, **overrides):
    values = dict(event_id=event['event_id'], event_revision=1, part_index=0,
                  body_sha256=event['body']['sha256'], reason='historical_recall',
                  proof_sha256=hashlib.sha256(b'operator-verified-native-recall').hexdigest(),
                  reviewer='operator:test')
    values.update(overrides)
    return record_source_exclusion(store, **values)


def test_nonassertive_fact_leaves_current_view_but_source_and_revision_stay(store):
    event = _source(store)
    revision = _record(store, event)
    before = fetch_source(store, event['event_id'], 1, 0)
    assert current_items(store, workspace_id='owner:demo')
    result = _exclude(store, event)
    assert revision['claim_id'] in result['affected_claim_ids']
    assert current_items(store, workspace_id='owner:demo') == []
    assert fetch_source(store, event['event_id'], 1, 0) == before
    with store.read() as conn:
        assert conn.execute('SELECT COUNT(*) FROM claim_revisions').fetchone()[0] == 1
        assert conn.execute('SELECT status FROM claim_revisions').fetchone()[0] == 'revoked'
    rebuild_projection(store)
    assert current_items(store, workspace_id='owner:demo') == []


def test_independent_supported_revision_remains_available(store):
    real = _source(store, 'real-measurement', value=4200)
    _record(store, real, 4200)
    recalled = _source(store, 'recall-later')
    _record(store, recalled, 4100)
    _exclude(store, recalled)
    assert [x['value'] for x in current_items(store, workspace_id='owner:demo')] == [4200]
    assert fetch_source(store, recalled['event_id'], 1, 0)['text']


@pytest.mark.parametrize('role,origin', [('user', 'user_input'), ('assistant', 'assistant_reply')])
def test_user_or_assistant_statement_cannot_be_reclassified_as_tool_utility(store, role, origin):
    event = _source(store, role, role, origin)
    with pytest.raises(ClaimError):
        _exclude(store, event)
    with store.read() as conn:
        assert conn.execute('SELECT COUNT(*) FROM source_semantic_exclusions').fetchone()[0] == 0


def test_fingerprint_fence_and_repeated_operator_review_are_idempotent(store):
    event = _source(store)
    _record(store, event)
    with pytest.raises(ClaimError) as error:
        _exclude(store, event, body_sha256='0' * 64)
    assert error.value.code == 'STALE_CURSOR'
    assert current_items(store, workspace_id='owner:demo')
    _exclude(store, event)
    reviewed_watermark = store.watermark()
    _exclude(store, event)
    assert store.watermark() == reviewed_watermark
    with store.read() as conn:
        assert conn.execute('SELECT COUNT(*) FROM source_semantic_exclusions').fetchone()[0] == 1
    with pytest.raises(ClaimError):
        _exclude(store, event, reason='clock_reading')


def test_reviewed_legacy_source_is_processed_without_model_call_or_new_facts(store):
    event = _source(store)
    _exclude(store, event)

    class NoCall:
        def extract(self, parts, context):
            raise AssertionError('A reviewed historical utility must not be re-extracted')

    result = run_worker_once(store, NoCall())
    assert result['state'] == 'succeeded' and result['claims_written'] == 0
    assert current_items(store, workspace_id='owner:demo') == []
    assert fetch_source(store, event['event_id'], 1, 0)['text'] == event['body']['text']
    with store.read() as conn:
        detail = json.loads(conn.execute('SELECT detail FROM job_attempts').fetchone()[0])
        assert detail['source_scope_reviewed'] == ['historical_recall']


def test_export_import_keeps_review_decision_and_cannot_resurrect_recalled_fact(store, tmp_path):
    event = _source(store)
    _record(store, event)
    _exclude(store, event)
    export_snapshot(store, tmp_path / 'snapshot')
    receipt = import_snapshot(tmp_path / 'snapshot', tmp_path / 'restored')
    from knowledge_weaver.store import Store
    restored = Store(tmp_path / 'restored/knowledge.db')
    assert current_items(restored, workspace_id='owner:demo') == []
    assert fetch_source(restored, event['event_id'], 1, 0)['text'] == event['body']['text']
    with restored.read() as conn:
        assert conn.execute('SELECT COUNT(*) FROM source_semantic_exclusions').fetchone()[0] == 1


def test_old_keyword_and_vector_entries_cannot_recall_quarantined_current_fact(store):
    from knowledge_weaver.embeddings.index import HashEmbeddingProvider, configure_model
    from knowledge_weaver.retrieval.embeddings_index import collect_vectors
    from knowledge_weaver.retrieval import execute_recall
    from ..retrieval.conftest import recall_request

    event = _source(store)
    revision = _record(store, event)
    provider = HashEmbeddingProvider(dimension=8)
    configure_model(provider.model_id, model_revision=provider.model_revision, dimension=8)
    try:
        assert revision['revision_id'] in collect_vectors(store, {revision['revision_id']}, provider)
        _exclude(store, event)
        binding = {'client_id': 'client:test', 'workspace_id': 'owner:demo',
                   'project_ids': ['project:songta'], 'actor_ids': ['*']}
        result = execute_recall(store, recall_request(query='服务端口'), binding=binding, embedder=provider)
        assert result['items'] == []
        assert fetch_source(store, event['event_id'], 1, 0)['text']
    finally:
        configure_model(None)
