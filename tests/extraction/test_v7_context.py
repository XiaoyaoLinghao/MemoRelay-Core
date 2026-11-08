"""Native source scope and bounded anchor context through real extraction jobs."""
import hashlib
import io
import json
from unittest import mock

from knowledge_weaver import contracts
from knowledge_weaver.extraction import HttpExtractor, load_event_group, run_worker_once, sanitized_parts
from knowledge_weaver.extraction.context import anchor_context, source_scope, repair_feedback
from knowledge_weaver.extraction.http_client import _SYSTEM_PROMPTS, EXTRACTION_SYSTEM_PROMPT_V7
from knowledge_weaver.store import current_items, receive_batch
from ..store.conftest import make_event, make_batch
from .test_extraction import MockExtractor, _claim, _proposal


def _anchor(index, **extra):
    return dict(claim_id='claim:' + str(index).zfill(40), project_id='project:songta',
                subject_id='subject:' + str(index), predicate='server.port', facet='observed',
                qualifiers={}, label='端口', value=index, state='current', authority='observed',
                ordering_time='2026-10-31T00:00:00Z', **extra)


def test_collision_anchor_is_first_within_shared_budget():
    entries = [_anchor(i) for i in range(48)]
    context = anchor_context({'claim_candidates': entries},
                             'FIELD_IDENTITY_COLLISION: null target_claim_hint collides with ' + entries[-1]['claim_id'])
    assert context['known_claims'][0]['claim_id'] == entries[-1]['claim_id']
    assert context['known_claim_details'][0]['claim_id'] == entries[-1]['claim_id']
    assert len(contracts.compact_json(context).encode()) <= 8192
    assert len(context['known_claims']) == len(context['known_claim_details'])


def test_repair_feedback_does_not_treat_the_previous_model_choice_as_source_truth():
    prefix = 'FIELD_IDENTITY_COLLISION: null target_claim_hint collides with claim:' + 'a' * 40
    old = prefix + '. Your null target declared a NEW field. Preserve that semantic choice.'
    current = repair_feedback(old)
    assert current.startswith(prefix)
    assert 'a previous null target is not authoritative' in current
    assert 'genuinely identical' in current and 'Do not bind an unrelated field' in current
    assert 'Preserve that semantic choice' not in current
    assert repair_feedback('MODEL_REVIEW_REQUIRED') == 'MODEL_REVIEW_REQUIRED'


def test_large_or_secret_anchor_does_not_hide_later_usable_fields():
    large = _anchor(0)
    large['qualifiers'] = {'note': 'x' * 10000}
    secret = _anchor(1)
    secret['value'] = 'sk-' + 'CONTEXT_CANARY_SECRET_9sK2' * 4
    good = _anchor(2)
    result = anchor_context({'claim_candidates': [large, secret, good]}, None)
    assert [x['claim_id'] for x in result['known_claims']] == [good['claim_id']]
    assert 'CONTEXT_CANARY_SECRET' not in json.dumps(result)


def test_large_value_is_digested_without_truncating_a_fact():
    item = _anchor(0)
    item['value'] = '中文' * 2000
    result = anchor_context({'claim_candidates': [item]}, None)
    value = result['known_claim_details'][0]
    assert value['value_omitted_large'] and 'value' not in value
    assert value['value_digest'] == contracts.sha256_hex(contracts.compact_json(item['value']))
    assert item['value'] == '中文' * 2000


def test_source_scope_only_exports_bounded_native_tool_identifiers():
    group = {'event_id': 'evt:' + 'a' * 64, 'origin_kind': 'tool_execution',
             'source_aliases': ['native-tool:terminal', 'native-tool:https://synthetic.invalid/key',
                                'content:123', 'native-tool:terminal']}
    scope = source_scope(group)
    assert scope['native_tool'] == 'terminal'
    assert scope['observation_id'] == group['event_id']
    assert 'synthetic.invalid' not in json.dumps(scope)
    group['source_aliases'].append('native-tool:session_search')
    assert source_scope(group)['native_tool'] is None


def test_v3_to_v6_instruction_bytes_remain_their_actual_versions():
    frozen = {
        'prompt:3': '49da6c47634913c8a02463b5f62175d5dd78bdc0922eb852588abc7a79034be7',
        'prompt:4': '556a05c71d934fcbf8418bdfd6150aadb96c050cdfaa9b2a4671477b318ae4e0',
        'prompt:5': '8f0a689dc817bb3a66c9e38831d0c085411fc63bd1bd7c197709897d17588f8a',
        'prompt:6': 'b61670d787ca6acc7b2a63eb440cd44ef54e72dd718c841ed365c8a66a347aae',
    }
    assert {key: hashlib.sha256(_SYSTEM_PROMPTS[key].encode()).hexdigest() for key in frozen} == frozen


def test_new_job_uses_v7_source_context_and_its_actual_http_instructions(store):
    event = make_event(native_id='v7:source-context', text='合成独立查询返回3条记录。',
                       role='tool', origin_kind='tool_execution')
    event['source']['aliases'] = ['native-tool:terminal']
    receive_batch(store, make_batch([event]))
    # This file verifies compatibility of the explicitly retained v7 processor.
    with store.transaction() as conn:
        conn.execute("UPDATE jobs SET prompt_version='prompt:7' WHERE input_key=?", (event["event_id"]+"@1",))
    requests = []

    def response(request, timeout=None):
        data = json.loads(request.data)
        requests.append(data)
        proposal = _proposal()
        proposal['excluded_event_ids'] = [event['event_id']]
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message':
                                                 {'content': json.dumps(proposal)}}]}).encode())

    with mock.patch('urllib.request.urlopen', side_effect=response):
        result = run_worker_once(store, HttpExtractor('http://synthetic.invalid', model='fixture'))
    assert result['state'] == 'succeeded'
    assert requests[0]['messages'][0]['content'].startswith(EXTRACTION_SYSTEM_PROMPT_V7)
    context = json.loads(requests[0]['messages'][1]['content'])['context']
    assert context['prompt_version'] == 'prompt:7'
    assert context['source_scope']['native_tool'] == 'terminal'
    assert context['source_scope']['observation_id'] == event['event_id']
    assert current_items(store, workspace_id='owner:demo') == []


def test_independent_invocations_can_keep_their_real_source_identity(store):
    for number in (1, 2):
        event = make_event(native_id=f'v7:invocation:{number}', text=f'本次独立查询返回{number}条记录。',
                           role='tool', origin_kind='tool_execution')
        receive_batch(store, make_batch([event], batch_id=f'batch:invocation:{number}'))
        part = sanitized_parts(load_event_group(store, event['event_id'], 1))[0]
        proposal = _proposal(_claim(part, predicate='query.result.count', facet='observed',
                                   value=number, display_text=part.text, assertion_kind='observation',
                                   qualifiers={'source_event_id': event['event_id']}))
        assert run_worker_once(store, MockExtractor(proposal))['state'] == 'succeeded'
    items = current_items(store, workspace_id='owner:demo')
    assert len(items) == 2 and {x['value'] for x in items} == {1, 2}


def test_v7_still_refuses_warned_new_facts_and_retains_source(store):
    event = make_event(native_id='v7:ambiguous', text='该服务的端口可能为4200。',
                       role='tool', origin_kind='tool_execution')
    receive_batch(store, make_batch([event]))
    # This file verifies compatibility of the explicitly retained v7 processor.
    with store.transaction() as conn:
        conn.execute("UPDATE jobs SET prompt_version='prompt:7' WHERE input_key=?", (event["event_id"]+"@1",))
    part = sanitized_parts(load_event_group(store, event['event_id'], 1))[0]
    proposal = _proposal(_claim(part, assertion_kind='inference'))
    proposal['warnings'] = ['Unresolved entity identity']
    outcome = run_worker_once(store, MockExtractor(proposal))
    assert outcome['state'] == 'retry_wait' and outcome['error'] == 'MODEL_REVIEW_REQUIRED'
    assert current_items(store, workspace_id='owner:demo') == []
    with store.read() as conn:
        assert conn.execute('SELECT body_text FROM event_parts WHERE event_id=?', (event['event_id'],)).fetchone()[0] == part.text


def test_semantically_rejected_page_is_not_replayed_as_a_successful_checkpoint(store):
    text = 'x' * 9000 + '端口为4200。'
    event = make_event(native_id='v7:rejected-cache', text=text)
    receive_batch(store, make_batch([event]))
    # This file verifies compatibility of the explicitly retained v7 processor.
    with store.transaction() as conn:
        conn.execute("UPDATE jobs SET prompt_version='prompt:7' WHERE input_key=?", (event["event_id"]+"@1",))

    class RepairingModel:
        calls = 0
        marker_calls = 0

        def extract(self, parts, context):
            self.calls += 1
            part = parts[0]
            if '端口为4200' not in part.text:
                result = _proposal()
                result['excluded_event_ids'] = [part.event_id]
                return result
            self.marker_calls += 1
            result = _proposal(_claim(part, quote='端口为4200', value=4200,
                                      assertion_kind='explicit_statement'))
            if self.marker_calls <= 2:
                result['warnings'] = ['Unresolved identity; do not commit this proposal']
            return result

    model = RepairingModel()
    due = None
    for attempt in range(3):
        outcome = run_worker_once(store, model, now=due)
        if attempt < 2:
            assert outcome['state'] == 'retry_wait'
            assert current_items(store, workspace_id='owner:demo') == []
            with store.read() as conn:
                due = conn.execute('SELECT run_after FROM jobs WHERE job_id=?', (outcome['job_id'],)).fetchone()[0]
        else:
            assert outcome['state'] == 'succeeded'
    assert model.marker_calls == 3
    assert model.calls == 5  # The valid sibling page is reused on the third attempt.
    assert [x['value'] for x in current_items(store, workspace_id='owner:demo')] == [4200]
