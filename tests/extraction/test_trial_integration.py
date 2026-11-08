"""Trial improvements through actual product entry points, synthetic data only."""
import io
import json
import time
from unittest import mock

import pytest
from knowledge_weaver.extraction import HttpExtractor, ExtractionError, run_worker_once, load_event_group, sanitized_parts
from knowledge_weaver.extraction.http_client import EXTRACTION_SYSTEM_PROMPT_V6
from knowledge_weaver.jobs import recover
from knowledge_weaver.store import receive_batch, current_items
from ..store.conftest import make_event, make_batch
from .test_extraction import MockExtractor, _proposal, _claim


def test_truncated_response_is_rejected_even_if_json_is_parseable():
    body={"choices":[{"finish_reason":"length","message":{"content":json.dumps(_proposal())}}],"usage":{"total_tokens":5}}
    with mock.patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(body).encode())):
        with pytest.raises(ExtractionError) as exc:
            HttpExtractor('http://synthetic.invalid',model='fixture').extract([],{})
    assert exc.value.category=='output_length' and exc.value.usage_present


def test_new_jobs_record_v10_and_legacy_instructions_remain_available(store):
    event=make_event(native_id='v6:new');receive_batch(store,make_batch([event]))
    with store.read() as conn:
        row=conn.execute("SELECT prompt_version FROM jobs WHERE kind='extract'").fetchone()
    assert row['prompt_version']=='prompt:10'
    calls=[]
    def response(req,timeout=None):
        calls.append(json.loads(req.data));return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(_proposal())}}]}).encode())
    with mock.patch('urllib.request.urlopen',side_effect=response):
        HttpExtractor('http://synthetic.invalid',model='fixture').extract([],{'prompt_version':'prompt:6'})
    assert calls[0]['messages'][0]['content'].startswith(EXTRACTION_SYSTEM_PROMPT_V6)


def test_v6_anchor_values_help_identification_without_becoming_evidence(store):
    old=make_event(native_id='v6:old',text='端口改为4200');receive_batch(store,make_batch([old]))
    part=sanitized_parts(load_event_group(store,old['event_id'],1))[0]
    assert run_worker_once(store,MockExtractor(_proposal(_claim(part))))['state']=='succeeded'
    new=make_event(native_id='v6:new',text='端口更正为4300');receive_batch(store,make_batch([new],batch_id='new'))
    class Capture:
        context=None
        def extract(self,parts,context):
            self.context=context
            return {'schema_version':'1.0','claims':[],'excluded_event_ids':[new['event_id']],'warnings':[]}
    model=Capture();assert run_worker_once(store,model)['state']=='succeeded'
    assert model.context['known_claim_details'][0]['value']==4200
    assert model.context['known_claim_details'][0]['authority']=='user_confirmed'
    # Seeing an anchor never writes a new fact or changes its clock/value.
    assert [i['value'] for i in current_items(store,workspace_id='owner:demo')]==[4200]


def test_lease_is_kept_alive_while_model_runs_past_original_deadline(store):
    event=make_event(native_id='v6:lease');receive_batch(store,make_batch([event]))
    class Slow:
        def extract(self,parts,context):
            time.sleep(2.1)
            outcome=recover(store)
            assert outcome['interrupted_attempts']==0
            return {'schema_version':'1.0','claims':[],'excluded_event_ids':[event['event_id']],'warnings':[]}
    result=run_worker_once(store,Slow(),lease_seconds=1)
    assert result['state']=='succeeded'


def test_long_source_over100_facts_is_atomic_and_keeps_original_spans(store):
    import re
    text="\n".join(f"条目{i}的值为{i}。" + "无新增信息。"*12 for i in range(150))
    event=make_event(native_id="v6:150facts",text=text)
    receive_batch(store,make_batch([event]))
    class SegmentModel:
        calls=0
        def extract(self,parts,context):
            self.calls+=1
            assert "segment" in context
            facts=[]
            for part in parts:
                for match in re.finditer(r"条目(\d+)的值为(\d+)",part.text):
                    index=int(match.group(1))
                    facts.append(_claim(part,quote=match.group(0),subject_hint=f"条目{index}",predicate="fixture.value",value=index,display_text=match.group(0),assertion_kind="explicit_statement"))
            return {"schema_version":"1.0","claims":facts,"excluded_event_ids":[] if facts else [event['event_id']],"warnings":[]}
    model=SegmentModel();result=run_worker_once(store,model)
    assert result["state"]=="succeeded" and result["claims_written"]==150
    assert model.calls>1
    with store.read() as conn:
        assert conn.execute("SELECT count(*) FROM claims").fetchone()[0]==150
        assert conn.execute("SELECT count(*) FROM extraction_segment_cache").fetchone()[0]==0
        coverage=json.loads(conn.execute("SELECT coverage_json FROM extraction_segment_receipts").fetchone()[0])
        assert coverage['complete'] and coverage['covered_bytes']==len(text.encode())
        links=conn.execute("SELECT start_byte,end_byte,body_sha256 FROM evidence_links").fetchall()
    data=text.encode('utf-8')
    assert all("条目" in data[r['start_byte']:r['end_byte']].decode() and r['body_sha256']==event['body']['sha256'] for r in links)


def test_segment_failure_never_reports_complete_coverage(store):
    from .test_segments import _part,_context,_StubExtractor,_proposal as piece
    from knowledge_weaver.extraction.segments import CheckpointingExtractor
    part=_part("A"*2000)
    def interrupted(parts,context,call):
        if call==2:raise ExtractionError('MODEL_UNAVAILABLE','synthetic interruption',retryable=True)
        return piece(excluded=[part.event_id])
    wrapper=CheckpointingExtractor(store,_StubExtractor(interrupted),max_bytes=800,overlap_bytes=100)
    with pytest.raises(ExtractionError):wrapper.extract([part],_context())
    assert not wrapper.coverage['complete']
    assert wrapper.coverage['covered_bytes']<wrapper.coverage['total_bytes']


def test_short_output_length_retry_does_not_recall_parent_or_successful_leaf(store):
    from .test_segments import _part,_context,_proposal as piece
    from knowledge_weaver.extraction.segments import CheckpointingExtractor
    part=_part('A'*1500);calls=[];failed=False
    class Model:
        def extract(self,parts,context):
            nonlocal failed
            ranges=context.get('segment_ranges')
            calls.append(None if ranges is None else ranges[0]['start_byte'])
            if ranges is None:raise ExtractionError('MODEL_UNAVAILABLE','length',retryable=True,category='output_length')
            if ranges[0]['start_byte']>0 and not failed:
                failed=True;raise ExtractionError('MODEL_UNAVAILABLE','synthetic transport',retryable=True,category='transport')
            return piece(excluded=[part.event_id])
    wrapper=CheckpointingExtractor(store,Model())
    with pytest.raises(ExtractionError):wrapper.extract([part],_context())
    assert calls==[None,0,750]
    assert wrapper.extract([part],_context())['excluded_event_ids']==[part.event_id]
    assert calls==[None,0,750,750]


def test_long_pipeline_failure_commits_no_partial_facts_and_resumes(store):
    text='HEAD端口4317。'+('没有新信息。'*1200)+'TAIL保留47天。'
    event=make_event(native_id='v6:atomic',text=text);receive_batch(store,make_batch([event]))
    failed=False;head_calls=0
    class Model:
        def extract(self,parts,context):
            nonlocal failed,head_calls
            body=parts[0].text;claims=[]
            if 'HEAD端口4317' in body:
                head_calls+=1;claims.append(_claim(parts[0],quote='HEAD端口4317',predicate='fixture.head',value=4317,display_text='端口4317',assertion_kind='explicit_statement'))
            if 'TAIL保留47天' in body:
                if not failed:
                    failed=True;raise ExtractionError('MODEL_UNAVAILABLE','synthetic transport',retryable=True,category='transport')
                claims.append(_claim(parts[0],quote='TAIL保留47天',predicate='fixture.tail',value=47,display_text='保留47天',assertion_kind='explicit_statement'))
            return {'schema_version':'1.0','claims':claims,'excluded_event_ids':[] if claims else [event['event_id']],'warnings':[]}
    model=Model();first=run_worker_once(store,model)
    assert first['state']=='retry_wait' and current_items(store,workspace_id='owner:demo')==[]
    with store.read() as conn:
        due=conn.execute("SELECT run_after FROM jobs WHERE kind='extract'").fetchone()[0]
    second=run_worker_once(store,model,now=due)
    assert second['state']=='succeeded' and second['claims_written']==2
    assert head_calls==1
    assert {x['value'] for x in current_items(store,workspace_id='owner:demo')}=={4317,47}


def test_zero_write_segment_warning_does_not_block_another_grounded_segment(store):
    text='HEAD端口4317。'+('没有新信息。'*14000)+'TAIL保留47天。'
    event=make_event(native_id='v6:217k',text=text);receive_batch(store,make_batch([event]))
    calls=0
    class Model:
        def extract(self,parts,context):
            nonlocal calls
            calls+=1;body=parts[0].text;claims=[]
            for marker,predicate,value in [('HEAD端口4317','fixture.head',4317),('TAIL保留47天','fixture.tail',47)]:
                if marker in body:claims.append(_claim(parts[0],quote=marker,predicate=predicate,value=value,display_text=marker,assertion_kind='explicit_statement'))
            return {'schema_version':'1.0','claims':claims,'excluded_event_ids':[] if claims else [event['event_id']],'warnings':[] if claims else ['No new durable information in this window']}
    result=run_worker_once(store,Model())
    assert len(text.encode())>217148 and calls>25
    assert result['state']=='succeeded' and result['claims_written']==2
    assert {x['value'] for x in current_items(store,workspace_id='owner:demo')}=={4317,47}
    # Warnings do not authorize source cleanup: the whole part stays protected.
    with store.read() as conn:
        assert conn.execute("SELECT state FROM lifecycle_records WHERE subject_kind='event_part'").fetchone()[0]=='bootstrap_pending'


def test_enriched_anchor_context_has_a_fixed_budget_and_omits_huge_values():
    from knowledge_weaver.extraction.pipeline import _known_anchor_context
    entries=[dict(claim_id=f"claim:{i}",project_id="project:songta",predicate="value",facet="desired",label="safe",qualifiers={},value="ordinary business fact "*500,state="current",authority="user_confirmed",subject_id=f"subject:{i}",ordering_time="2026-11-05T00:00:00Z") for i in range(48)]
    anchors=_known_anchor_context({'claim_candidates':entries},enriched=True)
    assert anchors and len(json.dumps(anchors,separators=(',',':')).encode())<=8192
    assert all('value' not in x and x['value_omitted_large'] and x['value_digest'] for x in anchors)
