"""Do not turn an assistant's unchanged confirmed fact into a fresh fact version.

Only assistant replies matching the exact current field AND typed
value are reused. User confirmations, real tool observations, changed values,
qualifiers, entities and projects retain their independent source semantics.
The source event remains durable; a job-attempt audit links the repetition to
the original revision without adding a fresh evidentiary clock to that fact.
Remembered confidence is preserved; a new task completion report after mere
speculation remains new, unverified information.
"""
import json
from knowledge_weaver import contracts
from knowledge_weaver.store.claims import claim_id_for, _time_evidence
from knowledge_weaver.lifecycle.retention import _canonical_anchor, _parse


def split_assistant_echoes(conn, resolved):
    fresh=[];reused=[]
    for claim in resolved:
        if claim.authority not in ('reported','inferred') or not claim.evidence:
            fresh.append(claim);continue
        sources=[conn.execute('SELECT role,origin_kind FROM events WHERE event_id=?',(ref['event_id'],)).fetchone() for ref in claim.evidence]
        if any(row is None or row['role']!='assistant' or row['origin_kind']!='assistant_reply' for row in sources):
            fresh.append(claim);continue
        cid=claim_id_for(claim.workspace_id,claim.project_id,claim.subject_id,claim.predicate,claim.facet,claim.qualifiers)
        current=conn.execute('SELECT state,selected_revision_ids_json FROM current_claims WHERE claim_id=?',(cid,)).fetchone()
        selected=json.loads(current['selected_revision_ids_json']) if current and current['state']=='current' else []
        candidates=[]
        for rid in selected:
            row=conn.execute("SELECT revision_id,value_json,authority,assertion_kind,event_time_value,event_time_precision,created_at FROM claim_revisions WHERE revision_id=? AND claim_id=? AND status NOT IN ('expired','revoked')",(rid,cid)).fetchone()
            if row is not None and row['value_json']==contracts.compact_json(claim.value):
                # A repeat cannot upgrade a remembered hypothesis or legacy
                # record merely by being spoken again. An actual new task
                # completion report after a hypothesis is a different act.
                new_task_report=(claim.predicate=='task.status' and claim.assertion_kind=='completion_report' and row['authority']=='inferred')
                if not new_task_report:candidates.append(row)
        if not candidates and current and current['state']=='uncertain' and not selected:
            # Unselected uncertain evidence may be undated or lack readable source. Repeating it is
            # not a new dated confirmation and must not make uncertainty current.
            rows=conn.execute("SELECT revision_id,value_json,authority,event_time_value,event_time_precision,created_at FROM claim_revisions WHERE claim_id=? AND status='uncertain'",(cid,)).fetchall()
            if rows and all(row['value_json']==contracts.compact_json(claim.value) and not (claim.predicate=='task.status' and claim.assertion_kind=='completion_report' and row['authority']=='inferred') for row in rows):
                candidates=list(rows)
        if not candidates:
            fresh.append(claim);continue
        # Current selection is normally one revision. Reusing an exact selected
        # value never resolves a conflicting field or improves its grade.
        target=candidates[0];value=target['event_time_value'] if target['event_time_precision'] in ('instant','day') else None
        key,anomalous=_time_evidence(dict(target))
        anchor=_canonical_anchor(value) if key is not None and not anomalous and _parse(value) is not None else None
        reused.append({'claim_id':cid,'revision_id':target['revision_id'],'source_anchor':anchor,'part_indices':sorted({ref['part_index'] for ref in claim.evidence})})
    return fresh,reused
