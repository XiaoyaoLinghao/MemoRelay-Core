"""AC-47: reversible entity merge/aliases, no cross-project guessing, no history rewrite."""
from __future__ import annotations

import pytest

from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.store import (
    EntityError,
    current_items,
    list_aliases,
    merge_entities,
    undo_merge,
)
from knowledge_weaver.store.entities import resolve_canonical_subject

from .conftest import record

SUBJECT_A = "subject:device:living-room"
SUBJECT_B = "subject:device:alias-name"


def _seed_two_subjects(store, project_id="project:songta"):
    record(
        store,
        display_text="客厅设备端口为4200",
        value=4200,
        predicate="server.port",
        subject_id=SUBJECT_A,
        project_id=project_id,
    )
    record(
        store,
        display_text="客厅设备备份周期为每周",
        value="weekly",
        predicate="backup_interval",
        subject_id=SUBJECT_B,
        project_id=project_id,
    )


def _raw_subjects(store):
    with store.read() as conn:
        return {
            row["subject_id"]
            for row in conn.execute("SELECT DISTINCT subject_id FROM claims").fetchall()
        }


def test_merge_exposes_both_properties_and_undo_restores(store):
    _seed_two_subjects(store)
    alias = merge_entities(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        source_subject_id=SUBJECT_B,
        canonical_subject_id=SUBJECT_A,
    )
    assert alias["state"] == "active"
    items = current_items(store, workspace_id="owner:demo")
    assert {item["predicate"] for item in items} == {"server.port", "backup_interval"}
    assert {item["subject_id"] for item in items} == {SUBJECT_A}
    # Historical claims were not rewritten.
    assert _raw_subjects(store) == {SUBJECT_A, SUBJECT_B}
    with store.read() as conn:
        assert resolve_canonical_subject(
            conn, workspace_id="owner:demo", project_id="project:songta", subject_id=SUBJECT_B
        ) == SUBJECT_A
    removed = undo_merge(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        source_subject_id=SUBJECT_B,
        canonical_subject_id=SUBJECT_A,
    )
    assert removed["state"] == "removed"
    items = current_items(store, workspace_id="owner:demo")
    assert {item["subject_id"] for item in items} == {SUBJECT_A, SUBJECT_B}
    assert list_aliases(store, workspace_id="owner:demo") == []


def test_cross_project_merge_is_refused(store):
    _seed_two_subjects(store, project_id="project:songta")
    record(
        store,
        display_text="另一个项目的设备端口为80",
        value=80,
        predicate="server.port",
        subject_id="subject:device:other-project",
        project_id="project:other",
    )
    with pytest.raises(EntityError):
        merge_entities(
            store,
            workspace_id="owner:demo",
            project_id="project:songta",
            source_subject_id="subject:device:other-project",
            canonical_subject_id=SUBJECT_A,
        )


def test_merge_cycle_is_refused(store):
    _seed_two_subjects(store)
    merge_entities(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        source_subject_id=SUBJECT_B,
        canonical_subject_id=SUBJECT_A,
    )
    with pytest.raises(EntityError):
        merge_entities(
            store,
            workspace_id="owner:demo",
            project_id="project:songta",
            source_subject_id=SUBJECT_A,
            canonical_subject_id=SUBJECT_B,
        )
    with store.read() as conn:
        assert resolve_canonical_subject(
            conn, workspace_id="owner:demo", project_id="project:songta", subject_id=SUBJECT_B
        ) == SUBJECT_A


def test_merge_requires_existing_subject_in_same_project(store):
    _seed_two_subjects(store)
    with pytest.raises(EntityError):
        merge_entities(
            store,
            workspace_id="owner:demo",
            project_id="project:songta",
            source_subject_id=SUBJECT_B,
            canonical_subject_id="subject:does-not-exist",
        )
    with pytest.raises(EntityError):
        merge_entities(
            store,
            workspace_id="owner:demo",
            project_id="project:songta",
            source_subject_id=SUBJECT_A,
            canonical_subject_id=SUBJECT_A,
        )


def test_new_imported_claims_resolve_through_active_alias(store):
    _seed_two_subjects(store)
    merge_entities(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        source_subject_id=SUBJECT_B,
        canonical_subject_id=SUBJECT_A,
    )
    # A legacy importer may replay a claim under the old subject id.
    record(
        store,
        display_text="旧导入记录：设备位置为客厅",
        value="living-room",
        predicate="location",
        subject_id=SUBJECT_B,
        project_id="project:songta",
        authority="legacy_unknown",
        assertion_kind="explicit_statement",
    )
    items = current_items(store, workspace_id="owner:demo", include_states=("current", "uncertain"))
    location = [item for item in items if item["predicate"] == "location"][0]
    assert location["subject_id"] == SUBJECT_A


def test_recall_returns_canonical_subject_after_merge(store):
    _seed_two_subjects(store)
    merge_entities(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        source_subject_id=SUBJECT_B,
        canonical_subject_id=SUBJECT_A,
    )
    binding = {
        "client_id": "client:test",
        "workspace_id": "owner:demo",
        "project_ids": ["project:songta"],
        "actor_ids": ["*"],
    }
    request = {
        "schema_version": "1.0",
        "request_id": "req:merge:1",
        "session_id": "session:test",
        "turn_id": "turn:1",
        "query": "客厅设备",
        "project_ids": ["project:songta"],
        "intent": "current",
        "budget_tokens": 2000,
        "limit": 8,
        "deadline_ms": 7000,
        "cursor": None,
    }
    result = execute_recall(store, request, binding=binding)
    assert result["items"]
    assert {item["subject_id"] for item in result["items"]} == {SUBJECT_A}


@pytest.mark.parametrize('source_type,target_type',[('device','person'),('device','subject'),('device',None)])
def test_merge_rejects_known_incompatible_or_unresolved_types_without_writes(store,source_type,target_type):
 from knowledge_weaver.store.entities import _entity_id
 _seed_two_subjects(store)
 with store.transaction() as c:
  for subject,kind in [(SUBJECT_B,source_type),(SUBJECT_A,target_type)]:
   if kind:c.execute("INSERT INTO entities(entity_id,workspace_id,project_id,entity_type,canonical_name,created_at) VALUES(?,?,?,?,?,'2026-01-01T00:00:00Z')",(_entity_id('owner:demo','project:songta',subject),'owner:demo','project:songta',kind,subject))
  before=[tuple(x) for x in c.execute('SELECT * FROM claim_revisions ORDER BY revision_id')]
 with pytest.raises(EntityError,match='types are incompatible'):
  merge_entities(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=SUBJECT_A)
 with store.read() as c:
  assert before==[tuple(x) for x in c.execute('SELECT * FROM claim_revisions ORDER BY revision_id')]
  assert c.execute('SELECT COUNT(*) FROM entity_aliases').fetchone()[0]==0

def test_same_known_type_merges_without_changing_type_or_provenance(store):
 from knowledge_weaver.store.entities import _entity_id
 _seed_two_subjects(store)
 with store.transaction() as c:
  for subject in [SUBJECT_A,SUBJECT_B]:c.execute("INSERT INTO entities(entity_id,workspace_id,project_id,entity_type,canonical_name,created_at) VALUES(?,?,?,'device',?,'2026-01-01T00:00:00Z')",(_entity_id('owner:demo','project:songta',subject),'owner:demo','project:songta',subject))
  before=[tuple(x) for x in c.execute('SELECT * FROM evidence_links ORDER BY revision_id,event_id,event_revision,part_index,start_byte,end_byte')]
 merge_entities(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=SUBJECT_A)
 with store.read() as c:
  assert before==[tuple(x) for x in c.execute('SELECT * FROM evidence_links ORDER BY revision_id,event_id,event_revision,part_index,start_byte,end_byte')]
  assert {x[0] for x in c.execute('SELECT entity_type FROM entities')}=={'device'}
 assert {x['subject_id'] for x in current_items(store,workspace_id='owner:demo')}=={SUBJECT_A}

def test_second_destination_requires_explicit_undo_before_remerge(store):
 _seed_two_subjects(store);third='subject:device:third'
 record(store,subject_id=third,predicate='location',value='kitchen',display_text='另一设备在厨房')
 first=merge_entities(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=SUBJECT_A)
 with pytest.raises(EntityError,match='different active alias'):
  merge_entities(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=third)
 assert len(list_aliases(store,workspace_id='owner:demo'))==1
 again=merge_entities(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=SUBJECT_A);assert again['alias_id']==first['alias_id']
 undo_merge(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=SUBJECT_A)
 merge_entities(store,workspace_id='owner:demo',project_id='project:songta',source_subject_id=SUBJECT_B,canonical_subject_id=third)
 with store.read() as c:assert resolve_canonical_subject(c,workspace_id='owner:demo',project_id='project:songta',subject_id=SUBJECT_B)==third
