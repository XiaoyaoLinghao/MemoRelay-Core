"""Builders for claim-projection tests: real events -> verified evidence spans."""
from __future__ import annotations

import itertools

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.store import Store, receive_batch
from knowledge_weaver.store.claims import record_claim_revision

from ..store.conftest import make_batch, make_event

_counter = itertools.count(1)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def create_user_event(store, text, *, native_id=None):
    native = native_id or f"message:{next(_counter)}"
    event = make_event(native_id=native, text=text)
    receive_batch(store, make_batch([event]))
    return event


def evidence_ref(event, *, start_byte=0, end_byte=None):
    text = event["body"]["text"]
    if end_byte is None:
        end_byte = len(text.encode("utf-8"))
    return {
        "event_id": event["event_id"],
        "event_revision": event["event_revision"],
        "part_index": event["body"]["part_index"],
        "start_byte": start_byte,
        "end_byte": end_byte,
        "body_sha256": event["body"]["sha256"],
        "availability": "full",
    }


def record(
    store,
    *,
    display_text,
    value,
    predicate="server.port",
    facet="desired",
    authority="user_confirmed",
    assertion_kind="explicit_statement",
    project_id="project:songta",
    subject_id="subject:songta-server",
    qualifiers=None,
    event_time=None,
    native_id=None,
    supersedes=(),
    correction_of=None,
    evidence_event=None,
):
    event = evidence_event or create_user_event(store, display_text, native_id=native_id)
    return record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id=project_id,
        subject_id=subject_id,
        predicate=predicate,
        facet=facet,
        qualifiers=qualifiers or {},
        value=value,
        display_text=display_text,
        authority=authority,
        assertion_kind=assertion_kind,
        event_time=event_time or event["event_time"],
        evidence=[evidence_ref(event)],
        supersedes=supersedes,
        correction_of=correction_of,
    )


def current_item(store, *, predicate="server.port", facet="desired", include_states=("current", "conflict", "uncertain")):
    from knowledge_weaver.store import current_items

    items = current_items(store, workspace_id="owner:demo", include_states=include_states)
    for item in items:
        if item["predicate"] == predicate and item["facet"] == facet:
            return item
    return None
