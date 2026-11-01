"""T20 F09b (round 3): cleanup clocks must compare real UTC instants.

Counterexamples migrated from the round-2 Codex review (review/T20-CODEX-ROUND2.md
F09b): ``sweep`` compared the raw ``--now`` string against the stored ``...Z``
expiry with SQLite string ordering, so a clock carrying a UTC offset expired
data early/late and ``+00:00`` did not match ``Z`` for the same instant.

Decision for a timezone-naive clock: **reject it**.  Stored business anchors keep
the historical lenient reading (naive means UTC per the event contract), but a
cleanup clock decides which bodies are deleted, so an ambiguous instant is
refused instead of guessed.  Tests below pin that decision down.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from knowledge_weaver.cli import main
from knowledge_weaver.lifecycle import (
    classify_claim_revisions,
    is_tombstoned,
    lifecycle_status,
    register_lifecycle,
    sweep,
)
from knowledge_weaver.lifecycle.retention import (
    EXPIRED_BY_DAYS,
    RetentionClockError,
    normalize_clock,
)
from knowledge_weaver.store import fetch_source

from ..state.conftest import create_user_event, record

ANCHOR = "2026-01-01T00:00:00Z"
EXPIRES = "2026-01-31T00:00:00Z"  # temporary_detail: ANCHOR + 30 days


def _temporary_detail(store, *, anchor=ANCHOR):
    event = create_user_event(store, "合成临时调试细节：连接被拒绝")
    subject = f"{event['event_id']}#1@0"
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=subject,
        class_="temporary_detail",
        business_anchor=anchor,
    )
    return event, subject


def _availability(store, event):
    source = fetch_source(store, event["event_id"], event["event_revision"], 0)
    assert source is not None
    return source["availability"]


def _run_cli(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    return code, json.loads("\n".join(lines)) if lines else {}


# ---------------------------------------------------------------------------
# (a) offset clocks compare the real instant, never the raw text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("clock", "expected_availability"),
    [
        # 2026-01-30T20:00:00Z: four hours before the expiry instant -> keep.
        ("2026-01-31T04:00:00+08:00", "full"),
        # 2026-01-31T04:00:00Z: four hours after the expiry instant -> expire.
        ("2026-01-30T20:00:00-08:00", "expired"),
    ],
)
def test_F09b_cleanup_clock_offsets_compare_actual_utc_instant(store, clock, expected_availability):
    event, _subject = _temporary_detail(store)
    sweep(store, now=clock)
    assert _availability(store, event) == expected_availability, {
        "clock": clock,
        "expires_at": EXPIRES,
    }


# ---------------------------------------------------------------------------
# (b) the same exact boundary is equal in every offset spelling, including +00:00
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clock",
    [
        "2026-01-31T00:00:00Z",
        "2026-01-31T00:00:00+00:00",
        "2026-01-31T08:00:00+08:00",
        "2026-01-30T16:00:00-08:00",
    ],
)
def test_F09b_exact_boundary_is_instant_based(store, clock):
    event, _subject = _temporary_detail(store)
    result = sweep(store, now=clock)
    assert result["expired"] == 1, {"clock": clock, "expires_at": EXPIRES}
    assert _availability(store, event) == "expired"


def test_F09b_one_second_before_boundary_in_offset_form_is_kept(store):
    event, _subject = _temporary_detail(store)
    result = sweep(store, now="2026-01-31T07:59:59+08:00")  # 2026-01-30T23:59:59Z
    assert result["expired"] == 0
    assert _availability(store, event) == "full"


def test_F09b_normalize_clock_returns_canonical_utc_z():
    assert normalize_clock("2026-01-31T04:00:00+08:00") == "2026-01-30T20:00:00Z"
    assert normalize_clock("2026-01-30T20:00:00-08:00") == "2026-01-31T04:00:00Z"
    assert normalize_clock("2026-01-31T00:00:00+00:00") == EXPIRES
    assert normalize_clock("2026-01-31T00:00:00Z") == EXPIRES


def test_F09b_sub_second_clock_precision_is_respected(store):
    event, _subject = _temporary_detail(store)
    assert sweep(store, now="2026-01-30T23:59:59.999999Z")["expired"] == 0
    assert _availability(store, event) == "full"
    assert sweep(store, now="2026-01-31T00:00:00.000001Z")["expired"] == 1
    assert _availability(store, event) == "expired"


# ---------------------------------------------------------------------------
# (d) invalid / ambiguous cleanup clocks are refused with zero side effects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clock",
    [
        "",
        "   ",
        "not-a-clock",
        "2026-01-31",
        "2026-01-31T00:00:00",  # naive: the documented reject decision
        "2026-13-01T00:00:00Z",
        "2026-01-31T00:00:00+25:00",
    ],
)
def test_F09b_invalid_or_naive_clock_is_rejected_without_side_effects(store, clock):
    event, subject = _temporary_detail(store)
    with store.read() as conn:
        before = dict(
            conn.execute(
                "SELECT * FROM lifecycle_records WHERE subject_id = ?", (subject,)
            ).fetchone()
        )
    with pytest.raises(RetentionClockError) as excinfo:
        sweep(store, now=clock)
    assert excinfo.value.code == "INVALID_CLOCK"
    message = str(excinfo.value).lower()
    assert "clock" in message and ("timezone" in message or "rfc 3339" in message)
    with store.read() as conn:
        after = dict(
            conn.execute(
                "SELECT * FROM lifecycle_records WHERE subject_id = ?", (subject,)
            ).fetchone()
        )
        tombstones = conn.execute("SELECT COUNT(*) AS c FROM tombstones").fetchone()["c"]
        status_changed = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_revisions WHERE status = 'expired'"
        ).fetchone()["c"]
    assert after == before, "a refused clock must not rewrite lifecycle records"
    assert tombstones == 0 and status_changed == 0
    assert _availability(store, event) == "full"


def test_F09b_classify_rejects_invalid_clock_before_writing(store):
    create_user_event(store, "端口4200")
    with pytest.raises(RetentionClockError):
        classify_claim_revisions(store, now="2026-01-31T00:00:00")  # naive
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM lifecycle_records").fetchone()["c"] == 0


def test_F09b_lifecycle_status_uses_the_same_instant_comparison(store):
    _temporary_detail(store)
    assert lifecycle_status(store, now="2026-01-31T04:00:00+08:00")["due_now"] == 0
    assert lifecycle_status(store, now="2026-01-30T20:00:00-08:00")["due_now"] == 1
    assert lifecycle_status(store, now="2026-01-31T00:00:00+00:00")["due_now"] == 1


# ---------------------------------------------------------------------------
# (c) 30/180/365-day Z boundaries keep their existing meaning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_", ["temporary_detail", "completed_task", "superseded_history"])
def test_F09b_day_boundaries_do_not_regress(store, class_):
    days = EXPIRED_BY_DAYS[class_]
    event = create_user_event(store, "合成历史细节")
    subject = f"{event['event_id']}#1@0"
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=subject,
        class_=class_,
        business_anchor=ANCHOR,
    )
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    before = (start + dt.timedelta(days=days - 1)).isoformat().replace("+00:00", "Z")
    boundary = (start + dt.timedelta(days=days)).isoformat().replace("+00:00", "Z")
    assert sweep(store, now=before)["expired"] == 0
    assert _availability(store, event) == "full"
    assert sweep(store, now=boundary)["expired"] == 1
    assert _availability(store, event) == "expired"


# ---------------------------------------------------------------------------
# internal write formats stay canonical Z even when the clock has an offset
# ---------------------------------------------------------------------------


def test_F09b_stored_event_part_timestamps_are_canonical_z(store):
    _event, subject = _temporary_detail(store, anchor="2026-01-01T08:00:00+08:00")
    result = sweep(store, now="2026-02-01T08:00:00+08:00")  # 2026-02-01T00:00:00Z
    assert result["expired"] == 1
    with store.read() as conn:
        record_ = conn.execute(
            "SELECT business_anchor, expires_at, updated_at FROM lifecycle_records "
            "WHERE subject_id = ?",
            (subject,),
        ).fetchone()
        tombstone = conn.execute(
            "SELECT created_at FROM tombstones WHERE subject_id = ?", (subject,)
        ).fetchone()
    assert record_["business_anchor"] == "2026-01-01T00:00:00Z"
    assert record_["expires_at"] == EXPIRES
    assert record_["updated_at"] == "2026-02-01T00:00:00Z"
    assert tombstone["created_at"] == "2026-02-01T00:00:00Z"


def test_F09b_expired_revision_status_changed_at_is_canonical_z(store):
    first = record(store, display_text="旧端口4100", value=4100)
    record(
        store,
        display_text="端口改为4200",
        value=4200,
        supersedes=[first["revision_id"]],
        event_time={"value": "2026-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    result = sweep(store, now="2027-01-02T08:00:00+08:00")  # 2027-01-02T00:00:00Z
    assert result["claim_revisions"] == 1
    with store.read() as conn:
        revision = conn.execute(
            "SELECT status, status_changed_at FROM claim_revisions WHERE revision_id = ?",
            (first["revision_id"],),
        ).fetchone()
    assert revision["status"] == "expired"
    assert revision["status_changed_at"] == "2027-01-02T00:00:00Z"


# ---------------------------------------------------------------------------
# CLI --now goes through the same normalization/rejection path
# ---------------------------------------------------------------------------


def test_F09b_cli_cleanup_now_uses_real_instant_and_rejects_bad_clock(store, capsys):
    event, subject = _temporary_detail(store)
    db = str(store.db_path)
    # Early offset clock: a dry run reports nothing due and expires nothing.
    code, payload = _run_cli(
        capsys, ["cleanup", "--db", db, "--dry-run", "--now", "2026-01-31T04:00:00+08:00"]
    )
    assert code == 0 and payload["ok"] is True and payload["dry_run"] is True
    assert payload["due_now"] == 0
    assert not is_tombstoned(store, "event_part", subject)
    # Naive/invalid clock: JSON error, non-zero exit, no side effects.
    for clock in ("2026-01-31T00:00:00", "not-a-clock"):
        code, payload = _run_cli(capsys, ["cleanup", "--db", db, "--now", clock])
        assert code != 0 and payload["ok"] is False
        assert payload["error"]["code"] == "INVALID_CLOCK"
        assert "clock" in payload["error"]["message"].lower()
        assert not is_tombstoned(store, "event_part", subject)
        assert _availability(store, event) == "full"
    # Late offset clock: real instant is past the expiry -> expires.
    code, payload = _run_cli(capsys, ["cleanup", "--db", db, "--now", "2026-01-30T20:00:00-08:00"])
    assert code == 0 and payload["ok"] is True and payload["expired"] == 1
    assert is_tombstoned(store, "event_part", subject)
    assert _availability(store, event) == "expired"
