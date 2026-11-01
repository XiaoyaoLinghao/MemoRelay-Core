"""T04: shared redact:1 policy data and synthetic canary cases (KW detection side)."""
from __future__ import annotations

from knowledge_weaver import contracts


def test_policy_health_passes():
    assert contracts.check_redaction_policy() == []
    assert contracts.redaction_policy_version() in contracts.REDACTION_POLICY_VERSIONS


def test_every_synthetic_case_behaves_as_declared():
    for case in contracts.redaction_examples()["cases"]:
        observed = set(contracts.secret_kinds(case["raw"]))
        assert set(case["expect_kinds"]) <= observed, case["id"]
        if case["expected_status"] == "passed":
            assert observed == set(), case["id"]
        if case["expected_status"] == "withheld":
            assert contracts.would_withhold(case["raw"]), case["id"]
        else:
            assert not contracts.would_withhold(case["raw"]), case["id"]


def test_detection_never_returns_secret_values():
    raw = "密码: CANARY-PW-7f3a"
    kinds = contracts.secret_kinds(raw)
    assert kinds == ["password_assignment"]
    assert all("CANARY" not in kind for kind in kinds)


def test_clean_work_text_is_not_flagged():
    for text in (
        "松塔项目端口改为4200。",
        "周备份保留4个成功代，NAS 挂载在 /Volumes/backup。",
        "端口 4200 与 4100 的差异已记录，等待复核。",
        "普通日志：exit code 0，未发现异常。",
    ):
        assert contracts.secret_kinds(text) == [], text
