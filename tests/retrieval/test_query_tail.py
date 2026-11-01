"""A normal multi-part question must not silently lose its last subject."""
from knowledge_weaver.store import Store
from knowledge_weaver.retrieval import keyword_ranking
from ..state.conftest import record


def test_specific_question_after_long_preface_is_searchable(tmp_path):
    store = Store(tmp_path / "kw" / "knowledge.db")
    target = record(store, display_text="逆变器保修期限为三年", value="三年", predicate="inverter.warranty")
    other = record(store, display_text="工作区例行进度汇报", value="进度", predicate="report.status")
    # Distinct legitimate query terms occupy the old 32-term prefix window.
    query = " ".join("topic" + str(i) for i in range(40)) + " 逆变器保修期限"
    result = keyword_ranking(store, candidate_revision_ids={target["revision_id"], other["revision_id"]}, query=query)
    assert result == [target["revision_id"]]


def test_multi_question_keeps_expansion_hypothesis_searchable(tmp_path):
    store = Store(tmp_path / "kw" / "knowledge.db")
    target = record(store, display_text="需要扩容只是推测尚未验证", value="需要扩容", predicate="capacity.expansion", authority="inferred", assertion_kind="inference")
    query = "银杏验收项目的发布标记、核对备份清单待办、用户汇报偏好、服务运行状态分别是什么？扩容是否已经验证？请保留各自状态，并说明来源和时间，最后说明你的身份。"
    result = keyword_ranking(store, candidate_revision_ids={target["revision_id"]}, query=query)
    assert target["revision_id"] in result
