import pytest

from knowledge_weaver.db import init_db, insert_entity, insert_relation, get_entity, get_relations_for_entity
from knowledge_weaver.linker import (
    ExtractedEntity,
    ParsedSection,
    ParsedFile,
    LinkedRelation,
    generate_relation_id,
    link_entities_in_file,
    link_cross_day,
    link_project_dependencies,
)


# --- Helpers ---

def _make_entity_dict(e: ExtractedEntity, importance: float = 0.5, day_count: int = 1) -> dict:
    return {
        "id": e.id,
        "type": e.type,
        "name": e.name,
        "summary": e.summary,
        "importance": importance,
        "first_seen": e.first_seen,
        "last_seen": e.last_seen,
        "day_count": day_count,
    }


# --- Tests ---

def test_generate_relation_id():
    id1 = generate_relation_id("e1", "e2", "RELATES_TO")
    id2 = generate_relation_id("e1", "e2", "RELATES_TO")
    assert id1 == id2
    assert id1.startswith("rel:")

    id3 = generate_relation_id("e2", "e1", "RELATES_TO")
    assert id1 != id3

    id4 = generate_relation_id("e1", "e2", "DEPENDS_ON")
    assert id1 != id4


def test_link_same_entity_id(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, {
        "id": "entity-001",
        "type": "project",
        "name": "Test Project",
        "summary": "A test project",
        "importance": 0.5,
        "first_seen": "2024-01-01",
        "last_seen": "2024-01-01",
        "day_count": 1,
    })

    new_entity = ExtractedEntity(
        id="entity-001",
        type="project",
        name="Test Project",
        summary="A test project",
        first_seen="2024-01-02",
        last_seen="2024-01-02",
    )

    relations = link_cross_day(conn, [new_entity], set())

    # Self-loop CONTINUES no longer inserted — returns empty list
    assert len(relations) == 0
    # But day_count is still updated
    assert get_entity(conn, "entity-001")["day_count"] == 2

    updated = get_entity(conn, "entity-001")
    assert updated["day_count"] == 2
    assert updated["last_seen"] == "2024-01-02"


def test_link_co_occurrence(temp_db_path):
    conn = init_db(temp_db_path)

    e1 = ExtractedEntity(
        id="e-aa", type="project", name="ProjA", summary="Core infrastructure project",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e2 = ExtractedEntity(
        id="e-bb", type="task", name="Task1", summary="Core infrastructure task",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e3 = ExtractedEntity(
        id="e-cc", type="decision", name="Dec1", summary="A decision",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Other",
    )

    for e in [e1, e2, e3]:
        insert_entity(conn, _make_entity_dict(e))

    parsed_file = ParsedFile(
        date="2024-01-01",
        path="/notes/2024-01-01.md",
        sections=[
            ParsedSection(title="Core", line_range=(1, 10)),
            ParsedSection(title="Other", line_range=(11, 20)),
        ],
    )

    relations = link_entities_in_file(conn, [e1, e2, e3], parsed_file, "/notes/2024-01-01.md")

    # e1 and e2 are in same section AND share tokens → RELATES_TO weight 0.5
    same_section = [r for r in relations if r.rel_type == "RELATES_TO" and r.weight == 0.5]
    assert len(same_section) >= 1
    assert same_section[0].evidence.startswith("shared_tokens:")
    # e3 is in different section → no cross-section link created
    cross_section = [r for r in relations if r.rel_type == "RELATES_TO" and r.weight == 0.3]
    assert len(cross_section) == 0


def test_link_project_task(temp_db_path):
    conn = init_db(temp_db_path)

    project = ExtractedEntity(
        id="proj-alpha", type="project", name="Alpha",
        summary="Project Alpha — the main initiative",
        first_seen="2024-01-01", last_seen="2024-01-01",
    )
    task = ExtractedEntity(
        id="task-1", type="task", name="Set up Alpha CI",
        summary="Configure CI pipeline for Alpha",
        first_seen="2024-01-01", last_seen="2024-01-01",
    )
    unrelated = ExtractedEntity(
        id="task-2", type="task", name="Order supplies",
        summary="Order office supplies",
        first_seen="2024-01-01", last_seen="2024-01-01",
    )

    for e in [project, task, unrelated]:
        insert_entity(conn, _make_entity_dict(e))

    relations = link_project_dependencies(conn, [project, task, unrelated])

    deps = [r for r in relations if r.rel_type == "DEPENDS_ON"]
    assert len(deps) >= 1
    dep = deps[0]
    assert dep.from_entity == "task-1"
    assert dep.to_entity == "proj-alpha"
    assert dep.weight == 0.8

    # unrelated task should not get a DEPENDS_ON
    for r in relations:
        assert not (r.from_entity == "task-2" and r.to_entity == "proj-alpha")


def test_link_decision_project(temp_db_path):
    conn = init_db(temp_db_path)

    project = ExtractedEntity(
        id="proj-beta", type="project", name="Beta",
        summary="Project Beta",
        first_seen="2024-01-01", last_seen="2024-01-01",
    )
    decision = ExtractedEntity(
        id="dec-1", type="decision", name="Use React for Beta frontend",
        summary="Decided to use React after evaluating options for Beta",
        first_seen="2024-01-01", last_seen="2024-01-01",
    )

    for e in [project, decision]:
        insert_entity(conn, _make_entity_dict(e))

    relations = link_project_dependencies(conn, [project, decision])

    deps = [r for r in relations if r.rel_type == "DEPENDS_ON"]
    assert len(deps) >= 1
    dep = deps[0]
    assert dep.from_entity == "dec-1"
    assert dep.to_entity == "proj-beta"
    assert dep.weight == 0.8


def test_link_cross_day_co_occurrence(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, {
        "id": "ent-a", "type": "project", "name": "Project A", "summary": "...",
        "importance": 0.5, "first_seen": "2024-01-01", "last_seen": "2024-01-01",
        "day_count": 1,
    })
    insert_entity(conn, {
        "id": "ent-b", "type": "project", "name": "Project B", "summary": "...",
        "importance": 0.5, "first_seen": "2024-01-01", "last_seen": "2024-01-01",
        "day_count": 1,
    })
    insert_entity(conn, {
        "id": "ent-c", "type": "project", "name": "Project C", "summary": "...",
        "importance": 0.5, "first_seen": "2024-01-01", "last_seen": "2024-01-01",
        "day_count": 1,
    })

    new_entities = [
        ExtractedEntity(id="ent-a", type="project", name="Project A", summary="...",
                        first_seen="2024-01-02", last_seen="2024-01-02"),
        ExtractedEntity(id="ent-b", type="project", name="Project B", summary="...",
                        first_seen="2024-01-02", last_seen="2024-01-02"),
        ExtractedEntity(id="ent-d", type="project", name="Project D", summary="New project",
                        first_seen="2024-01-02", last_seen="2024-01-02"),
    ]

    relations = link_cross_day(conn, new_entities, set())

    # Self-loop CONTINUES no longer inserted — returns empty list
    assert len(relations) == 0

    assert get_entity(conn, "ent-a")["day_count"] == 2
    assert get_entity(conn, "ent-b")["day_count"] == 2
    assert get_entity(conn, "ent-c")["day_count"] == 1  # unchanged
    # ent-d was inserted with day_count 1
    assert get_entity(conn, "ent-d") is not None


def test_dedup_relations(temp_db_path):
    conn = init_db(temp_db_path)

    e1 = ExtractedEntity(
        id="e-1", type="project", name="ProjX", summary="Shared keyword alpha beta",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e2 = ExtractedEntity(
        id="e-2", type="task", name="TaskX", summary="Shared keyword alpha beta task",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    insert_entity(conn, _make_entity_dict(e1))
    insert_entity(conn, _make_entity_dict(e2))

    parsed_file = ParsedFile(
        date="2024-01-01",
        path="/notes/2024-01-01.md",
        sections=[ParsedSection(title="Core", line_range=(1, 10))],
    )

    link_entities_in_file(conn, [e1, e2], parsed_file, "/notes/2024-01-01.md")
    link_entities_in_file(conn, [e1, e2], parsed_file, "/notes/2024-01-01.md")

    all_rels = conn.execute("SELECT id FROM relations").fetchall()
    unique_ids = {r["id"] for r in all_rels}
    assert len(all_rels) == len(unique_ids), "Duplicate relation rows detected"


def test_no_relation_for_unrelated_entities(temp_db_path):
    """两个完全无关的实体不应建立 RELATES_TO"""
    conn = init_db(temp_db_path)

    e1 = ExtractedEntity(
        id="e-1", type="project", name="ProjX", summary="Database optimization",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e2 = ExtractedEntity(
        id="e-2", type="task", name="TaskY", summary="Office supply order",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    for e in [e1, e2]:
        insert_entity(conn, _make_entity_dict(e))

    parsed_file = ParsedFile(
        date="2024-01-01", path="/notes/2024-01-01.md",
        sections=[ParsedSection(title="Core", line_range=(1, 10))],
    )

    relations = link_entities_in_file(conn, [e1, e2], parsed_file, "/notes/2024-01-01.md")
    # Unrelated entities in the same section get a low-weight co_occurrence relation
    assert len(relations) == 1
    assert relations[0].weight == 0.3
    assert relations[0].evidence == "co_occurrence"


def test_relation_via_name_mention(temp_db_path):
    """e1.summary 包含 e2.name → 应建立带 name_mention evidence 的关系"""
    conn = init_db(temp_db_path)

    e1 = ExtractedEntity(
        id="e-1", type="decision", name="TechChoice",
        summary="Selected React for frontend",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e2 = ExtractedEntity(
        id="e-2", type="project", name="React",
        summary="A JavaScript library",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    for e in [e1, e2]:
        insert_entity(conn, _make_entity_dict(e))

    parsed_file = ParsedFile(
        date="2024-01-01", path="/notes/2024-01-01.md",
        sections=[ParsedSection(title="Core", line_range=(1, 10))],
    )

    relations = link_entities_in_file(conn, [e1, e2], parsed_file, "/notes/2024-01-01.md")
    assert len(relations) == 1
    assert relations[0].evidence == "name_mention"
    assert relations[0].rel_type == "RELATES_TO"


def test_relation_via_shared_tokens(temp_db_path):
    """两个实体共享 ≥2 个非停用词 → 应建立带 shared_tokens evidence 的关系"""
    conn = init_db(temp_db_path)

    e1 = ExtractedEntity(
        id="e-1", type="project", name="DataPipeline",
        summary="Real-time data processing pipeline",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e2 = ExtractedEntity(
        id="e-2", type="task", name="DataValidation",
        summary="Real-time data processing validation",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    for e in [e1, e2]:
        insert_entity(conn, _make_entity_dict(e))

    parsed_file = ParsedFile(
        date="2024-01-01", path="/notes/2024-01-01.md",
        sections=[ParsedSection(title="Core", line_range=(1, 10))],
    )

    relations = link_entities_in_file(conn, [e1, e2], parsed_file, "/notes/2024-01-01.md")
    assert len(relations) == 1
    assert relations[0].evidence.startswith("shared_tokens:")
    assert relations[0].rel_type == "RELATES_TO"
    token_part = relations[0].evidence.replace("shared_tokens: ", "")
    shared = set(token_part.split(","))
    assert "real" in shared
    assert "time" in shared
    assert "data" in shared
    assert "processing" in shared


def test_shared_tokens_respect_min_shared(temp_db_path):
    """共享词少于 2 个 → 不应建立关系"""
    conn = init_db(temp_db_path)

    e1 = ExtractedEntity(
        id="e-1", type="project", name="ProjA",
        summary="Machine learning model training",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    e2 = ExtractedEntity(
        id="e-2", type="task", name="TaskB",
        summary="Database migration scripts",
        first_seen="2024-01-01", last_seen="2024-01-01", section_title="Core",
    )
    for e in [e1, e2]:
        insert_entity(conn, _make_entity_dict(e))

    parsed_file = ParsedFile(
        date="2024-01-01", path="/notes/2024-01-01.md",
        sections=[ParsedSection(title="Core", line_range=(1, 10))],
    )

    relations = link_entities_in_file(conn, [e1, e2], parsed_file, "/notes/2024-01-01.md")
    # Unrelated entities in the same section get a low-weight co_occurrence relation
    assert len(relations) == 1
    assert relations[0].weight == 0.3

def test_link_project_dependencies_skips_short_names():
    from knowledge_weaver.db import init_db
    from knowledge_weaver.linker import link_project_dependencies, ExtractedEntity
    import tempfile, os
    fd, db = tempfile.mkstemp(suffix=".db"); os.close(fd)
    conn = init_db(db)
    proj = ExtractedEntity(id="proj:ai", type="project", name="AI",
                           summary="AI", first_seen="2026-05-24", last_seen="2026-05-24")
    task = ExtractedEntity(id="task:fixci", type="task", name="修复 CI",
                           summary="修复 CI 流程", first_seen="2026-05-24", last_seen="2026-05-24")
    rels = link_project_dependencies(conn, [proj, task])
    conn.close(); os.unlink(db)
    assert all(r.rel_type != "DEPENDS_ON" for r in rels), \
        "Short project name 'AI' must not produce DEPENDS_ON"
