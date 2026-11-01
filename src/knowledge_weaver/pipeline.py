"""Consolidation pipeline — orchestrates parse → extract → link → score → embed → manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from knowledge_weaver.db import (
    get_access_count,
    get_entity,
    get_manifest,
    init_db,
    insert_entity,
    insert_relation,
    insert_review,
    record_merge,
    upsert_manifest,
)
from knowledge_weaver.embedder import EmbeddingClient
from knowledge_weaver.extractor import ExtractedEntity, extract_entities_from_section
from knowledge_weaver.resolver import load_alias_map, resolve_entity
from knowledge_weaver.linker import (
    ExtractedEntity as LinkerEntity,
    LinkedRelation,
    ParsedFile as LinkerParsedFile,
    ParsedSection as LinkerParsedSection,
    link_cross_day,
    link_entities_in_file,
    link_project_dependencies,
)
from knowledge_weaver.parser import ParsedFile, parse_dma_file
from knowledge_weaver.scorer import ImportanceScorer

logger = logging.getLogger(__name__)

# Pattern for valid DMA daily file names: YYYY-MM-DD.md or sample_YYYY-MM-DD.md
_DATE_FILE_RE = re.compile(r"^(?:sample_)?(\d{4}-\d{2}-\d{2})\.md$")


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    status: str = "ok"  # "ok" / "partial" / "error"
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    relations_created: int = 0
    errors: list[str] = field(default_factory=list)


def compute_file_hash(filepath: str) -> str:
    """SHA256 hash of file content."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _to_linker_entity(
    extracted: ExtractedEntity, section_title: str, parsed_date: str
) -> LinkerEntity:
    """Adapt extractor.ExtractedEntity → linker.ExtractedEntity."""
    return LinkerEntity(
        id=extracted.id,
        type=extracted.type,
        name=extracted.name,
        summary=extracted.summary,
        first_seen=parsed_date,
        last_seen=parsed_date,
        section_title=section_title,
    )


def _to_linker_parsed_file(
    parsed: ParsedFile, source_path: str
) -> LinkerParsedFile:
    """Adapt parser.ParsedFile → linker.ParsedFile."""
    sections = [
        LinkerParsedSection(
            title=s.title,
            line_range=(
                s.items[0].line_start if s.items else 0,
                s.items[-1].line_end if s.items else 0,
            ),
        )
        for s in parsed.sections
    ]
    return LinkerParsedFile(date=parsed.date, path=source_path, sections=sections)


def run_consolidation(
    db_path: str,
    memory_dir: str | None = None,
    embedder: EmbeddingClient | None = None,
    today: date | None = None,
    memory_dirs: list[tuple[str, str]] | None = None,
) -> ConsolidationResult:
    """Main consolidation pipeline.

    1. Scan memory dirs for YYYY-MM-DD.md files
    2. Check daily_manifest for already-processed unchanged files
    3. Parse each new/changed file
    4. Extract entities
    5. Link cross-day entities
    6. Score importance
    7. Embed summaries (if embedder available)
    8. Update daily_manifest

    memory_dirs: list of (source_name, path) tuples. Takes priority over memory_dir.
    memory_dir: legacy single-dir param. Treated as [("default", memory_dir)] if provided without memory_dirs.
    """
    if today is None:
        today = date.today()

    if memory_dirs is None:
        if memory_dir is not None:
            memory_dirs = [("default", memory_dir)]
        else:
            raise ValueError("Must provide memory_dir or memory_dirs")

    result = ConsolidationResult()
    conn = init_db(db_path)
    scorer = ImportanceScorer()

    try:
        # Collect all existing entity IDs for cross-day linking
        existing_ids: set[str] = {
            row["id"]
            for row in conn.execute("SELECT id FROM entities").fetchall()
        }

        for source_name, src_dir in memory_dirs:
            # Step 1: Discover candidate files
            candidates = _discover_files(src_dir)

            # Step 2: Filter out unchanged files via manifest hash check
            to_process, to_skip = _filter_unchanged(conn, candidates, src_dir, source_name)
            result.files_skipped += len(to_skip)

            # Process each file
            for date_str, filename in to_process:
                try:
                    _process_file(
                        conn=conn,
                        date_str=date_str,
                        filename=filename,
                        memory_dir=src_dir,
                        embedder=embedder,
                        scorer=scorer,
                        today=today,
                        existing_ids=existing_ids,
                        result=result,
                        source_name=source_name,
                    )
                except Exception as exc:
                    result.files_failed += 1
                    result.errors.append(f"{source_name}/{filename}: {exc}")
                    logger.exception("Failed to process %s/%s", source_name, filename)

        # Final status determination
        if result.files_failed > 0:
            if result.files_processed > 0:
                result.status = "partial"
            else:
                result.status = "error"
        else:
            result.status = "ok"

    except Exception as exc:
        result.errors.append(str(exc))
        result.status = "error"
    finally:
        # fix: VACUUM to reclaim space from DELETE/INSERT operations
        try:
            conn.execute("VACUUM")
        except Exception:
            pass
        conn.close()

    return result


def _discover_files(memory_dir: str) -> list[tuple[str, str]]:
    """Find all YYYY-MM-DD.md files in memory_dir.

    Returns list of (date_str, filename) sorted by date.
    """
    candidates: list[tuple[str, str]] = []
    try:
        entries = os.listdir(memory_dir)
    except OSError:
        return candidates

    for name in entries:
        m = _DATE_FILE_RE.match(name)
        if m and not name.startswith("."):
            date_str = m.group(1)  # extract date from filename
            candidates.append((date_str, name))

    candidates.sort(key=lambda x: x[0])
    return candidates


def _filter_unchanged(
    conn,
    candidates: list[tuple[str, str]],
    memory_dir: str,
    source_name: str = "default",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split candidates into (to_process, to_skip) based on manifest hash."""
    to_process: list[tuple[str, str]] = []
    to_skip: list[tuple[str, str]] = []

    for date_str, filename in candidates:
        filepath = os.path.join(memory_dir, filename)
        file_hash = compute_file_hash(filepath)
        manifest = get_manifest(conn, date_str, source_name)

        if manifest and manifest["file_hash"] == file_hash:
            to_skip.append((date_str, filename))
        else:
            to_process.append((date_str, filename))

    return to_process, to_skip


def _find_similar_entity(conn, entity_type: str, name: str, threshold: float = 0.85,
                         embedder=None, summary: str = "") -> str | None:
    """Find an existing same-type entity with a highly similar name.

    Strategy:
    1. SequenceMatcher on names (threshold >= 0.85)
    2. If embedder available, also check embedding cosine similarity (>= 0.85)
       for candidates where name similarity is between 0.70 and 0.85.

    Only compares against entities first seen within the last 30 days.
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    rows = conn.execute(
        "SELECT id, name FROM entities WHERE type=? AND first_seen >= ?",
        (entity_type, cutoff),
    ).fetchall()

    best_id, best_ratio = None, 0.0
    borderline: list[tuple[str, float]] = []  # (id, ratio) for 0.70-0.85 range

    for row in rows:
        ratio = SequenceMatcher(None, name.lower(), row["name"].lower()).ratio()
        if ratio >= threshold:
            if ratio > best_ratio:
                best_ratio, best_id = ratio, row["id"]
        elif ratio >= 0.70:
            borderline.append((row["id"], ratio))

    if best_id:
        return best_id

    # Embedding-based second pass for borderline candidates
    if embedder is not None and summary and borderline:
        try:
            query_vec = embedder.embed(summary[:500])
            if not query_vec:
                return None
            from knowledge_weaver.db import search_entity_vectors
            # Get top matches from vector search
            vec_results = search_entity_vectors(conn, query_vec, limit=20)
            vec_ids = {r["id"] for r in vec_results}
            for bid, _ in borderline:
                if bid in vec_ids:
                    return bid
        except Exception:
            pass

    return None


def _process_file(
    conn,
    date_str: str,
    filename: str,
    memory_dir: str,
    embedder: EmbeddingClient | None,
    scorer: ImportanceScorer,
    today: date,
    existing_ids: set[str],
    result: ConsolidationResult,
    source_name: str = "default",
) -> None:
    """Process a single DMA file through the full pipeline."""
    filepath = os.path.join(memory_dir, filename)
    file_hash = compute_file_hash(filepath)

    # Step 3: Parse
    parsed = parse_dma_file(filepath)
    if not parsed.sections:
        # Empty file or parse failure — record manifest but allow retry
        upsert_manifest(conn, {
            "date": date_str,
            "source": source_name,
            "file_path": filepath,
            "file_hash": "",  # don't lock the hash; allow retry on next run
            "entity_count": 0,
            "status": "empty",
        })
        result.files_processed += 1
        return

    # Step 4: Extract entities — W2 structured handoff (default regex; opt-in JSON).
    all_extracted: list[ExtractedEntity] = []
    # Map entity_id → section_title (for linker adaptation)
    entity_section_map: dict[str, str] = {}

    relative_path = f"memory/{filename}"

    json_entities = None
    if os.environ.get("KNOWLEDGE_WEAVER_EXTRACT_MODE", "regex").lower() == "json":
        try:
            from knowledge_weaver.json_facts import (
                extract_json_facts_block,
                facts_to_entities,
            )
            with open(filepath, "r", encoding="utf-8") as _f:
                _facts = extract_json_facts_block(_f.read())
            if _facts is not None:
                json_entities = facts_to_entities(_facts, relative_path)
        except Exception as exc:
            logger.warning("JSON facts extraction failed for %s: %s; falling back to regex",
                           filename, exc)
            json_entities = None

    if json_entities is not None:
        # Structured path: facts already carry clean type/name; no section context.
        all_extracted = json_entities
        for ent in all_extracted:
            entity_section_map[ent.id] = ""
    else:
        for section in parsed.sections:
            # W2: never regex-scrape the structured-facts JSON block (it is for the
            # JSON path only; scraping its JSON text would create garbage entities,
            # e.g. once a historical file has been backfilled with a facts block).
            if "结构化事实" in (section.title or ""):
                continue
            # Pass the DMA category name (section.title) so extract_entities_from_item
            # can map it to the correct entity type via CATEGORY_TO_TYPE.
            # section.category holds the pre-mapped entity type; pass the raw title instead.
            section_entities = extract_entities_from_section(
                section, relative_path, dma_category=section.title
            )
            for ent in section_entities:
                if ent.id not in {e.id for e in all_extracted}:
                    all_extracted.append(ent)
                    entity_section_map[ent.id] = section.title

    # Step 6 & 7: Score and upsert entities + embed (before linking,
    # because link_cross_day also inserts entities)
    entity_count = 0
    texts_to_embed: list[str] = []
    entity_ids_to_embed: list[str] = []
    # W1: registered alias -> canonical id map (mtime-cached; empty if no registry)
    alias_map = load_alias_map()

    for extracted in all_extracted:
        # Build entity dict for DB
        db_entity = get_entity(conn, extracted.id)
        if db_entity is not None:
            # Entity exists — update
            new_day_count = db_entity["day_count"] + 1
            first_seen = min(db_entity["first_seen"], date_str)
            result.entities_updated += 1
        else:
            # W1 entity resolution — vector-first three-band resolver (replaces the
            # legacy name-first _find_similar_entity, which missed cross-language
            # aliases like HomeBrain / 家庭大脑).
            res = resolve_entity(
                conn, new_id=extracted.id, entity_type=extracted.type,
                name=extracted.name, summary=extracted.summary,
                embedder=embedder, alias_to_canonical=alias_map,
            )
            if res.action in ("normalize", "merge") and res.into_id:
                into = get_entity(conn, res.into_id)
                orig_id = extracted.id
                old_sources = json.loads(into["source_lines"] or "[]") if into else []
                new_sources = json.loads(extracted.source_lines or "[]")
                merged_sources = list(dict.fromkeys(old_sources + new_sources))
                # Preserve the established (canonical) name; record the divergent
                # incoming name as an alias rather than overwriting it.
                meta = dict(extracted.metadata) if isinstance(extracted.metadata, dict) else {}
                surviving_name = extracted.name
                if into is not None:
                    surviving_name = into["name"]
                    old_meta = json.loads(into["metadata"] or "{}")
                    aliases = list(old_meta.get("aliases", []))
                    if extracted.name and extracted.name not in aliases \
                            and extracted.name != surviving_name:
                        aliases.append(extracted.name)
                    if aliases:
                        meta["aliases"] = aliases
                    logger.debug("Resolved %s → %s (%s, score=%.3f)",
                                 orig_id, res.into_id, res.reason, res.score)
                    record_merge(
                        conn, merged_from_id=orig_id, merged_into_id=res.into_id,
                        reason=res.reason, score=res.score,
                        from_snapshot=json.dumps({
                            "id": orig_id, "name": extracted.name,
                            "type": extracted.type, "summary": extracted.summary,
                        }, ensure_ascii=False),
                        auto_commit=False,
                    )
                extracted = ExtractedEntity(
                    id=res.into_id,
                    type=extracted.type,
                    name=surviving_name,
                    summary=extracted.summary,
                    source_lines=json.dumps(merged_sources, ensure_ascii=False),
                    metadata=meta,
                )
                db_entity = into
            elif res.action == "review" and res.into_id:
                # Uncertain middle band: keep BOTH entities live, queue for review.
                insert_review(
                    conn, kind="merge", new_entity_id=extracted.id,
                    candidate_id=res.into_id, entity_type=extracted.type,
                    score=res.score, reason=res.reason, auto_commit=False,
                )

            if db_entity is not None:
                new_day_count = db_entity["day_count"] + 1
                first_seen = min(db_entity["first_seen"], date_str)
                result.entities_updated += 1
            else:
                new_day_count = 1
                first_seen = date_str
                result.entities_created += 1

        # Compute importance score
        try:
            last_seen_date = date.fromisoformat(date_str)
            days_since = max(0, (today - last_seen_date).days)
        except ValueError:
            days_since = 0

        # Count categories from metadata
        meta = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        tags = meta.get("tags", [])
        tag_count = len(tags) if isinstance(tags, list) else 0
        access_count = get_access_count(conn, extracted.id) if db_entity else 0

        # Compute recent_day_count (days seen in last 30 days)
        recent_day_count = new_day_count  # default fallback
        if db_entity:
            cutoff_30 = (today - __import__("datetime").timedelta(days=30)).isoformat()
            row = conn.execute(
                "SELECT COUNT(DISTINCT date) as cnt FROM daily_manifest WHERE date >= ?",
                (cutoff_30,),
            ).fetchone()
            if row:
                # Approximate: assume entity seen on each manifest day since first_seen
                first = max(db_entity["first_seen"], cutoff_30)
                last = db_entity["last_seen"]
                if first <= last:
                    recent_day_count = min(new_day_count, row["cnt"])

        importance = scorer.calculate(
            days_since_last_seen=days_since,
            day_count=new_day_count,
            distinct_categories=1,  # one category per extraction
            tag_count=tag_count,
            access_count=access_count,
            entity_type=extracted.type,
            recent_day_count=recent_day_count,
        )

        # Attach source name to metadata (KW-F2)
        meta_dict = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        meta_dict["source"] = source_name
        entity_dict = {
            "id": extracted.id,
            "type": extracted.type,
            "name": extracted.name,
            "summary": extracted.summary,
            "importance": importance,
            "first_seen": first_seen,
            "last_seen": date_str,
            "day_count": new_day_count,
            "source_lines": extracted.source_lines,
            "metadata": json.dumps(meta_dict, ensure_ascii=False),
        }
        insert_entity(conn, entity_dict, auto_commit=False)
        existing_ids.add(extracted.id)
        entity_count += 1

        # Collect for batch embedding (skip if already has a vector)
        if embedder is not None and extracted.summary:
            existing_vec = conn.execute(
                "SELECT 1 FROM entity_vectors WHERE entity_id=?", (extracted.id,)
            ).fetchone()
            if not existing_vec:
                texts_to_embed.append(extracted.summary[:500])
                entity_ids_to_embed.append(extracted.id)

    # Step 5: Link — within-file co-occurrence
    linker_entities = [
        _to_linker_entity(e, entity_section_map.get(e.id, ""), date_str)
        for e in all_extracted
    ]
    linker_parsed = _to_linker_parsed_file(parsed, filepath)

    in_file_relations = link_entities_in_file(
        conn, linker_entities, linker_parsed, filepath
    )
    result.relations_created += len(in_file_relations)

    # Step 5b: Cross-day linking
    cross_day_relations = link_cross_day(conn, linker_entities, existing_ids)
    result.relations_created += len(cross_day_relations)

    # Step 5c: Project dependencies
    proj_relations = link_project_dependencies(conn, linker_entities)
    result.relations_created += len(proj_relations)

    # Commit all entity and relation writes for this file in one batch
    conn.commit()

    # Step 7b: Batch embed and store vectors
    if embedder is not None and texts_to_embed:
        try:
            vectors = embedder.embed_batch(texts_to_embed)
            from knowledge_weaver.db import upsert_entity_vector
            for eid, vec in zip(entity_ids_to_embed, vectors):
                if vec:  # skip empty results
                    try:
                        upsert_entity_vector(conn, eid, vec, auto_commit=False)
                    except Exception:
                        pass  # vector table may not exist
            conn.commit()
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", filename, exc)

    # Step 8: Update manifest
    upsert_manifest(conn, {
        "date": date_str,
        "source": source_name,
        "file_path": filepath,
        "file_hash": file_hash,
        "entity_count": entity_count,
        "status": "ok",
    })
    result.files_processed += 1
