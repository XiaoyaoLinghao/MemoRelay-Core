"""Importance scorer — 6-factor weighted scoring for knowledge entities."""

from __future__ import annotations

import math
from datetime import date


class ImportanceScorer:
    """Calculates importance score for entities using multiple factors."""

    WEIGHTS: dict[str, float] = {
        "freshness": 0.15,
        "frequency": 0.30,
        "diversity": 0.05,
        "richness": 0.05,
        "access": 0.10,
        "type_base": 0.35,
    }

    TYPE_BASE: dict[str, float] = {
        "decision": 0.80, "risk": 0.60, "project": 0.50,
        "preference": 0.40, "task": 0.15, "tech": 0.05,
        "fact": 0.0, "idea": 0.0,
    }

    def type_base(self, entity_type: str) -> float:
        return self.TYPE_BASE.get(entity_type, 0.0)

    def calculate(
        self,
        days_since_last_seen: int,
        day_count: int,
        distinct_categories: int = 1,
        tag_count: int = 0,
        access_count: int = 0,
        entity_type: str = "fact",
        recent_day_count: int | None = None,
    ) -> float:
        """Calculate composite importance score.

        recent_day_count: days the entity appeared in the last 30 days.
        If not provided, falls back to day_count for backward compatibility.
        """
        freq_input = recent_day_count if recent_day_count is not None else day_count
        score = (
            self.WEIGHTS["freshness"] * self.freshness(days_since_last_seen)
            + self.WEIGHTS["frequency"] * self.frequency(freq_input, days_since_last_seen)
            + self.WEIGHTS["diversity"] * self.diversity(distinct_categories)
            + self.WEIGHTS["richness"] * self.richness(tag_count)
            + self.WEIGHTS["access"] * self.access(access_count)
            + self.WEIGHTS["type_base"] * self.type_base(entity_type)
        )
        return round(max(0.0, score), 4)

    def freshness(self, days_since_last_seen: int) -> float:
        """max(0, 1 - days_since_last_seen / 30)"""
        return max(0.0, 1.0 - days_since_last_seen / 30.0)

    def frequency(self, day_count: int, days_since_last_seen: int = 0) -> float:
        """log(1 + day_count) / log(8), with recency decay for stale entities.

        Entities not seen in >30 days get a decay multiplier — this prevents
        historically hot but now irrelevant entities from permanently dominating.
        """
        if day_count <= 0:
            return 0.0
        base = math.log(1 + day_count) / math.log(8)
        if days_since_last_seen > 30:
            decay = max(0.3, 1.0 - (days_since_last_seen - 30) / 90.0)
            base *= decay
        return base

    def diversity(self, distinct_categories: int) -> float:
        """min(1, distinct_categories / 4)"""
        return min(1.0, distinct_categories / 4.0)

    def richness(self, tag_count: int) -> float:
        """min(1, tag_count / 5)"""
        return min(1.0, tag_count / 5.0)

    def access(self, access_count: int) -> float:
        """min(1, access_count / 5)"""
        return min(1.0, access_count / 5.0)


def score_entity(
    entity: dict,
    access_count: int = 0,
    today: date | None = None,
) -> float:
    """Convenience function to score a single entity dict."""
    if today is None:
        today = date.today()

    last_seen_str = entity.get("last_seen")
    if last_seen_str:
        last_seen = date.fromisoformat(last_seen_str)
        days_since = (today - last_seen).days
    else:
        days_since = 999

    day_count = entity.get("day_count", 0)
    distinct_categories = entity.get("distinct_categories", 0)
    tags = entity.get("tags", [])
    tag_count = len(tags) if isinstance(tags, (list, tuple)) else 0

    scorer = ImportanceScorer()
    return scorer.calculate(
        days_since_last_seen=max(0, days_since),
        day_count=day_count,
        distinct_categories=distinct_categories,
        tag_count=tag_count,
        access_count=access_count,
        entity_type=entity.get("type", "fact"),
    )


def filter_by_score(
    entities: list[dict],
    min_score: float = 0.0,
    access_counts: dict[str, int] | None = None,
    today: date | None = None,
) -> list[dict]:
    """Filter and sort entities by importance score, return sorted list."""
    if access_counts is None:
        access_counts = {}

    scored: list[tuple[float, dict]] = []
    for entity in entities:
        entity_id = entity.get("id", "")
        ac = access_counts.get(entity_id, 0)
        s = score_entity(entity, access_count=ac, today=today)
        if s >= min_score:
            scored.append((s, entity))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entity for _, entity in scored]
