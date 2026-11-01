"""Tests for the scorer module — importance scoring with 6-factor formula."""

import math
from datetime import date

from knowledge_weaver.scorer import ImportanceScorer, score_entity, filter_by_score


# ---------------------------------------------------------------------------
# Factor-level tests
# ---------------------------------------------------------------------------

class TestFreshness:
    def test_freshness_recent(self):
        """0 days since last seen → freshness = 1.0"""
        scorer = ImportanceScorer()
        assert scorer.freshness(0) == 1.0

    def test_freshness_old(self):
        """60 days since last seen → freshness = 0.0"""
        scorer = ImportanceScorer()
        assert scorer.freshness(60) == 0.0

    def test_freshness_partial(self):
        """15 days since last seen → freshness = 0.5"""
        scorer = ImportanceScorer()
        assert scorer.freshness(15) == 0.5


class TestFrequency:
    def test_frequency_high(self):
        """14 days → log(15)/log(8) ≈ 1.302, not capped."""
        scorer = ImportanceScorer()
        expected = math.log(15) / math.log(8)
        assert abs(scorer.frequency(14) - expected) < 0.01

    def test_frequency_low(self):
        """1 day → log(2)/log(8) ≈ 0.333"""
        scorer = ImportanceScorer()
        expected = math.log(2) / math.log(8)
        assert abs(scorer.frequency(1) - expected) < 0.01


class TestDiversity:
    def test_diversity(self):
        scorer = ImportanceScorer()
        assert scorer.diversity(0) == 0.0
        assert scorer.diversity(2) == 0.5
        assert scorer.diversity(4) == 1.0
        assert scorer.diversity(8) == 1.0


class TestRichness:
    def test_richness(self):
        scorer = ImportanceScorer()
        assert scorer.richness(0) == 0.0
        assert scorer.richness(3) == 0.6
        assert scorer.richness(5) == 1.0


class TestAccess:
    def test_access_count(self):
        scorer = ImportanceScorer()
        assert scorer.access(0) == 0.0
        assert scorer.access(3) == 0.6
        assert scorer.access(5) == 1.0


class TestTypeBase:
    def test_type_base_known_types(self):
        scorer = ImportanceScorer()
        assert scorer.type_base("decision") == 0.80
        assert scorer.type_base("risk") == 0.60
        assert scorer.type_base("project") == 0.50
        assert scorer.type_base("preference") == 0.40
        assert scorer.type_base("task") == 0.15
        assert scorer.type_base("tech") == 0.05
        assert scorer.type_base("fact") == 0.0
        assert scorer.type_base("idea") == 0.0

    def test_type_base_unknown_type(self):
        scorer = ImportanceScorer()
        assert scorer.type_base("gobbledygook") == 0.0


# ---------------------------------------------------------------------------
# Composite score tests
# ---------------------------------------------------------------------------

class TestCalculateImportance:
    def test_calculate_importance_full(self):
        """All factors maxed (day_count=14) with new weights."""
        scorer = ImportanceScorer()
        score = scorer.calculate(
            days_since_last_seen=0,
            day_count=14,
            distinct_categories=8,
            tag_count=10,
            access_count=10,
        )
        # freshness=1.0, frequency=log(15)/log(8)≈1.302, diversity=1.0,
        # richness=1.0, access=1.0, type_base("fact")=0.0
        # = 0.15*1.0 + 0.30*1.302 + 0.05*1.0 + 0.05*1.0 + 0.10*1.0 + 0.35*0.0 ≈ 0.7407
        assert abs(score - 0.7407) < 0.01

    def test_calculate_importance_zero(self):
        """All factors at 0 → importance = 0.0"""
        scorer = ImportanceScorer()
        score = scorer.calculate(
            days_since_last_seen=60,
            day_count=0,
            distinct_categories=0,
            tag_count=0,
            access_count=0,
        )
        assert score == 0.0

    def test_score_bounded(self):
        """Score always >= 0.0 (no upper ceiling)."""
        scorer = ImportanceScorer()
        for args in [
            (0, 1, 1, 1, 0),
            (0, 14, 8, 10, 20),
            (60, 0, 0, 0, 0),
            (15, 3, 2, 1, 4),
        ]:
            score = scorer.calculate(*args)
            assert score >= 0.0, f"score {score} negative for {args}"

    def test_score_can_exceed_one(self):
        """High day_count + decision type → score > 1.0 (ceiling removed)."""
        scorer = ImportanceScorer()
        score = scorer.calculate(
            days_since_last_seen=0,
            day_count=72,
            distinct_categories=8,
            tag_count=10,
            access_count=10,
            entity_type="decision",
        )
        assert score > 1.0, f"Expected > 1.0 (no ceiling), got {score}"

    def test_entity_type_param(self):
        """entity_type='fact' (default) vs 'decision' changes score."""
        scorer = ImportanceScorer()
        base = scorer.calculate(0, 7, 4, 5, 5)  # default entity_type="fact"
        decision = scorer.calculate(0, 7, 4, 5, 5, entity_type="decision")
        # decision should be higher by 0.35 * (0.80 - 0.0) = 0.28
        assert abs(decision - base - 0.28) < 0.01


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestScoreEntity:
    def test_score_entity_basic(self):
        entity = {
            "last_seen": "2026-05-25",
            "day_count": 7,
            "distinct_categories": 4,
            "tags": ["a", "b", "c", "d", "e"],
        }
        score = score_entity(entity, access_count=5, today=date(2026, 5, 25))
        # freshness=1.0, frequency(7)≈1.0, diversity=1.0, richness=1.0,
        # access=1.0, type_base("fact")=0.0
        # = 0.15+0.30+0.05+0.05+0.10+0.0 = 0.65
        assert score == 0.65

    def test_score_entity_stale(self):
        entity = {
            "last_seen": "2026-04-25",
            "day_count": 1,
            "distinct_categories": 1,
            "tags": [],
        }
        score = score_entity(entity, access_count=0, today=date(2026, 5, 25))
        assert 0.0 <= score < 0.5

    def test_score_entity_with_type(self):
        """Entity with type='decision' scores higher than default 'fact'."""
        entity = {
            "last_seen": "2026-05-25",
            "day_count": 7,
            "distinct_categories": 4,
            "tags": ["a", "b", "c", "d", "e"],
            "type": "decision",
        }
        score = score_entity(entity, access_count=5, today=date(2026, 5, 25))
        # Same as test_score_entity_basic but type_base("decision")=0.80
        # = 0.65 + 0.35*0.80 = 0.93
        assert score == 0.93


# ---------------------------------------------------------------------------
# Filter and sort tests
# ---------------------------------------------------------------------------

class TestFilterByScore:
    def test_scoring_filter(self):
        """filter_by_score removes entities below min_score."""
        entities = [
            {"id": "e1", "last_seen": "2026-05-25", "day_count": 7, "distinct_categories": 4, "tags": list(range(5))},
            {"id": "e2", "last_seen": "2026-04-25", "day_count": 1, "distinct_categories": 1, "tags": []},
        ]
        today = date(2026, 5, 25)
        result = filter_by_score(entities, min_score=0.5, today=today)
        ids = [e["id"] for e in result]
        assert "e1" in ids
        assert "e2" not in ids

    def test_score_and_sort(self):
        """Multiple entities returned in descending importance order."""
        entities = [
            {"id": "low", "last_seen": "2026-05-10", "day_count": 1, "distinct_categories": 1, "tags": []},
            {"id": "high", "last_seen": "2026-05-25", "day_count": 7, "distinct_categories": 4, "tags": list(range(5))},
            {"id": "mid", "last_seen": "2026-05-20", "day_count": 3, "distinct_categories": 2, "tags": ["a"]},
        ]
        today = date(2026, 5, 25)
        result = filter_by_score(entities, min_score=0.0, today=today)
        ids = [e["id"] for e in result]
        assert ids == ["high", "mid", "low"]


# ---------------------------------------------------------------------------
# Ceiling break tests (new)
# ---------------------------------------------------------------------------

class TestCeilingBreak:
    def test_no_ceiling_high_day_count(self):
        """High day_count + decision type → score > 0.85 (above old ceiling)."""
        scorer = ImportanceScorer()
        score = scorer.calculate(
            days_since_last_seen=0,
            day_count=72,
            distinct_categories=8,
            tag_count=10,
            access_count=10,
            entity_type="decision",
        )
        assert score > 0.85, f"Expected > 0.85, got {score}"

    def test_type_base_differentiation(self):
        """decision type should score significantly higher than tech type."""
        scorer = ImportanceScorer()
        base_args = dict(days_since_last_seen=0, day_count=7, distinct_categories=4, tag_count=5, access_count=5)
        decision_score = scorer.calculate(**base_args, entity_type="decision")
        tech_score = scorer.calculate(**base_args, entity_type="tech")
        assert decision_score > tech_score + 0.05, f"decision={decision_score}, tech={tech_score}"

    def test_tech_common_word_deprioritized(self):
        """tech type with high day_count should score lower than decision type with same inputs."""
        scorer = ImportanceScorer()
        score = scorer.calculate(
            days_since_last_seen=0,
            day_count=72,
            distinct_categories=1,
            tag_count=0,
            access_count=0,
            entity_type="tech",
        )
        # With log frequency, high day_count still yields a non-trivial score,
        # but tech type_base (0.05) is much lower than decision (0.40).
        decision_score = scorer.calculate(
            days_since_last_seen=0,
            day_count=72,
            distinct_categories=1,
            tag_count=0,
            access_count=0,
            entity_type="decision",
        )
        assert score < decision_score, f"tech={score}, decision={decision_score}"

    def test_frequency_logarithmic(self):
        """Frequency uses logarithmic formula."""
        scorer = ImportanceScorer()
        # day_count=1 → log(2)/log(8) ≈ 0.333
        f1 = scorer.frequency(1)
        assert abs(f1 - math.log(2) / math.log(8)) < 0.01
        # day_count=7 → log(8)/log(8) = 1.0
        f7 = scorer.frequency(7)
        assert abs(f7 - 1.0) < 0.01
        # day_count=30 → log(31)/log(8) ≈ 1.64
        f30 = scorer.frequency(30)
        assert abs(f30 - math.log(31) / math.log(8)) < 0.01
