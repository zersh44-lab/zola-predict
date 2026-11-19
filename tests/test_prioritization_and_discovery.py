"""
Unit and integration tests for BeSoccer-Inspired Match Discovery & Prioritization System.
"""
import pytest
from zola.database.connection import get_connection, SCHEMA_SQL
from zola.database.repository import FootballRepository
from zola.services.prioritization import (
    MatchPrioritizationEngine,
    PrioritizationWeights,
)


@pytest.fixture
def memory_repo():
    """Create in-memory SQLite repository with initialized schema."""
    conn = get_connection(":memory:")
    conn.executescript(SCHEMA_SQL)
    repo = FootballRepository(conn)
    return repo


class TestPrioritizationEngine:
    """Test suite for the 0-100 composite prioritization scoring engine."""

    def test_default_weights_sum_to_one(self):
        w = PrioritizationWeights()
        total = (
            w.competition_importance
            + w.competition_popularity
            + w.team_popularity
            + w.match_relevance
            + w.kickoff_proximity
            + w.data_completeness
        )
        assert round(total, 2) == 1.0
        assert w.competition_importance == 0.30
        assert w.competition_popularity == 0.20
        assert w.team_popularity == 0.20
        assert w.match_relevance == 0.15
        assert w.kickoff_proximity == 0.10
        assert w.data_completeness == 0.05

    def test_weight_normalization(self):
        w = PrioritizationWeights(
            competition_importance=30,
            competition_popularity=20,
            team_popularity=20,
            match_relevance=15,
            kickoff_proximity=10,
            data_completeness=5,
        ).normalize()
        total = (
            w.competition_importance
            + w.competition_popularity
            + w.team_popularity
            + w.match_relevance
            + w.kickoff_proximity
            + w.data_completeness
        )
        assert round(total, 2) == 1.0
        assert round(w.competition_importance, 2) == 0.30

    def test_marquee_matchup_high_priority(self):
        engine = MatchPrioritizationEngine()
        marquee_match = {
            "competition_name": "UEFA Champions League",
            "competition_code": "CL",
            "competition_country": "Europe",
            "home_team_name": "Real Madrid",
            "away_team_name": "Manchester City",
            "status": "SCHEDULED",
            "utc_date": "2026-11-17T20:00:00Z",
            "competition_logo": "https://media.api-sports.io/football/leagues/2.png",
            "home_team_logo": "https://media.api-sports.io/football/teams/541.png",
            "away_team_logo": "https://media.api-sports.io/football/teams/50.png",
            "round": "Semi-finals",
        }
        score, category, breakdown = engine.compute_priority_score(marquee_match)
        assert score >= 80.0
        assert category == "FEATURED"
        assert breakdown["competition_importance"] >= 95.0
        assert breakdown["team_popularity"] >= 90.0

    def test_lower_tier_match_low_priority(self):
        engine = MatchPrioritizationEngine()
        low_match = {
            "competition_name": "Tercera Division",
            "competition_code": "TD",
            "competition_country": "Spain",
            "home_team_name": "CD El Palo",
            "away_team_name": "UD San Pedro",
            "status": "SCHEDULED",
            "utc_date": "2026-11-25T11:00:00Z",
        }
        score, category, breakdown = engine.compute_priority_score(low_match)
        assert score < 40.0
        assert category in ("LOW", "NORMAL")


class TestFavoritesAndConfigRepository:
    """Test repository methods for favorites and scoring weights."""

    def test_user_favorites_toggle(self, memory_repo):
        # Initial is empty
        favs = memory_repo.get_user_favorites()
        assert len(favs["teams"]) == 0
        assert len(favs["competitions"]) == 0

        # Toggle Team ON
        added = memory_repo.toggle_user_favorite("team", "10", "Arsenal")
        assert added is True
        favs = memory_repo.get_user_favorites()
        assert len(favs["teams"]) == 1
        assert favs["teams"][0]["id"] == "10"
        assert favs["teams"][0]["name"] == "Arsenal"

        # Toggle Competition ON
        added_comp = memory_repo.toggle_user_favorite("competition", "39", "Premier League")
        assert added_comp is True
        favs = memory_repo.get_user_favorites()
        assert len(favs["competitions"]) == 1
        assert favs["competitions"][0]["id"] == "39"

        # Toggle Team OFF
        removed = memory_repo.toggle_user_favorite("team", "10", "Arsenal")
        assert removed is False
        favs = memory_repo.get_user_favorites()
        assert len(favs["teams"]) == 0
        assert len(favs["competitions"]) == 1

    def test_save_and_retrieve_custom_weights(self, memory_repo):
        custom = {
            "competition_importance": 0.40,
            "competition_popularity": 0.15,
            "team_popularity": 0.15,
            "match_relevance": 0.15,
            "kickoff_proximity": 0.10,
            "data_completeness": 0.05,
        }
        saved = memory_repo.save_prioritization_weights(custom)
        assert round(saved.competition_importance, 2) == 0.40

        retrieved = memory_repo.get_prioritization_weights()
        assert round(retrieved.competition_importance, 2) == 0.40
        assert round(retrieved.competition_popularity, 2) == 0.15

    def test_discovery_filtering_by_competition_name_and_featured_isolation(self, memory_repo):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Create two distinct competitions
        cursor = memory_repo.conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO competitions (id, external_id, provider, name, country, code) VALUES (837, 'afcon', 'p', 'Africa Cup of Nations - Qualification', 'World', 'AFCON')")
        cursor.execute("INSERT OR REPLACE INTO competitions (id, external_id, provider, name, country, code) VALUES (404, 'unl', 'p', 'UEFA Nations League', 'World', 'UNL')")
        cursor.execute("INSERT OR REPLACE INTO teams (id, external_id, provider, name) VALUES (1, 't1', 'p', 'Morocco'), (2, 't2', 'p', 'Niger'), (3, 't3', 'p', 'Portugal'), (4, 't4', 'p', 'Poland')")
        cursor.execute(f"INSERT OR REPLACE INTO fixtures (id, external_id, provider, utc_date, status, competition_id, home_team_id, away_team_id) VALUES (101, 'm1', 'p', '{today}T18:00:00Z', 'SCHEDULED', 837, 1, 2)")
        cursor.execute(f"INSERT OR REPLACE INTO fixtures (id, external_id, provider, utc_date, status, competition_id, home_team_id, away_team_id) VALUES (102, 'm2', 'p', '{today}T19:00:00Z', 'SCHEDULED', 404, 3, 4)")
        memory_repo.conn.commit()

        # Query filtered by AFCON
        res = memory_repo.get_discovery_fixtures(date=today, competition_name="Africa Cup of Nations - Qualification")
        assert res["total_matches"] == 1
        assert res["competitions"][0]["name"] == "Africa Cup of Nations - Qualification"
        assert res["competitions"][0]["competition_name"] == "Africa Cup of Nations - Qualification"
        # Ensure featured matches strictly contain only AFCON fixtures, not UEFA Nations League
        for fm in res["featured_matches"]:
            assert fm["competition_name"] == "Africa Cup of Nations - Qualification"
