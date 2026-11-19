"""Tests for SQLite database schema, constraints, and idempotent repository operations."""

import sqlite3
import pytest

from zola.database.models import (
    ApiCollectionLog,
    Competition,
    Fixture,
    MatchResult,
    ResearchSource,
    Team,
)
from zola.database.repository import FootballRepository


def test_upsert_competition(repo: FootballRepository):
    """Test inserting and updating a competition."""
    comp = Competition(
        external_id="2021",
        provider="football-data.org",
        name="Premier League",
        code="PL",
        country="England",
    )
    comp_id_1 = repo.upsert_competition(comp)
    assert comp_id_1 > 0

    # Upsert same competition with updated season
    comp.current_season = "2024"
    comp_id_2 = repo.upsert_competition(comp)
    assert comp_id_1 == comp_id_2

    stored = repo.get_competition_by_code_or_external_id("football-data.org", "PL")
    assert stored is not None
    assert stored.current_season == "2024"
    assert stored.name == "Premier League"


def test_upsert_team(repo: FootballRepository):
    """Test inserting and updating a team."""
    team = Team(
        external_id="57",
        provider="football-data.org",
        name="Arsenal FC",
        short_name="Arsenal",
        tla="ARS",
    )
    t_id_1 = repo.upsert_team(team)
    assert t_id_1 > 0

    # Update venue
    team.venue = "Emirates Stadium"
    t_id_2 = repo.upsert_team(team)
    assert t_id_1 == t_id_2

    stored = repo.get_team_by_external_id("football-data.org", "57")
    assert stored is not None
    assert stored.venue == "Emirates Stadium"


def test_upsert_fixture_and_deduplication(repo: FootballRepository):
    """Test inserting a fixture, and updating it without causing duplicates."""
    # Setup parent competition and teams
    comp_id = repo.upsert_competition(
        Competition(external_id="2021", provider="football-data.org", name="Premier League", code="PL")
    )
    home_id = repo.upsert_team(
        Team(external_id="57", provider="football-data.org", name="Arsenal FC", short_name="Arsenal")
    )
    away_id = repo.upsert_team(
        Team(external_id="65", provider="football-data.org", name="Manchester City FC", short_name="Man City")
    )

    # 1. First insert: scheduled fixture
    fix = Fixture(
        external_id="9991",
        provider="football-data.org",
        competition_id=comp_id,
        utc_date="2024-11-23T12:30:00Z",
        status="TIMED",
        home_team_id=home_id,
        away_team_id=away_id,
        matchday=12,
    )
    fix_id_1 = repo.upsert_fixture(fix)
    assert fix_id_1 > 0

    # Verify 1 fixture in DB
    fixtures_before = repo.get_fixtures(competition_id=comp_id)
    assert len(fixtures_before) == 1
    assert fixtures_before[0]["status"] == "TIMED"

    # 2. Second upsert: match is now FINISHED with referee and updated status
    fix.status = "FINISHED"
    fix.referee = "Michael Oliver"
    fix_id_2 = repo.upsert_fixture(fix)

    # IDs must match, meaning it was an update, not a new duplicate row
    assert fix_id_1 == fix_id_2

    fixtures_after = repo.get_fixtures(competition_id=comp_id)
    assert len(fixtures_after) == 1
    assert fixtures_after[0]["status"] == "FINISHED"


def test_upsert_match_result(repo: FootballRepository):
    """Test match result scores linked to fixture."""
    comp_id = repo.upsert_competition(
        Competition(external_id="2021", provider="football-data.org", name="Premier League")
    )
    home_id = repo.upsert_team(Team(external_id="1", provider="football-data.org", name="Team A"))
    away_id = repo.upsert_team(Team(external_id="2", provider="football-data.org", name="Team B"))
    fix_id = repo.upsert_fixture(
        Fixture(
            external_id="1001",
            provider="football-data.org",
            competition_id=comp_id,
            utc_date="2024-11-20T15:00:00Z",
            status="FINISHED",
            home_team_id=home_id,
            away_team_id=away_id,
        )
    )

    res = MatchResult(
        fixture_id=fix_id,
        winner="HOME_TEAM",
        duration="REGULAR",
        full_time_home_score=2,
        full_time_away_score=1,
        half_time_home_score=1,
        half_time_away_score=0,
    )
    res_id_1 = repo.upsert_match_result(res)
    assert res_id_1 > 0

    # Update result
    res.full_time_home_score = 3
    res_id_2 = repo.upsert_match_result(res)
    assert res_id_1 == res_id_2

    stored = repo.get_match_result_by_fixture_id(fix_id)
    assert stored is not None
    assert stored.full_time_home_score == 3
    assert stored.full_time_away_score == 1


def test_api_collection_log(repo: FootballRepository):
    """Test logging API calls and retrieving recent logs."""
    log_id = repo.log_api_call(
        ApiCollectionLog(
            provider="football-data.org",
            endpoint="competitions/PL/matches",
            status_code=200,
            response_time_ms=145,
            cache_hit=False,
        )
    )
    assert log_id > 0

    logs = repo.get_recent_api_logs(limit=5)
    assert len(logs) == 1
    assert logs[0].provider == "football-data.org"
    assert logs[0].status_code == 200
    assert logs[0].cache_hit is False


def test_database_stats(repo: FootballRepository):
    """Test database entity counts."""
    stats = repo.get_database_stats()
    assert "competitions" in stats
    assert "teams" in stats
    assert "fixtures" in stats
    assert "match_results" in stats
    assert "research_sources" in stats
    assert "api_collection_logs" in stats


def test_get_fixtures_upcoming_only(repo: FootballRepository):
    """Test that upcoming_only strictly filters out past and finished matches."""
    comp_id = repo.upsert_competition(
        Competition(external_id="PL_TEST", provider="mock", name="Premier League", code="PL")
    )
    t1 = repo.upsert_team(Team(external_id="T1", provider="mock", name="Team A"))
    t2 = repo.upsert_team(Team(external_id="T2", provider="mock", name="Team B"))

    # 1. Past scheduled fixture (August 2023)
    repo.upsert_fixture(
        Fixture(
            external_id="PAST_SCHED",
            provider="mock",
            competition_id=comp_id,
            home_team_id=t1,
            away_team_id=t2,
            utc_date="2023-08-12T15:00:00Z",
            status="SCHEDULED",
        )
    )

    # 2. Genuine future scheduled fixture (2027)
    future_id = repo.upsert_fixture(
        Fixture(
            external_id="FUTURE_SCHED",
            provider="mock",
            competition_id=comp_id,
            home_team_id=t1,
            away_team_id=t2,
            utc_date="2027-12-01T15:00:00Z",
            status="SCHEDULED",
        )
    )

    # 3. Finished fixture
    repo.upsert_fixture(
        Fixture(
            external_id="FINISHED_MATCH",
            provider="mock",
            competition_id=comp_id,
            home_team_id=t1,
            away_team_id=t2,
            utc_date="2027-12-02T15:00:00Z",
            status="FINISHED",
        )
    )

    upcoming = repo.get_fixtures(upcoming_only=True)
    assert len(upcoming) == 1
    assert upcoming[0]["id"] == future_id
    assert upcoming[0]["external_id"] == "FUTURE_SCHED"
