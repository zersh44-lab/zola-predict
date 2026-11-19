"""Tests for the end-to-end PredictionEngine with database integration."""

import sqlite3
import pytest

from zola.database.models import Competition, Fixture, MatchResult, Team
from zola.database.repository import FootballRepository
from zola.prediction.engine import PredictionEngine


@pytest.fixture
def populated_repo(in_memory_db: sqlite3.Connection) -> FootballRepository:
    """Repository pre-populated with a league and 10 finished matches."""
    repo = FootballRepository(in_memory_db)
    comp_id = repo.upsert_competition(
        Competition(external_id="PL", provider="api-football", name="Premier League", code="PL")
    )
    t1_id = repo.upsert_team(Team(external_id="1", provider="api-football", name="Arsenal", short_name="Arsenal"))
    t2_id = repo.upsert_team(Team(external_id="2", provider="api-football", name="Chelsea", short_name="Chelsea"))
    t3_id = repo.upsert_team(Team(external_id="3", provider="api-football", name="Liverpool", short_name="Liverpool"))

    # Finished historical matches
    for i in range(1, 6):
        f_id = repo.upsert_fixture(
            Fixture(
                external_id=f"hist_fix_{i}",
                provider="api-football",
                competition_id=comp_id,
                utc_date=f"2024-10-0{i} 15:00:00",
                status="FINISHED",
                home_team_id=t1_id,
                away_team_id=t3_id,
            )
        )
        repo.upsert_match_result(
            MatchResult(fixture_id=f_id, winner="HOME_TEAM", full_time_home_score=2, full_time_away_score=1)
        )

    for i in range(6, 11):
        f_id = repo.upsert_fixture(
            Fixture(
                external_id=f"hist_fix_{i}",
                provider="api-football",
                competition_id=comp_id,
                utc_date=f"2024-10-1{i-5} 15:00:00",
                status="FINISHED",
                home_team_id=t3_id,
                away_team_id=t2_id,
            )
        )
        repo.upsert_match_result(
            MatchResult(fixture_id=f_id, winner="AWAY_TEAM", full_time_home_score=0, full_time_away_score=2)
        )

    return repo


def test_predict_fixture_pipeline(populated_repo: FootballRepository):
    """Test full prediction engine pipeline across all 6 models and database persistence."""
    comp = populated_repo.get_competition_by_code_or_external_id("api-football", "PL")
    t1 = populated_repo.get_team_by_name("Arsenal")
    t2 = populated_repo.get_team_by_name("Chelsea")

    # Upcoming fixture
    fixture_id = populated_repo.upsert_fixture(
        Fixture(
            external_id="upcoming_derby_101",
            provider="api-football",
            competition_id=comp.id,
            utc_date="2024-12-15 17:30:00",
            status="SCHEDULED",
            home_team_id=t1.id,
            away_team_id=t2.id,
        )
    )

    engine = PredictionEngine(populated_repo)
    results = engine.predict_fixture(fixture_id, persist=True)

    # Verify all 6 models present
    expected_models = {"poisson", "dixon_coles", "elo", "form_momentum", "head_to_head", "ensemble"}
    assert set(results.keys()) == expected_models

    # Verify predictions saved to database
    db_preds = populated_repo.get_predictions_for_fixture(fixture_id)
    # 6 models * 8 markets = 48 persisted predictions
    assert len(db_preds) == 48

    ensemble_1x2 = [p for p in db_preds if p.model_name == "ensemble" and p.market == "1X2"][0]
    assert ensemble_1x2.predicted_outcome in ("HOME_WIN", "DRAW", "AWAY_WIN")
    assert ensemble_1x2.status == "PENDING"
