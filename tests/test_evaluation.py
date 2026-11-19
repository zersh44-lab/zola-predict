"""Tests for prediction outcome evaluation and accuracy tracking."""

import sqlite3
import pytest

from zola.database.models import Competition, Fixture, MatchResult, Prediction, Team
from zola.database.repository import FootballRepository


def test_evaluate_predictions_logic(in_memory_db: sqlite3.Connection):
    """Test evaluation logic for 1X2, BTTS, and Over/Under markets."""
    repo = FootballRepository(in_memory_db)
    comp_id = repo.upsert_competition(Competition(external_id="PL", provider="api-football", name="Premier League"))
    t1 = repo.upsert_team(Team(external_id="1", provider="api-football", name="Team A"))
    t2 = repo.upsert_team(Team(external_id="2", provider="api-football", name="Team B"))

    # Finished match: 2-1 (Home win, BTTS yes, Over 2.5)
    f_id = repo.upsert_fixture(
        Fixture(
            external_id="eval_fix_1",
            provider="api-football",
            competition_id=comp_id,
            utc_date="2024-11-01 15:00:00",
            status="FINISHED",
            home_team_id=t1,
            away_team_id=t2,
        )
    )
    repo.upsert_match_result(
        MatchResult(fixture_id=f_id, winner="HOME_TEAM", full_time_home_score=2, full_time_away_score=1)
    )

    # Predictions to evaluate
    p1 = Prediction(fixture_id=f_id, model_name="poisson", market="1X2", predicted_outcome="HOME_WIN", probability=0.60, probabilities_json="{}")
    p2 = Prediction(fixture_id=f_id, model_name="poisson", market="BTTS", predicted_outcome="BTTS_YES", probability=0.55, probabilities_json="{}")
    p3 = Prediction(fixture_id=f_id, model_name="elo", market="OU25", predicted_outcome="UNDER_25", probability=0.52, probabilities_json="{}")  # Incorrect

    repo.upsert_prediction(p1)
    repo.upsert_prediction(p2)
    repo.upsert_prediction(p3)

    stats = repo.evaluate_predictions()
    assert stats["evaluated"] == 3
    assert stats["correct"] == 2
    assert stats["accuracy"] == pytest.approx(0.6667, abs=0.001)

    # Verify accuracy metrics table updated
    metrics = repo.get_accuracy_metrics()
    assert len(metrics) > 0
    poisson_1x2 = [m for m in metrics if m.model_name == "poisson" and m.market == "1X2"][0]
    assert poisson_1x2.accuracy_rate == 1.0
