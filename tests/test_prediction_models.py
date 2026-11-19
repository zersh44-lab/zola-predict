"""Tests for all 6 independent statistical prediction models."""

import pytest

from zola.prediction.dixon_coles import DixonColesModel
from zola.prediction.elo import EloRatingModel
from zola.prediction.ensemble import ConsensusEnsembleModel
from zola.prediction.form_momentum import FormMomentumModel
from zola.prediction.head_to_head import HeadToHeadModel
from zola.prediction.poisson import PoissonModel


@pytest.fixture
def sample_fixture():
    return {
        "id": 101,
        "home_team_id": 1,
        "away_team_id": 2,
        "competition_id": 10,
        "utc_date": "2024-12-01 15:00:00",
    }


@pytest.fixture
def rich_match_history():
    """Synthetic historical matches for testing high-sample predictions."""
    home_matches = [
        {"full_time_home_score": 2, "full_time_away_score": 1, "is_home": 1, "utc_date": "2024-11-20"},
        {"full_time_home_score": 3, "full_time_away_score": 0, "is_home": 1, "utc_date": "2024-11-10"},
        {"full_time_home_score": 1, "full_time_away_score": 1, "is_home": 0, "utc_date": "2024-11-01"},
        {"full_time_home_score": 2, "full_time_away_score": 2, "is_home": 1, "utc_date": "2024-10-25"},
        {"full_time_home_score": 4, "full_time_away_score": 1, "is_home": 1, "utc_date": "2024-10-15"},
        {"full_time_home_score": 1, "full_time_away_score": 0, "is_home": 0, "utc_date": "2024-10-05"},
    ]
    away_matches = [
        {"full_time_home_score": 1, "full_time_away_score": 2, "is_home": 0, "utc_date": "2024-11-22"},
        {"full_time_home_score": 0, "full_time_away_score": 1, "is_home": 0, "utc_date": "2024-11-12"},
        {"full_time_home_score": 2, "full_time_away_score": 2, "is_home": 1, "utc_date": "2024-11-02"},
        {"full_time_home_score": 1, "full_time_away_score": 0, "is_home": 0, "utc_date": "2024-10-22"},
        {"full_time_home_score": 0, "full_time_away_score": 0, "is_home": 1, "utc_date": "2024-10-12"},
        {"full_time_home_score": 1, "full_time_away_score": 3, "is_home": 0, "utc_date": "2024-10-02"},
    ]
    h2h_matches = [
        {"home_team_id": 1, "away_team_id": 2, "full_time_home_score": 2, "full_time_away_score": 1},
        {"home_team_id": 2, "away_team_id": 1, "full_time_home_score": 1, "full_time_away_score": 1},
        {"home_team_id": 1, "away_team_id": 2, "full_time_home_score": 3, "full_time_away_score": 0},
    ]
    league_matches = home_matches + away_matches
    return home_matches, away_matches, h2h_matches, league_matches


def test_poisson_model(sample_fixture, rich_match_history):
    """Test Poisson distribution model probability outputs and markets."""
    h_m, a_m, h2h, lg = rich_match_history
    model = PoissonModel()
    preds = model.predict(sample_fixture, h_m, a_m, h2h, lg)

    # Verify all 6 markets exist
    for market in ("1X2", "BTTS", "OU25", "OU15", "DOUBLE_CHANCE", "DNB"):
        assert market in preds
        p = preds[market]
        assert p.probability >= 0.0
        assert p.probability <= 1.0
        assert p.confidence in ("HIGH", "MEDIUM", "LOW", "UNAVAILABLE")

    # Verify 1X2 sums to ~1.0
    p_1x2 = preds["1X2"].probabilities
    assert sum(p_1x2.values()) == pytest.approx(1.0, abs=0.01)


def test_dixon_coles_model(sample_fixture, rich_match_history):
    """Test Dixon-Coles model with low-score correlation adjustment."""
    h_m, a_m, h2h, lg = rich_match_history
    model = DixonColesModel()
    preds = model.predict(sample_fixture, h_m, a_m, h2h, lg)

    assert "1X2" in preds
    assert "BTTS" in preds
    assert "OU25" in preds
    assert preds["1X2"].features["rho"] == -0.11


def test_elo_model(sample_fixture, rich_match_history):
    """Test Elo power rating model."""
    h_m, a_m, h2h, lg = rich_match_history
    model = EloRatingModel()
    preds = model.predict(sample_fixture, h_m, a_m, h2h, lg)

    assert "1X2" in preds
    features = preds["1X2"].features
    assert "home_elo" in features
    assert "away_elo" in features
    assert features["effective_home_elo"] > features["home_elo"]  # Home advantage verified


def test_form_momentum_model(sample_fixture, rich_match_history):
    """Test Recent Form and Momentum model."""
    h_m, a_m, h2h, lg = rich_match_history
    model = FormMomentumModel()
    preds = model.predict(sample_fixture, h_m, a_m, h2h, lg)

    assert "1X2" in preds
    assert "form_ppg_diff" in preds["1X2"].features
    assert preds["1X2"].features["home_form_sample"] == 6


def test_head_to_head_model_sparse_data(sample_fixture):
    """Test H2H model handles zero past encounters by calibrating confidence gracefully."""
    model = HeadToHeadModel()
    preds = model.predict(sample_fixture, [], [], [], [])

    assert preds["1X2"].confidence in ("MEDIUM", "LOW")
    assert any("direct meetings" in n.lower() or "head-to-head" in n.lower() for n in preds["1X2"].data_quality_notes)


def test_consensus_ensemble_combination(sample_fixture, rich_match_history):
    """Test consensus ensemble dynamically aggregates independent models."""
    h_m, a_m, h2h, lg = rich_match_history
    models = [PoissonModel(), DixonColesModel(), EloRatingModel(), FormMomentumModel(), HeadToHeadModel()]

    all_preds = {m.name: m.predict(sample_fixture, h_m, a_m, h2h, lg) for m in models}

    ensemble = ConsensusEnsembleModel()
    consensus = ensemble.combine(all_preds)

    assert len(consensus) == 8
    assert consensus["1X2"].model_name == "ensemble"
    assert "inter_model_variance" in consensus["1X2"].features
    assert consensus["1X2"].features["active_models_count"] == 5
