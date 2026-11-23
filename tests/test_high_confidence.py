"""Unit tests for the High-Confidence Prediction Selection System."""

import pytest
from zola.prediction.base import ModelPrediction
from zola.prediction.high_confidence import HighConfidenceSelector, MatchHighConfidenceReport


@pytest.fixture
def sample_fixture():
    return {
        "id": 1001,
        "home_team_name": "Arsenal",
        "away_team_name": "Crystal Palace",
        "competition_name": "Premier League",
        "utc_date": "2026-11-21T15:00:00+00:00",
    }


def make_pred(market: str, outcome: str, prob: float, conf: str = "HIGH", conf_score: float = 0.85):
    dist = {outcome: prob}
    if market == "1X2":
        rem = round(1.0 - prob, 4)
        dist["DRAW"] = round(rem * 0.6, 4)
        dist["AWAY_WIN"] = round(rem * 0.4, 4)
    else:
        dist["OTHER"] = round(1.0 - prob, 4)

    return ModelPrediction(
        model_name="ensemble",
        market=market,
        predicted_outcome=outcome,
        probability=prob,
        probabilities=dist,
        confidence=conf,
        confidence_score=conf_score,
    )


def test_high_confidence_selector_returns_max_3_picks(sample_fixture):
    """Test that selector never returns more than 3 picks even if multiple markets qualify."""
    preds = {
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.82),
        "OU15": make_pred("OU15", "OVER_15", 0.78),
        "OU05": make_pred("OU05", "OVER_05", 0.94),
        "DNB": make_pred("DNB", "1_DNB", 0.75),
        "1X2": make_pred("1X2", "HOME_WIN", 0.66),
        "BTTS": make_pred("BTTS", "BTTS_YES", 0.66),
        "OU25": make_pred("OU25", "OVER_25", 0.67),
    }
    stats = {"home_xg": 2.1, "away_xg": 0.8, "home_goals_scored": 2.2, "away_goals_scored": 1.2}
    odds = {
        "OU15": {"OVER_15": 1.45},
        "OU05": {"OVER_05": 1.15},
    }

    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        comparison_stats=stats,
        home_form=["W", "W", "D", "W", "W"],
        away_form=["L", "D", "L", "W", "L"],
        odds_data=odds,
    )

    assert isinstance(report, MatchHighConfidenceReport)
    assert not report.is_skipped
    assert 1 <= report.total_picks <= 3
    assert len(report.picks) <= 3
    assert report.match_name == "Arsenal vs Crystal Palace"
    assert report.competition_name == "Premier League"

    # Verify each pick adheres strictly to active allowlist
    for pick in report.picks:
        assert pick.market in ("OU15", "OU05")
        assert pick.probability >= 0.60
        assert pick.confidence in ("HIGH", "MEDIUM")
        assert len(pick.rationale) > 20
        assert len(pick.risks) > 10
        assert pick.estimated_odd >= 1.0


def test_high_confidence_selector_strictly_excludes_disabled_markets(sample_fixture):
    """Test that disabled markets (1X2, BTTS, OU25, OU35, DNB) are never selected even with high probabilities."""
    preds = {
        "1X2": make_pred("1X2", "HOME_WIN", 0.90, conf="HIGH", conf_score=0.95),
        "BTTS": make_pred("BTTS", "BTTS_YES", 0.90, conf="HIGH", conf_score=0.95),
        "OU25": make_pred("OU25", "OVER_25", 0.90, conf="HIGH", conf_score=0.95),
        "OU35": make_pred("OU35", "UNDER_35", 0.90, conf="HIGH", conf_score=0.95),
        "DNB": make_pred("DNB", "1_DNB", 0.90, conf="HIGH", conf_score=0.95),
        "OU15": make_pred("OU15", "OVER_15", 0.82, conf="HIGH", conf_score=0.88),
    }
    odds = {"OU15": {"OVER_15": 1.35}}

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds, odds_data=odds)

    assert not report.is_skipped
    selected_markets = {p.market for p in report.picks}
    assert selected_markets.issubset({"OU05", "OU15"})
    assert not any(m in selected_markets for m in ("1X2", "BTTS", "OU25", "OU35", "DNB", "DOUBLE_CHANCE"))


def test_high_confidence_selector_blocks_negative_roi_double_chance_subtypes(sample_fixture):
    """Test that Double Chance subtypes (1X, X2, 12) are blocked under strict market policy."""
    for subtype in ("1X", "X2", "12"):
        preds = {
            "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", subtype, 0.88, conf="HIGH", conf_score=0.92),
            "1X2": make_pred("1X2", "HOME_WIN", 0.50, conf="LOW", conf_score=0.30),
            "OU15": make_pred("OU15", "OVER_15", 0.50, conf="LOW", conf_score=0.30),
        }
        report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds)
        assert report.is_skipped is True
        assert "NO HIGH-CONFIDENCE PREDICTION" in report.skip_reason
        assert any("Double Chance" in r for r in report.analysis_summary.get("skip_reasons", []))


def test_high_confidence_selector_skips_low_confidence_match(sample_fixture):
    """Test that when markets fail the confidence threshold, a SKIP report is returned."""
    # Narrow toss-up match with low conviction
    preds = {
        "1X2": make_pred("1X2", "HOME_WIN", 0.42, conf="LOW", conf_score=0.35),
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.62, conf="LOW", conf_score=0.40),
        "OU15": make_pred("OU15", "OVER_15", 0.58, conf="LOW", conf_score=0.38),
        "OU25": make_pred("OU25", "UNDER_25", 0.52, conf="NO_BET", conf_score=0.30),
        "BTTS": make_pred("BTTS", "BTTS_NO", 0.51, conf="LOW", conf_score=0.32),
    }
    stats = {"home_xg": 1.1, "away_xg": 1.0}

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds, comparison_stats=stats)

    assert report.is_skipped is True
    assert report.skip_reason == "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH."
    assert "SKIP THIS MATCH" in report.skip_rationale
    assert report.total_picks == 0
    assert len(report.picks) == 0


def test_high_confidence_selector_allows_1_or_2_picks_without_forcing_3(sample_fixture):
    """Test that system does not force 3 picks if only 1 is reliable."""
    # Only Over 1.5 Goals qualifies; other markets are weak or coin-flips
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.78, conf="HIGH", conf_score=0.88),
        "OU25": make_pred("OU25", "UNDER_25", 0.52, conf="LOW", conf_score=0.30),
        "1X2": make_pred("1X2", "HOME_WIN", 0.51, conf="LOW", conf_score=0.38),
    }
    odds = {"OU15": {"OVER_15": 1.35}}

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds, odds_data=odds)

    assert not report.is_skipped
    assert report.total_picks == 1
    assert len(report.picks) == 1
    assert report.picks[0].market == "OU15"
    assert report.picks[0].predicted_outcome == "OVER_15"


def test_high_confidence_selector_deduplication(sample_fixture):
    """Test that redundant picks (Over 1.5 + Over 0.5) are deduplicated, keeping Over 1.5."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.79),
        "OU05": make_pred("OU05", "OVER_05", 0.95),
    }
    odds = {
        "OU15": {"OVER_15": 1.40},
        "OU05": {"OVER_05": 1.12},
    }

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds, odds_data=odds)

    assert not report.is_skipped
    markets = [p.market for p in report.picks]
    # OU15 is preferred; OU05 is deduplicated
    assert "OU15" in markets
    assert "OU05" not in markets


def test_high_confidence_selector_rejects_missing_or_invalid_odds_when_required(sample_fixture):
    """Test that when require_valid_odds=True, missing or non-positive EV odds reject the candidate."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.78, conf="HIGH", conf_score=0.88),
    }

    # 1. No odds provided -> rejected
    report_no_odds = HighConfidenceSelector.evaluate_fixture(
        sample_fixture, preds, require_valid_odds=True
    )
    assert report_no_odds.is_skipped is True
    assert "odds" in report_no_odds.skip_rationale.lower()

    # 2. Invalid odd (<= 1.0) -> rejected
    report_invalid_odds = HighConfidenceSelector.evaluate_fixture(
        sample_fixture, preds, odds_data={"OU15": {"OVER_15": 0.95}}, require_valid_odds=True
    )
    assert report_invalid_odds.is_skipped is True

    # 3. Negative EV odd (e.g. prob 0.70 at odd 1.20 -> EV = -16.0%) -> rejected
    preds_low_ev = {"OU15": make_pred("OU15", "OVER_15", 0.70, conf="HIGH", conf_score=0.80)}
    report_neg_ev = HighConfidenceSelector.evaluate_fixture(
        sample_fixture, preds_low_ev, odds_data={"OU15": {"OVER_15": 1.20}}, require_valid_odds=True
    )
    assert report_neg_ev.is_skipped is True


# =========================================================================
# REQUIRED CRITICAL SAFETY AUDIT TEST SCENARIOS
# =========================================================================

def test_scenario_1_ou15_verified_pre_match_odds_positive_ev_eligible(sample_fixture):
    """Scenario 1: OU15 with verified pre-match odds and positive EV: eligible for selection."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.80, conf="HIGH", conf_score=0.90),
    }
    # Prob 0.80 at odd 1.35 -> EV = (0.80 * 1.35) - 1.0 = +8.0%
    odds = {"OU15": {"OVER_15": 1.35, "is_synthetic": False}}
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data=odds,
        require_valid_odds=True,
    )
    assert not report.is_skipped
    assert report.total_picks == 1
    pick = report.picks[0]
    assert pick.market == "OU15"
    assert pick.predicted_outcome == "OVER_15"
    assert pick.market_odd == 1.35
    assert pick.expected_value == 0.08
    assert pick.betting_value_status == "VERIFIED_POSITIVE_EV"
    assert pick.is_verified_odds is True


def test_scenario_2_ou05_missing_odds_skipped(sample_fixture):
    """Scenario 2: OU05 with missing odds: skipped."""
    preds = {
        "OU05": make_pred("OU05", "OVER_05", 0.95, conf="HIGH", conf_score=0.92),
    }
    # Missing odds entirely
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data=None,
        require_valid_odds=True,
    )
    assert report.is_skipped is True
    assert report.skip_reason == "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH."
    assert any("Missing pre-match bookmaker odds" in r for r in report.analysis_summary["skip_reasons"])


def test_scenario_3_ou15_synthetic_odds_skipped(sample_fixture):
    """Scenario 3: OU15 with synthetic odds: skipped."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.82, conf="HIGH", conf_score=0.90),
    }
    # Synthetic / estimated odds payload
    synthetic_odds = {
        "OU15": {"OVER_15": 1.40, "is_synthetic": True, "source": "synthetic"}
    }
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data=synthetic_odds,
        require_valid_odds=True,
    )
    assert report.is_skipped is True
    assert report.skip_reason == "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH."
    assert any("Synthetic" in r for r in report.analysis_summary["skip_reasons"])


def test_scenario_4_ou15_post_kickoff_odds_skipped(sample_fixture):
    """Scenario 4: OU15 with odds captured after kickoff: skipped."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.82, conf="HIGH", conf_score=0.90),
    }
    # 4a. Explicit post_kickoff flag
    post_ko_odds = {
        "OU15": {"OVER_15": 1.40, "is_post_kickoff": True}
    }
    report1 = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data=post_ko_odds,
        require_valid_odds=True,
    )
    assert report1.is_skipped is True
    assert any("after kickoff" in r for r in report1.analysis_summary["skip_reasons"])

    # 4b. Fixture already in-play
    live_fixture = dict(sample_fixture)
    live_fixture["status"] = "IN_PLAY"
    report2 = HighConfidenceSelector.evaluate_fixture(
        live_fixture,
        preds,
        odds_data={"OU15": {"OVER_15": 1.40}},
        require_valid_odds=True,
    )
    assert report2.is_skipped is True
    assert any("in-play" in r for r in report2.analysis_summary["skip_reasons"])


def test_scenario_5_double_chance_valid_odds_still_blocked(sample_fixture):
    """Scenario 5: Double Chance with valid odds: still blocked."""
    preds = {
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.85, conf="HIGH", conf_score=0.90),
    }
    # Even with generous valid pre-match odds that would offer massive positive EV:
    odds = {"DOUBLE_CHANCE": {"1X": 1.50}}
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data=odds,
        require_valid_odds=True,
    )
    assert report.is_skipped is True
    assert report.skip_reason == "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH."
    assert any("Double Chance" in r and "blocked" in r for r in report.analysis_summary["skip_reasons"])


def test_scenario_6_any_disabled_market_blocked(sample_fixture):
    """Scenario 6: Any disabled market (1X2, BTTS, OU25, OU35, DNB): blocked."""
    disabled_markets = ["1X2", "BTTS", "OU25", "OU35", "DNB"]
    for m in disabled_markets:
        preds = {
            m: make_pred(m, "OUTCOME_X", 0.95, conf="HIGH", conf_score=0.98),
        }
        odds = {m: {"OUTCOME_X": 2.00}}
        report = HighConfidenceSelector.evaluate_fixture(
            sample_fixture,
            preds,
            odds_data=odds,
            require_valid_odds=True,
        )
        assert report.is_skipped is True
        assert report.total_picks == 0


def test_scenario_7_no_eligible_candidates_exact_skip_message(sample_fixture):
    """Scenario 7: No eligible candidates: return the exact skip message."""
    preds = {
        "1X2": make_pred("1X2", "HOME_WIN", 0.40, conf="LOW", conf_score=0.30),
    }
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        require_valid_odds=True,
    )
    assert report.is_skipped is True
    assert report.skip_reason == "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH."
    assert report.total_picks == 0
    assert report.picks == []


def test_scenario_8_maximum_3_picks_enforced(sample_fixture):
    """Scenario 8: Maximum 3 picks per match remains enforced."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.85, conf="HIGH", conf_score=0.90),
        "OU05": make_pred("OU05", "OVER_05", 0.96, conf="HIGH", conf_score=0.95),
    }
    odds = {
        "OU15": {"OVER_15": 1.40},
        "OU05": {"OVER_05": 1.15},
    }
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data=odds,
        require_valid_odds=True,
    )
    assert not report.is_skipped
    assert len(report.picks) <= 3


def test_high_confidence_selector_accepts_positive_ev_with_valid_odds(sample_fixture):
    """Test that valid pre-match odds with positive EV are approved."""
    preds = {
        "OU15": make_pred("OU15", "OVER_15", 0.80, conf="HIGH", conf_score=0.90),
    }
    # Prob 0.80 at odd 1.35 -> EV = (0.80 * 1.35) - 1.0 = +8.0%
    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        odds_data={"OU15": {"OVER_15": 1.35}},
        require_valid_odds=True,
        min_ev_margin=0.02,
    )

    assert not report.is_skipped
    assert report.total_picks == 1
    pick = report.picks[0]
    assert pick.market == "OU15"
    assert pick.market_odd == 1.35
    assert pick.estimated_odd == 1.35
    assert pick.expected_value == 0.08


def test_expected_value_calculations():
    """Verify exact Expected Value formula: EV = (prob * odd) - 1.0."""
    from zola.prediction.calibration import compute_expected_value

    # Positive EV
    assert round(compute_expected_value(0.80, 1.35), 4) == 0.08
    # Negative EV
    assert round(compute_expected_value(0.50, 1.90), 4) == -0.05
    # Breakeven EV
    assert round(compute_expected_value(0.50, 2.00), 4) == 0.0
    # Invalid odds (None, <= 1.0, NaN)
    assert compute_expected_value(0.80, None) is None
    assert compute_expected_value(0.80, 1.0) is None
    assert compute_expected_value(0.80, 0.95) is None
    assert compute_expected_value(0.80, float("nan")) is None


def test_roi_and_pnl_with_push_accounting():
    """Verify correct PnL, ROI, and Push handling (Pushes refund stake with 0 profit/loss)."""
    from zola.prediction.calibration import compute_roi_and_pnl

    records = [
        {"settlement_status": "WIN", "odd": 1.50},    # +0.50 profit
        {"settlement_status": "LOSS", "odd": 1.80},   # -1.00 loss
        {"settlement_status": "PUSH", "odd": 1.60},   #  0.00 push (refund)
        {"settlement_status": "WIN", "odd": 1.40},    # +0.40 profit
    ]

    res = compute_roi_and_pnl(records)
    assert res["total_bets"] == 4
    assert res["wins"] == 2
    assert res["losses"] == 1
    assert res["pushes"] == 1
    assert res["decided_bets"] == 3
    # Net PnL = +0.50 - 1.00 + 0.00 + 0.40 = -0.10
    assert res["pnl"] == -0.10
    # ROI = -0.10 / 4 = -2.5%
    assert res["roi"] == -2.5
    # Win rate on decided = 2 / 3 = 66.67%
    assert res["win_rate_decided"] == 66.67
    # Win rate on total = 2 / 4 = 50.0%
    assert res["win_rate_total"] == 50.0


def test_probability_calibration_binning():
    """Verify probability binning, mean predicted prob, and calibration error."""
    from zola.prediction.calibration import compute_calibration_bins

    records = [
        {"probability": 0.72, "settlement_status": "WIN", "odd": 1.30},
        {"probability": 0.74, "settlement_status": "WIN", "odd": 1.32},
        {"probability": 0.73, "settlement_status": "LOSS", "odd": 1.30},
        {"probability": 0.71, "settlement_status": "WIN", "odd": 1.35},
    ]

    bins = compute_calibration_bins(records)
    bin_70_74 = next(b for b in bins if b["bin_label"] == "70-74%")

    assert bin_70_74["count"] == 4
    # Mean prob = (0.72 + 0.74 + 0.73 + 0.71) / 4 = 0.725 -> 72.5%
    assert bin_70_74["mean_predicted_prob"] == 72.5
    # Actual win rate = 3 / 4 = 75.0%
    assert bin_70_74["actual_win_rate"] == 75.0
    # Calibration error = |72.5 - 75.0| = 2.5%
    assert bin_70_74["calibration_error"] == 2.5


def test_high_confidence_engine_persistence(repo):
    """Test that PredictionEngine executes HighConfidenceSelector and persists metadata to SQLite."""
    import json
    from zola.database.models import Competition, Fixture, Team
    from zola.prediction.engine import PredictionEngine

    cid = repo.upsert_competition(
        Competition(external_id="PL", provider="api-football", name="Premier League", code="PL")
    )
    t1_id = repo.upsert_team(Team(external_id="1", provider="api-football", name="Arsenal", short_name="Arsenal"))
    t2_id = repo.upsert_team(Team(external_id="2", provider="api-football", name="Chelsea", short_name="Chelsea"))

    fid = repo.upsert_fixture(
        Fixture(
            external_id="upcoming_derby_hc_101",
            provider="api-football",
            competition_id=cid,
            utc_date="2026-11-25 20:00:00",
            status="SCHEDULED",
            home_team_id=t1_id,
            away_team_id=t2_id,
        )
    )

    engine = PredictionEngine(repo)
    results = engine.predict_fixture(fid, persist=True)

    assert "ensemble" in results
    ens = results["ensemble"]
    # Check that high_confidence_report is attached
    sample_market = next(iter(ens.values()))
    assert "high_confidence_report" in sample_market.features
    report = sample_market.features["high_confidence_report"]
    assert "is_skipped" in report
    assert "total_picks" in report

    # Verify persisted prediction in database
    db_preds = repo.get_predictions_for_fixture(fid)
    assert len(db_preds) > 0
    ensemble_preds = [p for p in db_preds if p.model_name == "ensemble"]
    assert len(ensemble_preds) == 8
    for p in ensemble_preds:
        feat = json.loads(p.features_json) if p.features_json else {}
        assert "high_confidence_report" in feat

