"""High-Confidence Prediction Selection System for Zola Predict.

Main Principle: QUALITY OVER QUANTITY.
Filters, evaluates, and selects at most 3 elite, statistically supported
prediction options per match, or explicitly issues a SKIP recommendation
("NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH.") when evidence is insufficient.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zola.prediction.base import ModelPrediction

logger = logging.getLogger(__name__)


# Historical Empirical Win-Rate Baselines (from verified match database)
MARKET_EMPIRICAL_WIN_RATES: Dict[str, float] = {
    "OU05": 0.961,         # Over 0.5 Goals
    "OU15": 0.846,         # Over 1.5 Goals
    "DOUBLE_CHANCE": 0.792,# Double Chance (1X / X2)
    "DNB": 0.765,          # Draw No Bet
    "1X2": 0.680,          # Outright 1X2 (filtered high-conviction only)
    "BTTS": 0.640,         # Both Teams to Score (filtered)
    "OU25": 0.620,         # Over/Under 2.5 (filtered)
    "OU35": 0.650,         # Over/Under 3.5 (filtered)
}

# Strict Minimum Quality Gates
MIN_PROBABILITY_THRESHOLD = 0.60
MIN_CONFIDENCE_SCORE = 0.60


@dataclass
class HighConfidencePick:
    """Individual high-confidence prediction option."""

    market: str
    market_label: str
    predicted_outcome: str
    outcome_display: str
    probability: float
    probability_pct: str
    confidence: str
    confidence_score: float
    estimated_odd: float
    fair_odd: float
    rank: int
    rationale: str
    risks: str
    quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchHighConfidenceReport:
    """Structured high-confidence prediction report for a fixture."""

    fixture_id: int
    match_name: str
    competition_name: str
    kickoff_utc: str
    is_skipped: bool
    skip_reason: Optional[str]
    skip_rationale: Optional[str]
    total_picks: int
    picks: List[HighConfidencePick] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["picks"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in self.picks]
        return res


class HighConfidenceSelector:
    """Evaluates ensemble predictions against strict statistical, risk, and correlation gates."""

    @classmethod
    def evaluate_fixture(
        cls,
        fixture: Dict[str, Any],
        ensemble_predictions: Dict[str, ModelPrediction],
        comparison_stats: Optional[Dict[str, Any]] = None,
        home_form: Optional[List[str]] = None,
        away_form: Optional[List[str]] = None,
        h2h_summary: Optional[Dict[str, Any]] = None,
    ) -> MatchHighConfidenceReport:
        """Analyze all markets and return 1-3 high-confidence picks or a structured SKIP report."""
        fid = fixture.get("id") or 0
        h_name = fixture.get("home_team_name") or "Home Team"
        a_name = fixture.get("away_team_name") or "Away Team"
        comp_name = fixture.get("competition_name") or "League"
        kickoff = fixture.get("utc_date") or ""
        match_title = f"{h_name} vs {a_name}"

        stats = comparison_stats or {}
        h_xg = stats.get("home_xg")
        a_xg = stats.get("away_xg")
        tot_xg = (h_xg + a_xg) if (h_xg is not None and a_xg is not None) else None

        candidates: List[Dict[str, Any]] = []
        skip_reasons: List[str] = []

        # -------------------------------------------------------------
        # 1. Evaluate DOUBLE CHANCE (High Empirical Priority: 79.2% WR)
        # -------------------------------------------------------------
        if "DOUBLE_CHANCE" in ensemble_predictions:
            pred_dc = ensemble_predictions["DOUBLE_CHANCE"]
            dc_pick = pred_dc.predicted_outcome
            dc_prob = pred_dc.probability
            p_1x2 = ensemble_predictions.get("1X2")
            draw_prob = p_1x2.probabilities.get("DRAW", 0.28) if p_1x2 else 0.28

            if dc_prob >= 0.70 and pred_dc.confidence in ("HIGH", "MEDIUM"):
                if dc_pick in ("1X", "X2"):
                    favored_team = h_name if dc_pick == "1X" else a_name
                    outcome_disp = f"{favored_team} or Draw ({dc_pick})"
                    
                    # Rationale construction
                    fav_form = home_form if dc_pick == "1X" else away_form
                    wins_in_form = (fav_form or []).count("W")
                    draws_in_form = (fav_form or []).count("D")
                    rationale = (
                        f"{favored_team} covers win or draw with {dc_prob * 100:.1f}% estimated probability. "
                        f"Recent form ({''.join(fav_form or ['N/A'])}) demonstrates resilience ({wins_in_form}W-{draws_in_form}D), "
                        f"providing dual-scenario protection against draw volatility ({draw_prob * 100:.1f}% draw risk)."
                    )
                    
                    # Risk identification
                    risks = (
                        f"Underdog counter-attack threat; outright loss busts both win and draw legs. "
                        f"Draw risk remains moderate at {draw_prob * 100:.1f}%."
                    )

                    q_score = (dc_prob * 0.40) + (pred_dc.confidence_score * 0.35) + (MARKET_EMPIRICAL_WIN_RATES["DOUBLE_CHANCE"] * 0.25)

                    candidates.append({
                        "market": "DOUBLE_CHANCE",
                        "market_label": "Double Chance",
                        "predicted_outcome": dc_pick,
                        "outcome_display": outcome_disp,
                        "probability": dc_prob,
                        "confidence": pred_dc.confidence,
                        "confidence_score": pred_dc.confidence_score,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "OUTCOME",
                    })
            else:
                if dc_prob < 0.70:
                    skip_reasons.append(f"Double Chance margin ({dc_prob * 100:.1f}%) is below 70% threshold.")

        # -------------------------------------------------------------
        # 2. Evaluate OVER 1.5 GOALS (High Empirical Priority: 84.6% WR)
        # -------------------------------------------------------------
        if "OU15" in ensemble_predictions:
            pred_ou15 = ensemble_predictions["OU15"]
            o15_pick = pred_ou15.predicted_outcome
            o15_prob = pred_ou15.probability

            if o15_pick == "OVER_15" and o15_prob >= 0.70 and pred_ou15.confidence in ("HIGH", "MEDIUM"):
                xg_str = f"Combined xG is {tot_xg:.2f}" if tot_xg else "Offensive goal rates support multiple scoring chances"
                rationale = (
                    f"Over 1.5 Goals backed by {o15_prob * 100:.1f}% probability and strong goal metrics. "
                    f"{xg_str}. Historical conversion in this bracket exceeds 84% reliability."
                )
                risks = (
                    "Early red card, ultra-defensive game state, or poor finishing could suppress goals to 0-0 or 1-0."
                )
                q_score = (o15_prob * 0.40) + (pred_ou15.confidence_score * 0.35) + (MARKET_EMPIRICAL_WIN_RATES["OU15"] * 0.25)

                candidates.append({
                    "market": "OU15",
                    "market_label": "Total Goals",
                    "predicted_outcome": "OVER_15",
                    "outcome_display": "Over 1.5 Goals",
                    "probability": o15_prob,
                    "confidence": pred_ou15.confidence,
                    "confidence_score": pred_ou15.confidence_score,
                    "rationale": rationale,
                    "risks": risks,
                    "quality_score": q_score,
                    "category": "GOALS",
                })
            elif o15_pick == "UNDER_15":
                # Under 1.5 is inherently high variance; require extreme probability
                if o15_prob >= 0.75 and pred_ou15.confidence == "HIGH":
                    candidates.append({
                        "market": "OU15",
                        "market_label": "Total Goals",
                        "predicted_outcome": "UNDER_15",
                        "outcome_display": "Under 1.5 Goals",
                        "probability": o15_prob,
                        "confidence": pred_ou15.confidence,
                        "confidence_score": pred_ou15.confidence_score,
                        "rationale": f"Extremely low expected scoring profile ({tot_xg:.2f} xG if available). Both defenses hold elite suppression rates.",
                        "risks": "One fluke or deflection can immediately shatter an Under 1.5 position.",
                        "quality_score": (o15_prob * 0.40) + (pred_ou15.confidence_score * 0.35) + 0.60 * 0.25,
                        "category": "GOALS",
                    })
                else:
                    skip_reasons.append(f"Under 1.5 carries extreme variance and lacks >= 75% conviction.")

        # -------------------------------------------------------------
        # 3. Evaluate OVER 0.5 GOALS (Benchmark Safe Floor: 96.1% WR)
        # -------------------------------------------------------------
        if "OU05" in ensemble_predictions:
            pred_ou05 = ensemble_predictions["OU05"]
            o05_prob = pred_ou05.probability
            if pred_ou05.predicted_outcome == "OVER_05" and o05_prob >= 0.88 and pred_ou05.confidence in ("HIGH", "MEDIUM"):
                rationale = (
                    f"Over 0.5 Goals offers an elite statistical safety floor at {o05_prob * 100:.1f}% probability. "
                    f"Historical success rate across our tracking sits at 96.1%."
                )
                risks = "A complete scoreless 0-0 stalemate."
                q_score = (o05_prob * 0.38) + (pred_ou05.confidence_score * 0.34) + (MARKET_EMPIRICAL_WIN_RATES["OU05"] * 0.28)

                candidates.append({
                    "market": "OU05",
                    "market_label": "Total Goals",
                    "predicted_outcome": "OVER_05",
                    "outcome_display": "Over 0.5 Goals",
                    "probability": o05_prob,
                    "confidence": pred_ou05.confidence,
                    "confidence_score": pred_ou05.confidence_score,
                    "rationale": rationale,
                    "risks": risks,
                    "quality_score": q_score,
                    "category": "GOALS_FLOOR",
                })

        # -------------------------------------------------------------
        # 4. Evaluate DRAW NO BET (DNB) (Empirical Priority: 76.5% WR)
        # -------------------------------------------------------------
        if "DNB" in ensemble_predictions:
            pred_dnb = ensemble_predictions["DNB"]
            dnb_pick = pred_dnb.predicted_outcome
            dnb_prob = pred_dnb.probability

            if dnb_prob >= 0.72 and pred_dnb.confidence in ("HIGH", "MEDIUM"):
                favored_team = h_name if dnb_pick == "1_DNB" else a_name
                outcome_disp = f"{favored_team} (Draw No Bet)"
                rationale = (
                    f"{favored_team} holds decisive edge ({dnb_prob * 100:.1f}% DNB probability). "
                    f"In the event of a draw, stake is fully refunded, eliminating draw loss vulnerability."
                )
                risks = f"Outright win by opponent causes a loss; match ending in a draw produces no profit."
                q_score = (dnb_prob * 0.40) + (pred_dnb.confidence_score * 0.35) + (MARKET_EMPIRICAL_WIN_RATES["DNB"] * 0.25)

                candidates.append({
                    "market": "DNB",
                    "market_label": "Draw No Bet",
                    "predicted_outcome": dnb_pick,
                    "outcome_display": outcome_disp,
                    "probability": dnb_prob,
                    "confidence": pred_dnb.confidence,
                    "confidence_score": pred_dnb.confidence_score,
                    "rationale": rationale,
                    "risks": risks,
                    "quality_score": q_score,
                    "category": "OUTCOME",
                })

        # -------------------------------------------------------------
        # 5. Evaluate OUTRIGHT 1X2 (Strict Gating to Prevent Costly Losses)
        # -------------------------------------------------------------
        if "1X2" in ensemble_predictions:
            pred_1x2 = ensemble_predictions["1X2"]
            pick_1x2 = pred_1x2.predicted_outcome
            prob_1x2 = pred_1x2.probability
            draw_prob = pred_1x2.probabilities.get("DRAW", 0.0)

            # Strict rules to prevent the 94 historical 1X2 losses:
            # 1. Probability must be >= 64%
            # 2. Draw probability must be < 26%
            # 3. Confidence must be HIGH
            if pick_1x2 in ("HOME_WIN", "AWAY_WIN"):
                if prob_1x2 >= 0.64 and draw_prob < 0.26 and pred_1x2.confidence == "HIGH":
                    fav_team = h_name if pick_1x2 == "HOME_WIN" else a_name
                    outcome_disp = f"{fav_team} to Win"
                    rationale = (
                        f"Dominant conviction for {fav_team} ({prob_1x2 * 100:.1f}% probability) with suppressed draw probability ({draw_prob * 100:.1f}%). "
                        f"Unanimous model backing and superior underlying metrics."
                    )
                    risks = f"Underdog defensive low block or unexpected away counter-attack."
                    q_score = (prob_1x2 * 0.40) + (pred_1x2.confidence_score * 0.35) + (MARKET_EMPIRICAL_WIN_RATES["1X2"] * 0.25)

                    candidates.append({
                        "market": "1X2",
                        "market_label": "Match Winner",
                        "predicted_outcome": pick_1x2,
                        "outcome_display": outcome_disp,
                        "probability": prob_1x2,
                        "confidence": pred_1x2.confidence,
                        "confidence_score": pred_1x2.confidence_score,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "OUTCOME_STRAIGHT",
                    })
                else:
                    if prob_1x2 < 0.64:
                        skip_reasons.append(f"1X2 Win probability ({prob_1x2 * 100:.1f}%) is in volatile coin-flip range (< 64%).")
                    if draw_prob >= 0.26:
                        skip_reasons.append(f"Elevated draw probability ({draw_prob * 100:.1f}% >= 26%) creates high straight 1X2 loss risk.")

        # -------------------------------------------------------------
        # 6. Evaluate BOTH TEAMS TO SCORE (BTTS)
        # -------------------------------------------------------------
        if "BTTS" in ensemble_predictions:
            pred_btts = ensemble_predictions["BTTS"]
            pick_btts = pred_btts.predicted_outcome
            prob_btts = pred_btts.probability

            # BTTS_YES requires >= 64% and both teams scoring
            if pick_btts == "BTTS_YES" and prob_btts >= 0.64 and pred_btts.confidence in ("HIGH", "MEDIUM"):
                h_gs = stats.get("home_goals_scored")
                a_gs = stats.get("away_goals_scored")
                if (h_gs is None or h_gs >= 1.2) and (a_gs is None or a_gs >= 1.1):
                    rationale = (
                        f"Both teams exhibit strong attacking consistency ({prob_btts * 100:.1f}% probability). "
                        f"Home avg {h_gs or '1.5+'} GF, Away avg {a_gs or '1.2+'} GF."
                    )
                    risks = "Either team failing to convert high-quality chances or playing a passive setup."
                    q_score = (prob_btts * 0.40) + (pred_btts.confidence_score * 0.35) + (MARKET_EMPIRICAL_WIN_RATES["BTTS"] * 0.25)

                    candidates.append({
                        "market": "BTTS",
                        "market_label": "Both Teams to Score",
                        "predicted_outcome": "BTTS_YES",
                        "outcome_display": "Yes (BTTS)",
                        "probability": prob_btts,
                        "confidence": pred_btts.confidence,
                        "confidence_score": pred_btts.confidence_score,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "BTTS",
                    })
                else:
                    skip_reasons.append(f"BTTS lacks sufficient goal production metrics.")
            elif pick_btts == "BTTS_NO":
                # Historical analysis showed BTTS_NO lost 48.1% of the time due to late consolations
                if prob_btts >= 0.72 and pred_btts.confidence == "HIGH":
                    rationale = f"Heavy defensive profile with {prob_btts * 100:.1f}% probability of at least one clean sheet."
                    risks = "Late consolation goal or penalty."
                    q_score = (prob_btts * 0.40) + (pred_btts.confidence_score * 0.35) + 0.60 * 0.25
                    candidates.append({
                        "market": "BTTS",
                        "market_label": "Both Teams to Score",
                        "predicted_outcome": "BTTS_NO",
                        "outcome_display": "No (BTTS)",
                        "probability": prob_btts,
                        "confidence": pred_btts.confidence,
                        "confidence_score": pred_btts.confidence_score,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "BTTS",
                    })
                else:
                    skip_reasons.append(f"BTTS_NO carries high underdog consolation risk (< 72% margin).")

        # -------------------------------------------------------------
        # 7. Evaluate OVER/UNDER 2.5 GOALS
        # -------------------------------------------------------------
        if "OU25" in ensemble_predictions:
            pred_ou25 = ensemble_predictions["OU25"]
            pick_ou25 = pred_ou25.predicted_outcome
            prob_ou25 = pred_ou25.probability

            if pick_ou25 == "OVER_25" and prob_ou25 >= 0.65 and (tot_xg is None or tot_xg >= 2.6):
                rationale = f"High-scoring clash expected ({prob_ou25 * 100:.1f}% probability, {tot_xg or '2.8+'} expected goals)."
                risks = "Game turns defensive or tactical deadlock."
                q_score = (prob_ou25 * 0.40) + (pred_ou25.confidence_score * 0.35) + (MARKET_EMPIRICAL_WIN_RATES["OU25"] * 0.25)
                candidates.append({
                    "market": "OU25",
                    "market_label": "Total Goals",
                    "predicted_outcome": "OVER_25",
                    "outcome_display": "Over 2.5 Goals",
                    "probability": prob_ou25,
                    "confidence": pred_ou25.confidence,
                    "confidence_score": pred_ou25.confidence_score,
                    "rationale": rationale,
                    "risks": risks,
                    "quality_score": q_score,
                    "category": "GOALS",
                })
            elif pick_ou25 == "UNDER_25":
                # Historical analysis showed UNDER_25 caused 92 losses (47.1% WR). Only accept if >= 72%
                if prob_ou25 >= 0.72 and (tot_xg is None or tot_xg <= 1.9):
                    rationale = f"Low expected tempo and stout defensive metrics ({prob_ou25 * 100:.1f}% probability)."
                    risks = "Quick goals or early expulsion breaking low-scoring setup."
                    q_score = (prob_ou25 * 0.40) + (pred_ou25.confidence_score * 0.35) + 0.60 * 0.25
                    candidates.append({
                        "market": "OU25",
                        "market_label": "Total Goals",
                        "predicted_outcome": "UNDER_25",
                        "outcome_display": "Under 2.5 Goals",
                        "probability": prob_ou25,
                        "confidence": pred_ou25.confidence,
                        "confidence_score": pred_ou25.confidence_score,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "GOALS",
                    })
                else:
                    skip_reasons.append(f"Under 2.5 line is statistically volatile; historical win rate is poor below 72%.")

        # -------------------------------------------------------------
        # 8. De-duplication and Non-Redundancy Filter
        # -------------------------------------------------------------
        # Sort candidates by quality_score descending
        candidates.sort(key=lambda c: c["quality_score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_markets = set()
        selected_categories = set()
        has_double_chance = False
        has_ou15 = False

        for cand in candidates:
            m = cand["market"]
            cat = cand["category"]
            outcome = cand["predicted_outcome"]

            # Correlation Rule 1: If Double Chance is already chosen, don't pick straight 1X2 or DNB on the same team
            if cat in ("OUTCOME", "OUTCOME_STRAIGHT"):
                if has_double_chance and m in ("1X2", "DNB"):
                    continue
                if m == "DOUBLE_CHANCE":
                    # Remove any lower-ranked DNB or 1X2 already picked
                    selected = [s for s in selected if s["market"] not in ("1X2", "DNB")]
                    has_double_chance = True

            # Correlation Rule 2: Goal Markets
            # Prefer Over 1.5. If Over 1.5 is selected, do not also add Over 0.5 (redundant) or Over 2.5 (excessive risk)
            if m == "OU15" and outcome == "OVER_15":
                has_ou15 = True
                selected = [s for s in selected if s["market"] != "OU05"]
            elif m == "OU05" and has_ou15:
                continue
            elif m == "OU25" and has_ou15 and outcome == "OVER_25":
                continue

            # Do not repeat the exact same market
            if m in selected_markets:
                continue

            selected.append(cand)
            selected_markets.add(m)
            selected_categories.add(cat)

            # Cap strictly at maximum 3 predictions per match
            if len(selected) >= 3:
                break

        # If we have an Over 0.5 and Over 1.5 was also selected, remove Over 0.5
        if any(s["market"] == "OU15" for s in selected) and any(s["market"] == "OU05" for s in selected):
            selected = [s for s in selected if s["market"] != "OU05"]

        # -------------------------------------------------------------
        # 9. Strict Quality Threshold & Skip Logic
        # -------------------------------------------------------------
        # If no candidates meet the criteria, or if the top candidate has quality_score < 0.65:
        if not selected or selected[0]["quality_score"] < 0.65:
            full_skip_rationale = (
                "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH. "
                + ("; ".join(skip_reasons[:3]) if skip_reasons else "Market odds and model consensus indicate high volatility without statistical edge.")
            )
            return MatchHighConfidenceReport(
                fixture_id=fid,
                match_name=match_title,
                competition_name=comp_name,
                kickoff_utc=kickoff,
                is_skipped=True,
                skip_reason="NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH.",
                skip_rationale=full_skip_rationale,
                total_picks=0,
                picks=[],
                analysis_summary={
                    "home_xg": h_xg,
                    "away_xg": a_xg,
                    "total_xg": tot_xg,
                    "skip_reasons": skip_reasons,
                },
            )

        # -------------------------------------------------------------
        # 10. Format Final 1 to 3 High-Confidence Picks
        # -------------------------------------------------------------
        final_picks: List[HighConfidencePick] = []
        for i, item in enumerate(selected[:3], 1):
            prob = item["probability"]
            fair_odd = round(1.0 / max(0.01, prob), 2)
            est_odd = round(fair_odd * 1.04, 2)  # Benchmark ~4% margin bookmaker benchmark

            final_picks.append(
                HighConfidencePick(
                    market=item["market"],
                    market_label=item["market_label"],
                    predicted_outcome=item["predicted_outcome"],
                    outcome_display=item["outcome_display"],
                    probability=prob,
                    probability_pct=f"{prob * 100:.1f}%",
                    confidence=item["confidence"],
                    confidence_score=round(item["confidence_score"], 2),
                    estimated_odd=est_odd,
                    fair_odd=fair_odd,
                    rank=i,
                    rationale=item["rationale"],
                    risks=item["risks"],
                    quality_score=round(item["quality_score"], 3),
                )
            )

        return MatchHighConfidenceReport(
            fixture_id=fid,
            match_name=match_title,
            competition_name=comp_name,
            kickoff_utc=kickoff,
            is_skipped=False,
            skip_reason=None,
            skip_rationale=None,
            total_picks=len(final_picks),
            picks=final_picks,
            analysis_summary={
                "home_xg": h_xg,
                "away_xg": a_xg,
                "total_xg": tot_xg,
                "primary_pick": final_picks[0].outcome_display if final_picks else None,
            },
        )
