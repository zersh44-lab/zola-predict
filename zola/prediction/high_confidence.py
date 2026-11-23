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
from zola.prediction.calibration import compute_expected_value, is_subtype_approved

logger = logging.getLogger(__name__)


# Historical Empirical Win-Rate Baselines (from verified match database)
MARKET_EMPIRICAL_WIN_RATES: Dict[str, float] = {
    "OU05": 0.961,         # Over 0.5 Goals
    "OU15": 0.846,         # Over 1.5 Goals
    "DOUBLE_CHANCE": 0.792,# Double Chance (1X / X2)
    "DNB": 0.765,          # Draw No Bet (Disabled)
    "1X2": 0.680,          # Outright 1X2 (Disabled)
    "BTTS": 0.640,         # Both Teams to Score (Disabled)
    "OU25": 0.620,         # Over/Under 2.5 (Disabled)
    "OU35": 0.650,         # Over/Under 3.5 (Disabled)
}

# Strict Active Market Allowlist (Only empirically verified markets active)
APPROVED_HIGH_CONFIDENCE_MARKETS: set[str] = {"OU05", "OU15"}
DISABLED_MARKETS: set[str] = {"1X2", "BTTS", "OU25", "OU35", "DNB", "DOUBLE_CHANCE"}

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
    expected_value: Optional[float] = None
    expected_value_pct: Optional[str] = None
    market_odd: Optional[float] = None
    is_experimental: bool = False
    betting_value_status: str = "UNVERIFIED"
    probability_source: str = "STATISTICAL_ENSEMBLE_CONSENSUS"
    odds_source: str = "VERIFIED_PRE_MATCH_BOOKMAKER"
    is_verified_odds: bool = False

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
    def _extract_and_validate_odds(
        cls,
        pred: ModelPrediction,
        market: str,
        outcome: str,
        odds_data: Optional[Dict[str, Any]] = None,
        fixture: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], bool, Optional[str]]:
        """Extract and validate pre-match bookmaker decimal odd against strict authenticity checks.

        Returns:
            (odd_val, is_valid, rejection_reason)
        """
        raw_odd: Optional[float] = None

        if odds_data and market in odds_data:
            odds_item = odds_data[market]
            if isinstance(odds_item, dict):
                # 1. Reject synthetic / estimated odds
                if odds_item.get("is_synthetic") is True or odds_item.get("synthetic") is True:
                    return None, False, f"{market} rejected: Synthetic odds cannot be treated as verified bookmaker odds."
                if str(odds_item.get("source", "")).lower() in ("synthetic", "estimated", "fair_odd", "model"):
                    return None, False, f"{market} rejected: Synthetic or estimated odds cannot be treated as verified bookmaker odds."
                if str(odds_item.get("status", "")).upper() in ("SYNTHETIC", "ESTIMATED"):
                    return None, False, f"{market} rejected: Synthetic or estimated odds cannot be treated as verified bookmaker odds."
                if odds_item.get("is_estimated") is True or odds_item.get("estimated") is True:
                    return None, False, f"{market} rejected: Estimated odds cannot be treated as verified bookmaker odds."

                # 2. Reject post-kickoff / in-play odds
                if odds_item.get("is_post_kickoff") is True or odds_item.get("post_kickoff") is True:
                    return None, False, f"{market} rejected: Odds captured after kickoff cannot be used for pre-match predictions."
                if odds_item.get("in_play") is True or odds_item.get("live") is True or odds_item.get("is_in_play") is True:
                    return None, False, f"{market} rejected: Live in-play odds captured after kickoff cannot be used for pre-match predictions."
                if str(odds_item.get("status", "")).upper() in ("IN_PLAY", "LIVE", "POST_KICKOFF"):
                    return None, False, f"{market} rejected: Live in-play odds cannot be used for pre-match predictions."

                # 3. Reject stale odds
                if odds_item.get("is_stale") is True or odds_item.get("stale") is True:
                    return None, False, f"{market} rejected: Stale bookmaker odds detected."
                if str(odds_item.get("status", "")).upper() == "STALE":
                    return None, False, f"{market} rejected: Stale bookmaker odds detected."

                # 4. Check timestamp comparison if captured_at is provided
                captured_at = odds_item.get("captured_at")
                kickoff_utc = (fixture.get("utc_date") if fixture else None) or odds_item.get("kickoff_utc")
                if captured_at and kickoff_utc:
                    try:
                        from datetime import datetime
                        cap_dt = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
                        k_dt = datetime.fromisoformat(str(kickoff_utc).replace("Z", "+00:00"))
                        if cap_dt >= k_dt:
                            return None, False, f"{market} rejected: Odds captured after kickoff ({captured_at} >= {kickoff_utc}) cannot be used."
                    except Exception:
                        pass

                raw = odds_item.get(outcome) or odds_item.get("market_odd") or odds_item.get("odd")
                if raw is not None:
                    try:
                        raw_odd = float(raw)
                    except (ValueError, TypeError):
                        pass
            elif isinstance(odds_item, (int, float)):
                raw_odd = float(odds_item)

        # Fallback to prediction value_analysis if not in odds_data
        if raw_odd is None and hasattr(pred, "value_analysis") and pred.value_analysis:
            va = pred.value_analysis
            if isinstance(va, dict):
                if va.get("is_synthetic") is True or str(va.get("status", "")).upper() == "SYNTHETIC":
                    return None, False, f"{market} rejected: Synthetic odds cannot be treated as verified bookmaker odds."
                if str(va.get("source", "")).lower() in ("synthetic", "estimated", "fair_odd"):
                    return None, False, f"{market} rejected: Estimated odds cannot be treated as verified bookmaker odds."
                if va.get("status") == "AVAILABLE":
                    raw = va.get("market_odd")
                    if raw is not None:
                        try:
                            raw_odd = float(raw)
                        except (ValueError, TypeError):
                            pass

        # Check fixture status for post-kickoff
        if fixture:
            f_status = str(fixture.get("status", "")).upper()
            if f_status in ("IN_PLAY", "LIVE", "PAUSED", "FINISHED", "COMPLETED"):
                return None, False, f"{market} rejected: Fixture is already in-play or completed ({f_status}); pre-match odds invalid."

        if raw_odd is None:
            return None, False, f"{market} rejected: Missing pre-match bookmaker odds (strict odds requirement enabled)."

        import math
        if math.isnan(raw_odd) or math.isinf(raw_odd) or raw_odd <= 1.0:
            return None, False, f"{market} rejected: Invalid bookmaker odd ({raw_odd}). Odds must be numeric and > 1.0."

        return raw_odd, True, None

    @classmethod
    def evaluate_fixture(
        cls,
        fixture: Dict[str, Any],
        ensemble_predictions: Dict[str, ModelPrediction],
        comparison_stats: Optional[Dict[str, Any]] = None,
        home_form: Optional[List[str]] = None,
        away_form: Optional[List[str]] = None,
        h2h_summary: Optional[Dict[str, Any]] = None,
        odds_data: Optional[Dict[str, Any]] = None,
        require_valid_odds: bool = True,
        min_ev_margin: float = 0.0,
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
        # 1. Evaluate DOUBLE CHANCE (Strict Policy: DISABLED)
        # -------------------------------------------------------------
        if "DOUBLE_CHANCE" in ensemble_predictions:
            pred_dc = ensemble_predictions["DOUBLE_CHANCE"]
            dc_pick = pred_dc.predicted_outcome
            skip_reasons.append(
                f"Double Chance ({dc_pick}) blocked: Market disabled under strict policy (only OU05 and OU15 approved)."
            )

        # -------------------------------------------------------------
        # 2. Evaluate OVER 1.5 GOALS (High Empirical Priority: 84.6% WR)
        # -------------------------------------------------------------
        if "OU15" in ensemble_predictions:
            pred_ou15 = ensemble_predictions["OU15"]
            o15_pick = pred_ou15.predicted_outcome
            o15_prob = pred_ou15.probability

            is_app, app_reason = is_subtype_approved("OU15", o15_pick)
            if not is_app:
                skip_reasons.append(f"OU15 ({o15_pick}) rejected: {app_reason}")
            elif o15_pick == "OVER_15" and o15_prob >= 0.70 and pred_ou15.confidence in ("HIGH", "MEDIUM"):
                m_odd, is_odd_valid, odd_rej_reason = cls._extract_and_validate_odds(
                    pred_ou15, "OU15", o15_pick, odds_data, fixture
                )
                ev = compute_expected_value(o15_prob, m_odd) if (m_odd and is_odd_valid) else None

                if require_valid_odds and not is_odd_valid:
                    skip_reasons.append(odd_rej_reason or "Over 1.5 Goals rejected: Missing valid pre-match odds.")
                elif require_valid_odds and (ev is None or ev < min_ev_margin or ev <= 0.0):
                    ev_str = f"{ev*100:+.1f}%" if ev is not None else "N/A"
                    skip_reasons.append(f"Over 1.5 Goals rejected: Expected Value ({ev_str}) is not positive (positive EV required).")
                else:
                    xg_str = f"Combined xG is {tot_xg:.2f}" if tot_xg else "Offensive goal rates support multiple scoring chances"
                    ev_info = f" Verified pre-match odd @ {m_odd:.2f} offers positive EV ({ev*100:+.1f}%)." if (m_odd and ev) else ""
                    rationale = (
                        f"Over 1.5 Goals backed by {o15_prob * 100:.1f}% model probability and strong goal metrics.{ev_info} "
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
                        "market_odd": m_odd,
                        "expected_value": ev,
                        "is_verified_odds": is_odd_valid,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "GOALS",
                    })

        # -------------------------------------------------------------
        # 3. Evaluate OVER 0.5 GOALS (Benchmark Safe Floor: 96.1% WR)
        # -------------------------------------------------------------
        if "OU05" in ensemble_predictions:
            pred_ou05 = ensemble_predictions["OU05"]
            o05_pick = pred_ou05.predicted_outcome
            o05_prob = pred_ou05.probability

            is_app, app_reason = is_subtype_approved("OU05", o05_pick)
            if not is_app:
                skip_reasons.append(f"OU05 ({o05_pick}) rejected: {app_reason}")
            elif o05_pick == "OVER_05" and o05_prob >= 0.88 and pred_ou05.confidence in ("HIGH", "MEDIUM"):
                m_odd, is_odd_valid, odd_rej_reason = cls._extract_and_validate_odds(
                    pred_ou05, "OU05", o05_pick, odds_data, fixture
                )
                ev = compute_expected_value(o05_prob, m_odd) if (m_odd and is_odd_valid) else None

                if require_valid_odds and not is_odd_valid:
                    skip_reasons.append(odd_rej_reason or "Over 0.5 Goals rejected: Missing valid pre-match odds.")
                elif require_valid_odds and (ev is None or ev < min_ev_margin or ev <= 0.0):
                    ev_str = f"{ev*100:+.1f}%" if ev is not None else "N/A"
                    skip_reasons.append(f"Over 0.5 Goals rejected: Expected Value ({ev_str}) is not positive (positive EV required).")
                else:
                    ev_info = f" Verified pre-match odd @ {m_odd:.2f} offers positive EV ({ev*100:+.1f}%)." if (m_odd and ev) else ""
                    rationale = (
                        f"Over 0.5 Goals offers an elite statistical safety floor at {o05_prob * 100:.1f}% model probability.{ev_info} "
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
                        "market_odd": m_odd,
                        "expected_value": ev,
                        "is_verified_odds": is_odd_valid,
                        "rationale": rationale,
                        "risks": risks,
                        "quality_score": q_score,
                        "category": "GOALS_FLOOR",
                    })

        # -------------------------------------------------------------
        # 4. Enforce Active Market Allowlist (OU05, OU15 only)
        # Blocked: 1X2, BTTS, OU25, OU35, DNB, DOUBLE_CHANCE
        # -------------------------------------------------------------
        candidates = [c for c in candidates if c["market"] in APPROVED_HIGH_CONFIDENCE_MARKETS]

        # -------------------------------------------------------------
        # 5. De-duplication and Non-Redundancy Filter
        # -------------------------------------------------------------
        # Sort candidates by quality_score descending
        candidates.sort(key=lambda c: c["quality_score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_markets = set()
        has_ou15 = False

        for cand in candidates:
            m = cand["market"]
            outcome = cand["predicted_outcome"]

            # Strict verification: must be in approved allowlist
            if m not in APPROVED_HIGH_CONFIDENCE_MARKETS:
                continue

            # Correlation Rule: Prefer Over 1.5 Goals. If Over 1.5 is selected, do not add Over 0.5
            if m == "OU15" and outcome == "OVER_15":
                has_ou15 = True
                selected = [s for s in selected if s["market"] != "OU05"]
            elif m == "OU05" and has_ou15:
                continue

            # Do not repeat the exact same market
            if m in selected_markets:
                continue

            selected.append(cand)
            selected_markets.add(m)

            # Cap strictly at maximum 3 predictions per match
            if len(selected) >= 3:
                break

        # Final guarantee: strictly approved markets and max 3
        selected = [s for s in selected if s["market"] in APPROVED_HIGH_CONFIDENCE_MARKETS][:3]

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
            actual_odd = item.get("market_odd")
            est_odd = round(actual_odd, 2) if actual_odd is not None else round(fair_odd * 1.04, 2)
            ev = item.get("expected_value")

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
                    expected_value=round(ev, 4) if ev is not None else None,
                    expected_value_pct=f"{ev * 100:+.1f}%" if ev is not None else None,
                    market_odd=actual_odd,
                    is_experimental=False,
                    betting_value_status="VERIFIED_POSITIVE_EV" if (ev is not None and ev > 0) else "NO_VERIFIED_VALUE",
                    probability_source="STATISTICAL_ENSEMBLE_CONSENSUS",
                    odds_source="VERIFIED_PRE_MATCH_BOOKMAKER" if item.get("is_verified_odds") else "UNVERIFIED",
                    is_verified_odds=bool(item.get("is_verified_odds")),
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
