"""Dixon-Coles bivariate match prediction model with low-score correlation and adaptive competition tuning."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from zola.prediction.base import (
    BasePredictionModel,
    ModelPrediction,
    derive_markets_from_score_matrix,
    poisson_pmf,
)


def parse_utc_dt(dt_str: Optional[str]) -> Optional[datetime]:
    """Safely parse UTC datetime from strings with or without timezone info."""
    if not dt_str:
        return None
    try:
        s = str(dt_str).strip().replace("Z", "+00:00")
        if "+" not in s and "-" not in s[10:]:
            dt = datetime.fromisoformat(s)
            return dt.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(s)
    except Exception:
        return None


class DixonColesModel(BasePredictionModel):
    """Dixon-Coles model adjusting for low-score dependence, recency decay, and competition dynamics."""

    def __init__(self, rho: float = -0.11, decay_half_life_days: float = 120.0):
        self.rho = rho
        self.decay_half_life_days = max(30.0, decay_half_life_days)
        # decay factor xi where exp(-xi * half_life) = 0.5 -> xi = ln(2) / half_life
        self.xi = math.log(2.0) / self.decay_half_life_days

    @property
    def name(self) -> str:
        return "dixon_coles"

    @property
    def description(self) -> str:
        return (
            "Dixon-Coles bivariate model that adjusts for the correlation between low scores "
            "(0-0, 1-0, 0-1, 1-1) and applies exponential time-decay weighting to recent matches."
        )

    def _tau(self, x: int, y: int, lambda_h: float, lambda_a: float, rho: Optional[float] = None) -> float:
        """Dixon-Coles adjustment factor tau(x, y)."""
        active_rho = self.rho if rho is None else rho
        if x == 0 and y == 0:
            return max(0.01, 1.0 - (lambda_h * lambda_a * active_rho))
        elif x == 1 and y == 0:
            return max(0.01, 1.0 + (lambda_a * active_rho))
        elif x == 0 and y == 1:
            return max(0.01, 1.0 + (lambda_h * active_rho))
        elif x == 1 and y == 1:
            return max(0.01, 1.0 - active_rho)
        return 1.0

    def predict(
        self,
        fixture: Dict[str, Any],
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
        h2h_matches: List[Dict[str, Any]],
        league_matches: List[Dict[str, Any]],
    ) -> Dict[str, ModelPrediction]:
        notes: List[str] = []
        n_home = len(home_matches)
        n_away = len(away_matches)

        sample_depth = min(1.0, (n_home + n_away) / 10.0)
        league_depth = min(1.0, len(league_matches) / 20.0) if league_matches else 0.45
        confidence_score = round(max(0.45, min(0.95, 0.65 * sample_depth + 0.35 * league_depth)), 2)

        if confidence_score >= 0.70:
            confidence = "HIGH"
        elif confidence_score >= 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if n_home < 3 or n_away < 3:
            notes.append("Regularized with competition baseline and recency-weighted decay.")

        # Reference kickoff time: use fixture date for backtesting fidelity; fallback to now
        ref_dt = parse_utc_dt(fixture.get("utc_date")) or datetime.now(timezone.utc)

        comp_str = str(fixture.get("competition_name") or fixture.get("competition_code") or "").lower()
        is_intl = any(k in comp_str for k in ("nations", "uefa", "fifa", "world", "cup", "qualifier"))
        active_half_life = 365.0 if is_intl else self.decay_half_life_days
        active_xi = math.log(2.0) / active_half_life

        # Competition-adaptive rho: tactical low-scoring leagues have stronger draw correlation
        if any(c in comp_str for c in ("serie a", "la liga", "laliga")):
            active_rho = -0.14
        elif any(c in comp_str for c in ("bundesliga", "eredivisie")):
            active_rho = -0.07
        else:
            active_rho = self.rho

        def weighted_stats(matches: List[Dict[str, Any]]) -> Tuple[float, float, float]:
            tot_scored_w = 0.0
            tot_conceded_w = 0.0
            tot_w = 0.0

            for m in matches:
                # Calculate elapsed days prior to match kickoff
                days_ago = 21.0
                utc_str = m.get("utc_date")
                if utc_str:
                    match_dt = parse_utc_dt(utc_str)
                    if match_dt:
                        diff = (ref_dt - match_dt).total_seconds() / 86400.0
                        days_ago = max(0.5, diff)

                weight = math.exp(-active_xi * days_ago)
                is_home = bool(m.get("is_home", 1))
                h_goals = m.get("full_time_home_score", 0) or 0
                a_goals = m.get("full_time_away_score", 0) or 0

                scored = h_goals if is_home else a_goals
                conceded = a_goals if is_home else h_goals

                tot_scored_w += scored * weight
                tot_conceded_w += conceded * weight
                tot_w += weight

            if tot_w <= 0.0:
                return 1.45, 1.25, 1.0
            return tot_scored_w / tot_w, tot_conceded_w / tot_w, tot_w

        home_scored_w, home_conceded_w, _ = weighted_stats(home_matches)
        away_scored_w, away_conceded_w, _ = weighted_stats(away_matches)

        # Dynamic competition baseline from league matches
        if league_matches and len(league_matches) >= 8:
            tot_h_goals = sum(m.get("full_time_home_score", 0) or 0 for m in league_matches)
            tot_a_goals = sum(m.get("full_time_away_score", 0) or 0 for m in league_matches)
            league_home_baseline = max(0.6, tot_h_goals / len(league_matches))
            league_away_baseline = max(0.6, tot_a_goals / len(league_matches))
        else:
            league_home_baseline = 1.55
            league_away_baseline = 1.25

        # Calculate lambda with time decay and competition attack/defense strengths
        home_att = home_scored_w / league_home_baseline
        away_def = away_conceded_w / league_home_baseline
        lambda_h = max(0.25, home_att * away_def * league_home_baseline)

        away_att = away_scored_w / league_away_baseline
        home_def = home_conceded_w / league_away_baseline
        lambda_a = max(0.25, away_att * home_def * league_away_baseline)

        lambda_h = min(4.5, round(lambda_h, 3))
        lambda_a = min(4.5, round(lambda_a, 3))

        # Generate bivariate score matrix with Dixon-Coles tau adjustment
        score_matrix: Dict[Tuple[int, int], float] = {}
        for h in range(8):
            for a in range(8):
                base_prob = poisson_pmf(h, lambda_h) * poisson_pmf(a, lambda_a)
                tau_adj = self._tau(h, a, lambda_h, lambda_a, active_rho)
                score_matrix[(h, a)] = base_prob * tau_adj

        features = {
            "lambda_home_dc": lambda_h,
            "lambda_away_dc": lambda_a,
            "rho": active_rho,
            "half_life_days": active_half_life,
            "home_weighted_scored": round(home_scored_w, 2),
            "away_weighted_scored": round(away_scored_w, 2),
            "league_home_baseline": round(league_home_baseline, 2),
            "league_away_baseline": round(league_away_baseline, 2),
        }

        comp_name = fixture.get("competition_name") if fixture else None

        return derive_markets_from_score_matrix(
            score_matrix=score_matrix,
            model_name=self.name,
            confidence=confidence,
            confidence_score=confidence_score,
            notes=notes,
            features=features,
            competition_name=comp_name,
        )
