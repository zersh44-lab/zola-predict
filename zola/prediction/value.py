"""Market value analysis, odds-to-fair probability, and Expected Value (EV) computation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def implied_probability(odds: float) -> float:
    """Convert decimal bookmaker odds to implied probability."""
    if odds <= 1.0:
        return 1.0
    return round(1.0 / odds, 4)


def remove_overround_proportional(odds_dict: Dict[str, float]) -> Dict[str, float]:
    """Remove bookmaker overround via proportional normalization.
    
    P_fair_i = (1 / odds_i) / sum_j(1 / odds_j)
    """
    if not odds_dict:
        return {}
    implied = {k: 1.0 / max(1.001, o) for k, o in odds_dict.items()}
    total_implied = sum(implied.values()) or 1.0
    return {k: round(p / total_implied, 4) for k, p in implied.items()}


def remove_overround_shin(odds_dict: Dict[str, float], max_iter: int = 50) -> Dict[str, float]:
    """Remove bookmaker overround using Shin (1993) model.
    
    Accounts for the favorite-longshot bias by solving for the insider trading parameter z.
    """
    if not odds_dict:
        return {}
    implied = {k: 1.0 / max(1.001, o) for k, o in odds_dict.items()}
    overround = sum(implied.values()) - 1.0
    
    if overround <= 0.001:
        return remove_overround_proportional(odds_dict)

    keys = list(odds_dict.keys())
    # Bisection search for z in [0, overround]
    low = 0.0
    high = min(0.35, max(0.01, overround))
    z = (low + high) / 2.0

    for _ in range(max_iter):
        z = (low + high) / 2.0
        # Shin probability: P_i = (sqrt(z^2 + 4 * (1 - z) * (pi_i^2 / sum_pi)) - z) / (2 * (1 - z))
        probs = []
        for k in keys:
            pi_i = implied[k]
            val = (z ** 2) + (4.0 * (1.0 - z) * (pi_i ** 2) / (1.0 + overround))
            val = max(0.0, val)
            p_i = (math.sqrt(val) - z) / max(1e-6, 2.0 * (1.0 - z))
            probs.append(p_i)
        
        sum_p = sum(probs)
        if abs(sum_p - 1.0) < 1e-4:
            break
        elif sum_p > 1.0:
            low = z
        else:
            high = z

    # Normalize final result
    shin_probs = {}
    for k in keys:
        pi_i = implied[k]
        val = (z ** 2) + (4.0 * (1.0 - z) * (pi_i ** 2) / (1.0 + overround))
        val = max(0.0, val)
        shin_probs[k] = max(0.001, (math.sqrt(val) - z) / max(1e-6, 2.0 * (1.0 - z)))
    
    tot = sum(shin_probs.values()) or 1.0
    return {k: round(v / tot, 4) for k, v in shin_probs.items()}


def calculate_expected_value(model_probability: float, odds: float) -> float:
    """Calculate Expected Value (EV): EV = (Probability * Odds) - 1.0."""
    if odds <= 0.0:
        return -1.0
    return round((model_probability * odds) - 1.0, 4)


def classify_value_opportunity(ev: float) -> Tuple[str, str]:
    """Classify betting value opportunity tier based on expected value."""
    if ev >= 0.15:
        return "HIGH_VALUE", f"High Value Edge (+{ev * 100:.1f}% EV)"
    elif ev >= 0.08:
        return "MODERATE_VALUE", f"Moderate Value (+{ev * 100:.1f}% EV)"
    elif ev >= 0.05:
        return "SLIGHT_VALUE", f"Slight Value (+{ev * 100:.1f}% EV)"
    elif ev > 0.0:
        return "MARGINAL_VALUE", f"Marginal Edge (+{ev * 100:.1f}% EV)"
    else:
        return "FAIR_OR_NEGATIVE", f"No Value ({ev * 100:+.1f}% EV)"


def analyze_market_value(
    market: str,
    predicted_outcome: str,
    model_probabilities: Dict[str, float],
    bookmaker_odds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate market value comparing model probabilities against bookmaker odds."""
    if not bookmaker_odds:
        return {
            "status": "UNAVAILABLE",
            "message": "VALUE ANALYSIS UNAVAILABLE — ODDS NOT PROVIDED",
            "market": market,
            "has_value": False,
            "best_ev": None,
            "recommendation": None,
        }

    # Verify matching outcomes
    available_keys = [k for k in bookmaker_odds.keys() if k in model_probabilities]
    if not available_keys:
        return {
            "status": "UNAVAILABLE",
            "message": "VALUE ANALYSIS UNAVAILABLE — ODDS DO NOT MATCH MARKET OUTCOMES",
            "market": market,
            "has_value": False,
            "best_ev": None,
            "recommendation": None,
        }

    fair_market_probs = remove_overround_shin(bookmaker_odds)

    outcomes_analysis: Dict[str, Dict[str, Any]] = {}
    best_outcome = None
    max_ev = -999.0

    for outcome in available_keys:
        m_prob = model_probabilities[outcome]
        odds = bookmaker_odds[outcome]
        implied_p = implied_probability(odds)
        fair_p = fair_market_probs.get(outcome, implied_p)
        ev = calculate_expected_value(m_prob, odds)
        edge = round(m_prob - fair_p, 4)
        tier, label = classify_value_opportunity(ev)

        outcomes_analysis[outcome] = {
            "model_probability": m_prob,
            "bookmaker_odds": odds,
            "implied_probability": implied_p,
            "fair_market_probability": fair_p,
            "expected_value": ev,
            "edge": edge,
            "value_tier": tier,
            "tier_label": label,
        }

        if ev > max_ev:
            max_ev = ev
            best_outcome = outcome

    has_value = max_ev >= 0.05
    best_info = outcomes_analysis.get(best_outcome, {})

    return {
        "status": "AVAILABLE",
        "market": market,
        "has_value": has_value,
        "best_value_outcome": best_outcome,
        "best_ev": max_ev,
        "best_tier": best_info.get("value_tier"),
        "best_label": best_info.get("tier_label"),
        "outcomes": outcomes_analysis,
    }
