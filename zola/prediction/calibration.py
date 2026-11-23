"""Probability calibration, scoring metrics, and reliability assessment."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    """Calculate binary Brier score: (1/N) * sum((p_i - y_i)^2).
    
    Range: [0.0, 1.0]. Lower is better. 0.0 is perfect calibration and sharpness.
    """
    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return 0.0
    total = sum((p - y) ** 2 for y, p in zip(y_true, y_prob))
    return round(total / len(y_true), 4)


def multiclass_brier_score(y_true_indices: List[int], prob_matrix: List[List[float]]) -> float:
    """Calculate multiclass Brier score across K classes:
    (1/N) * sum_i sum_k ((p_ik - I(y_i == k))^2).
    """
    if not y_true_indices or not prob_matrix or len(y_true_indices) != len(prob_matrix):
        return 0.0
    total = 0.0
    for true_idx, probs in zip(y_true_indices, prob_matrix):
        for k, p in enumerate(probs):
            y_ik = 1.0 if k == true_idx else 0.0
            total += (p - y_ik) ** 2
    return round(total / len(y_true_indices), 4)


def log_loss_score(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:
    """Calculate binary log loss (cross-entropy)."""
    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return 0.0
    total = 0.0
    for y, p in zip(y_true, y_prob):
        p_clamped = max(eps, min(1.0 - eps, p))
        total += y * math.log(p_clamped) + (1 - y) * math.log(1.0 - p_clamped)
    return round(-total / len(y_true), 4)


def expected_calibration_error(
    y_true: List[int],
    y_prob: List[float],
    bin_edges: Optional[List[float]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculate Expected Calibration Error (ECE) and reliability bin details.
    
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return 0.0, []

    if bin_edges is None:
        bin_edges = [0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]

    n = len(y_true)
    bins_data: List[Dict[str, Any]] = []
    total_weighted_error = 0.0

    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        # Filter items in bin [low, high) or [low, high] for the last bin
        bin_indices = [
            idx for idx, p in enumerate(y_prob)
            if (low <= p < high) or (i == len(bin_edges) - 2 and low <= p <= high)
        ]

        count = len(bin_indices)
        if count == 0:
            bins_data.append({
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "bin_label": f"{int(low * 100)}%-{int(high * 100)}%",
                "count": 0,
                "mean_confidence": 0.0,
                "observed_accuracy": 0.0,
                "calibration_error": 0.0,
            })
            continue

        mean_conf = sum(y_prob[idx] for idx in bin_indices) / count
        obs_acc = sum(y_true[idx] for idx in bin_indices) / count
        error = abs(obs_acc - mean_conf)

        total_weighted_error += (count / n) * error

        bins_data.append({
            "bin_low": round(low, 2),
            "bin_high": round(high, 2),
            "bin_label": f"{int(low * 100)}%-{int(high * 100)}%",
            "count": count,
            "mean_confidence": round(mean_conf, 4),
            "observed_accuracy": round(obs_acc, 4),
            "calibration_error": round(error, 4),
        })

    return round(total_weighted_error, 4), bins_data


class PlattScaler:
    """Logistic calibration scaling (Platt scaling) for probability calibration."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b
        self.is_fitted = False

    def fit(self, y_true: List[int], y_prob: List[float], lr: float = 0.05, max_iter: int = 150) -> PlattScaler:
        """Fit parameters a and b using gradient descent on binary cross-entropy."""
        if not y_true or not y_prob or len(y_true) < 10:
            self.is_fitted = True
            return self

        a = 1.0
        b = 0.0
        n = len(y_true)
        eps = 1e-6

        # Convert probs to logits
        logits = []
        for p in y_prob:
            p_c = max(eps, min(1.0 - eps, p))
            logits.append(math.log(p_c / (1.0 - p_c)))

        for _ in range(max_iter):
            grad_a = 0.0
            grad_b = 0.0
            for y, logit in zip(y_true, logits):
                # Sigmoid of (a * logit + b)
                z = a * logit + b
                z_clamped = max(-20.0, min(20.0, z))
                p_hat = 1.0 / (1.0 + math.exp(-z_clamped))
                err = p_hat - y
                grad_a += err * logit
                grad_b += err

            a -= lr * (grad_a / n)
            b -= lr * (grad_b / n)

        self.a = round(a, 4)
        self.b = round(b, 4)
        self.is_fitted = True
        return self

    def transform(self, prob: float) -> float:
        """Calibrate a single probability value."""
        if not self.is_fitted:
            return prob
        eps = 1e-6
        p_c = max(eps, min(1.0 - eps, prob))
        logit = math.log(p_c / (1.0 - p_c))
        z = self.a * logit + self.b
        z_clamped = max(-20.0, min(20.0, z))
        calibrated = 1.0 / (1.0 + math.exp(-z_clamped))
        return round(max(0.01, min(0.99, calibrated)), 4)


class TemperatureScaler:
    """Scales logits by temperature T for softmax/distribution calibration."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.1, temperature)

    def calibrate_distribution(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Apply temperature scaling to a probability distribution."""
        if abs(self.temperature - 1.0) < 1e-4 or not probs:
            return dict(probs)

        eps = 1e-6
        exp_logits = {}
        for outcome, p in probs.items():
            p_c = max(eps, min(1.0 - eps, p))
            logit = math.log(p_c) / self.temperature
            exp_logits[outcome] = math.exp(max(-30.0, min(30.0, logit)))

        sum_exp = sum(exp_logits.values()) or 1.0
        return {
            outcome: round(val / sum_exp, 4)
            for outcome, val in exp_logits.items()
        }


# Subtype approval status based on empirical out-of-sample audit
# OU05 and OU15 have verified positive empirical ROI across in-sample and out-of-sample.
# DOUBLE_CHANCE subtypes (1X, X2, 12) have negative out-of-sample ROI (-13.5%) and are blocked.
APPROVED_MARKET_SUBTYPES: Dict[str, set[str]] = {
    "OU05": {"OVER_05"},
    "OU15": {"OVER_15"},
    "DOUBLE_CHANCE": set(),  # All subtypes suspended due to out-of-sample negative ROI
}

SUBTYPE_SUSPENSION_REASONS: Dict[str, str] = {
    "1X": "Suspended: Out-of-sample ROI is -13.5% (N=45) with severe probability overconfidence (88% model prob delivered 79% win rate).",
    "X2": "Suspended: Sample size insufficient (N=35) and out-of-sample ROI collapsed from +23.5% in-sample to -13.5% out-of-sample (N=20).",
    "12": "Suspended: Empirical ROI is negative (-0.65%, N=153) and market carries high draw-variance exposure.",
}


def is_subtype_approved(market: str, subtype: str) -> Tuple[bool, str]:
    """Check whether a specific market outcome subtype is approved for high-confidence selection."""
    if market not in APPROVED_MARKET_SUBTYPES:
        return False, f"Market '{market}' is not in approved high-confidence allowlist."

    approved_set = APPROVED_MARKET_SUBTYPES[market]
    if subtype in approved_set:
        return True, "Approved"

    reason = SUBTYPE_SUSPENSION_REASONS.get(
        subtype,
        f"Subtype '{subtype}' of market '{market}' is not approved for high-confidence selection.",
    )
    return False, reason


def compute_expected_value(probability: float, market_odd: Optional[float]) -> Optional[float]:
    """Calculate decimal Expected Value (EV = p * odd - 1.0).
    
    Returns None if odds are missing, non-positive, or invalid (<= 1.0).
    """
    if market_odd is None or not isinstance(market_odd, (int, float)):
        return None
    if math.isnan(market_odd) or market_odd <= 1.0:
        return None
    if probability is None or not (0.0 <= probability <= 1.0):
        return None

    return (probability * market_odd) - 1.0


def compute_roi_and_pnl(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute exact Wins, Losses, Pushes, Net PnL, ROI, and Win Rates with correct push handling.
    
    Push bets (e.g. Draw No Bet draw) return the 1-unit stake with exactly 0 profit and 0 loss.
    """
    total = len(records)
    if total == 0:
        return {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "decided_bets": 0,
            "win_rate_decided": 0.0,
            "win_rate_total": 0.0,
            "pnl": 0.0,
            "roi": 0.0,
            "avg_odd": 0.0,
        }

    wins = 0
    losses = 0
    pushes = 0
    pnl = 0.0
    odds_sum = 0.0
    valid_odds_count = 0

    for r in records:
        status = str(r.get("settlement_status") or "").upper()
        is_corr = r.get("is_correct")
        odd = r.get("odd")

        if odd is not None and isinstance(odd, (int, float)) and odd > 1.0:
            odds_sum += odd
            valid_odds_count += 1
        else:
            odd = 1.0

        if status == "WIN" or is_corr == 1:
            wins += 1
            pnl += (odd - 1.0)
        elif status == "LOSS" or is_corr == 0:
            losses += 1
            pnl -= 1.0
        elif status == "PUSH":
            pushes += 1
            # Stake refunded, 0 profit/loss
        elif status == "VOID":
            # Wager cancelled/voided, 0 profit/loss
            pass

    decided = wins + losses
    win_rate_decided = (wins / decided * 100.0) if decided > 0 else 0.0
    win_rate_total = (wins / total * 100.0) if total > 0 else 0.0
    roi = (pnl / total * 100.0) if total > 0 else 0.0
    avg_odd = (odds_sum / valid_odds_count) if valid_odds_count > 0 else 0.0

    return {
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "decided_bets": decided,
        "win_rate_decided": round(win_rate_decided, 2),
        "win_rate_total": round(win_rate_total, 2),
        "pnl": round(pnl, 2),
        "roi": round(roi, 2),
        "avg_odd": round(avg_odd, 3),
    }


def compute_calibration_bins(
    records: List[Dict[str, Any]],
    bins: Optional[List[Tuple[float, float, str]]] = None,
) -> List[Dict[str, Any]]:
    """Partition prediction records into probability bins and evaluate empirical calibration."""
    if bins is None:
        bins = [
            (0.50, 0.5999, "50-59%"),
            (0.60, 0.6499, "60-64%"),
            (0.65, 0.6999, "65-69%"),
            (0.70, 0.7499, "70-74%"),
            (0.75, 0.7999, "75-79%"),
            (0.80, 0.8499, "80-84%"),
            (0.85, 0.8999, "85-89%"),
            (0.90, 1.0000, "90%+"),
        ]

    results: List[Dict[str, Any]] = []

    for b_low, b_high, label in bins:
        in_bin = [
            r for r in records
            if r.get("probability") is not None and b_low <= float(r["probability"]) <= b_high
        ]
        n = len(in_bin)
        if n == 0:
            results.append({
                "bin_label": label,
                "bin_range": (b_low, b_high),
                "count": 0,
                "mean_predicted_prob": 0.0,
                "actual_win_rate": 0.0,
                "calibration_error": 0.0,
                "avg_odd": 0.0,
                "pnl": 0.0,
                "roi": 0.0,
            })
            continue

        pred_probs = [float(r["probability"]) for r in in_bin]
        mean_prob = sum(pred_probs) / n

        perf = compute_roi_and_pnl(in_bin)
        actual_wr = perf["win_rate_decided"] / 100.0
        calib_error = abs(mean_prob - actual_wr)

        results.append({
            "bin_label": label,
            "bin_range": (b_low, b_high),
            "count": n,
            "mean_predicted_prob": round(mean_prob * 100.0, 2),
            "actual_win_rate": round(actual_wr * 100.0, 2),
            "calibration_error": round(calib_error * 100.0, 2),
            "avg_odd": perf["avg_odd"],
            "pnl": perf["pnl"],
            "roi": perf["roi"],
        })

    return results
