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
