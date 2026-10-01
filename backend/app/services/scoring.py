from typing import List, Dict, Tuple, Any
import numpy as np

def score_batch_predictions(
    probabilities: List[float],
    capacity_k: int
) -> List[Dict[str, Any]]:
    """
    Ranks a batch of wafer predictions, computes percentiles, flags top K, and assigns risk levels.

    Risk rules:
        HIGH = Flagged top K (rank <= min(K, N_samples))
        MEDIUM = Non-flagged rows above 50th percentile of the batch
        LOW = Rows at or below 50th percentile of the batch
    """
    n_samples = len(probabilities)
    if n_samples == 0:
        return []

    # Get descending rank order indices
    prob_array = np.array(probabilities, dtype=float)
    sorted_indices = np.argsort(-prob_array)

    results = [None] * n_samples
    effective_k = min(capacity_k, n_samples)

    for rank, orig_idx in enumerate(sorted_indices, start=1):
        proba = float(prob_array[orig_idx])
        # Percentile rank: 100 for top row down to 100/N for bottom row
        percentile = float((1.0 - (rank - 1) / n_samples) * 100.0)
        is_flagged = rank <= effective_k

        if is_flagged:
            risk_level = "HIGH"
        elif percentile > 50.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        results[orig_idx] = {
            "rank": rank,
            "fail_probability": proba,
            "percentile_rank": round(percentile, 2),
            "is_flagged_top_k": is_flagged,
            "risk_level": risk_level
        }

    return results


def score_single_prediction(
    probability: float,
    baseline_distribution: List[float],
    capacity_k: int = 30
) -> Dict[str, Any]:
    """
    Scores a single prediction by comparing its probability against the saved baseline test-batch distribution.
    
    Returns:
        Dict with fail_probability, percentile_rank, is_flagged_top_k, risk_level.
    """
    baseline_array = np.array(baseline_distribution, dtype=float)
    n_baseline = len(baseline_array)
    
    if n_baseline == 0:
        percentile = 50.0
        cutoff_percentile = 90.0
    else:
        # Percentile relative to baseline test batch
        count_below = np.sum(baseline_array <= probability)
        percentile = float((count_below / n_baseline) * 100.0)
        cutoff_percentile = float((1.0 - capacity_k / n_baseline) * 100.0)

    is_flagged = percentile >= cutoff_percentile
    if is_flagged:
        risk_level = "HIGH"
    elif percentile >= 50.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fail_probability": float(probability),
        "percentile_rank": round(percentile, 2),
        "is_flagged_top_k": is_flagged,
        "risk_level": risk_level
    }
