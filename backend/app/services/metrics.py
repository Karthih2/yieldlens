import numpy as np

def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """
    Computes Precision@K: Proportion of actual failures in the top-K highest risk wafers.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) == 0:
        return 0.0
    effective_k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:effective_k]
    return float(y_true[top_k_idx].sum() / effective_k)


def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """
    Computes Recall@K: Proportion of total actual failures caught in top-K highest risk wafers.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    total_fails = y_true.sum()
    if total_fails == 0:
        return 0.0
    effective_k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:effective_k]
    return float(y_true[top_k_idx].sum() / total_fails)
