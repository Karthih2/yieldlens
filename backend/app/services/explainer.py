from typing import List, Dict, Tuple, Any
import numpy as np
import shap

def compute_shap_explanation(
    explainer: shap.TreeExplainer,
    X_array: np.ndarray,
    shortlist_cols: List[str]
) -> List[Tuple[Dict[str, float], List[Dict[str, Any]]]]:
    """
    Computes SHAP explanations for 1 or more sample rows using TreeExplainer.
    Handles binary classification output variations across SHAP versions (list, 2D, 3D).

    Args:
        explainer: Initialized shap.TreeExplainer
        X_array: numpy 2D array of shape (N_samples, N_features)
        shortlist_cols: List of sensor names in feature order

    Returns:
        List of tuples: (shap_dict, top_contributors) for each sample row.
    """
    raw_shap = explainer.shap_values(X_array)
    
    # Extract fail class (class index 1) SHAP matrix of shape (N_samples, N_features)
    if isinstance(raw_shap, list):
        # List of arrays [class_0_shap, class_1_shap]
        fail_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    elif getattr(raw_shap, "ndim", 0) == 3:
        # 3D array of shape (N_samples, N_features, N_classes)
        fail_shap = raw_shap[:, :, 1]
    else:
        # 2D array of shape (N_samples, N_features)
        fail_shap = raw_shap

    fail_shap = np.array(fail_shap, dtype=float)

    results = []
    for row_idx in range(len(X_array)):
        row_vals = fail_shap[row_idx]
        shap_dict = {col: float(row_vals[i]) for i, col in enumerate(shortlist_cols)}
        
        # Top contributors sorted by magnitude |SHAP value|
        contributors = sorted(
            [{"sensor": col, "shap_value": float(row_vals[i])} for i, col in enumerate(shortlist_cols)],
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )
        results.append((shap_dict, contributors[:5]))

    return results
