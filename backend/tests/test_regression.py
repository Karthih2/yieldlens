import pytest
import pandas as pd
import numpy as np
import shap
import joblib

from app.config import settings
from app.services.metrics import recall_at_k
from app.services.explainer import compute_shap_explanation

def test_regression_recall_at_30_and_shap():
    artifacts_dir = settings.ARTIFACTS_DIR
    sample_path = artifacts_dir / "sample_rows.csv"
    model_path = artifacts_dir / "random_forest_reduced_shortlist.pkl"
    shortlist_path = artifacts_dir / "shortlist_sensors.csv"
    shap_path = artifacts_dir / "shap_per_prediction.csv"

    assert sample_path.exists(), "sample_rows.csv missing"
    assert model_path.exists(), "model file missing"
    assert shortlist_path.exists(), "shortlist_sensors.csv missing"
    assert shap_path.exists(), "shap_per_prediction.csv missing"

    # 1. Load data & model
    sample_df = pd.read_csv(sample_path)
    shortlist_cols = pd.read_csv(shortlist_path)["sensor"].tolist()
    model = joblib.load(model_path)
    y_true = sample_df["labels"].to_numpy()

    # 2. Predict probabilities
    X_mat = np.ascontiguousarray(sample_df[shortlist_cols].to_numpy(dtype=np.float64))
    y_proba = model.predict_proba(X_mat)[:, 1]

    # 3. Calculate Recall@30
    rec_30 = recall_at_k(y_true, y_proba, k=30)
    print(f"Regression Recall@30 = {rec_30:.4f}")

    # Assert Recall@30 matches 0.353 within 0.001 tolerance
    assert abs(rec_30 - 0.353) < 0.001, f"Recall@30 mismatch: expected 0.353, got {rec_30:.4f}"

    # 4. Compute SHAP values dynamically and compare against saved shap_per_prediction.csv
    explainer = shap.TreeExplainer(model)
    shap_results = compute_shap_explanation(explainer, X_mat, shortlist_cols)

    shap_df_expected = pd.read_csv(shap_path)
    assert len(shap_df_expected) == len(sample_df), (
        f"Row count mismatch: expected SHAP {len(shap_df_expected)}, got {len(sample_df)}"
    )

    # Verify SHAP value agreement for all shortlist columns
    max_diff = 0.0
    for row_idx, (shap_dict, _) in enumerate(shap_results):
        exp_row = shap_df_expected.iloc[row_idx]
        for col in shortlist_cols:
            val_computed = shap_dict[col]
            val_expected = float(exp_row[col])
            diff = abs(val_computed - val_expected)
            if diff > max_diff:
                max_diff = diff

    print(f"Max absolute SHAP difference across {len(sample_df)} test rows: {max_diff:.6f}")
    assert max_diff < 1e-4, f"SHAP value regression discrepancy too large: max diff = {max_diff}"
