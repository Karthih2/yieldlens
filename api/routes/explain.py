#api/routes/explain.py

import shap
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from schemas import ExplainResponse
from routes.predict import _prediction_log, shortlist, model  # reuse loaded model + log

router = APIRouter(tags=["explain"])

explainer = shap.TreeExplainer(model)

# Pre-computed SHAP values for the fixed test set (from explain_shap.py),
# used to answer /explain/{id} for ids that came from the offline test set
# rather than a live /predict call.
_shap_test_df = pd.read_csv("../reports/shap_per_prediction.csv")


@router.get("/explain/{row_id}", response_model=ExplainResponse)
def explain(row_id: int):
    # Case 1: row_id came from a live /predict call this session
    if row_id in _prediction_log:
        X = _prediction_log[row_id]
        shap_values = explainer.shap_values(X)
        vals = shap_values[:, :, 1] if getattr(shap_values, "ndim", 0) == 3 else (
            shap_values[1] if isinstance(shap_values, list) else shap_values
        )
        shap_dict = dict(zip(shortlist, vals[0].tolist()))
        actual_label = None

    # Case 2: row_id refers to a pre-computed row from the offline test set
    elif row_id < len(_shap_test_df):
        row = _shap_test_df.iloc[row_id]
        shap_dict = {s: float(row[s]) for s in shortlist}
        actual_label = int(row["actual_label"])

    else:
        raise HTTPException(status_code=404, detail=f"row_id {row_id} not found")

    top_contributors = sorted(
        [{"sensor": s, "shap_value": v} for s, v in shap_dict.items()],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )[:5]

    return ExplainResponse(
        row_id=row_id,
        actual_label=actual_label,
        shap_values=shap_dict,
        top_contributors=top_contributors,
    )