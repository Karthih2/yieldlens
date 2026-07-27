#api/routes/predict.py

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])

# Load once at import time, not per-request
MODEL_PATH = "../models/random_forest_reduced_shortlist.pkl"
SHORTLIST_PATH = "../reports/shortlist_sensors.csv"

model = joblib.load(MODEL_PATH)
shortlist = pd.read_csv(SHORTLIST_PATH)["sensor"].tolist()

# Simple in-memory counter to assign row_ids, so /explain/{id} can reference
# predictions made in this session. Resets on server restart — fine for a demo/POC.
_prediction_log = {}
_next_row_id = 0


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    global _next_row_id

    missing = [s for s in shortlist if s not in request.sensor_values]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required sensor values: {missing}"
        )

    X = np.array([[request.sensor_values[s] for s in shortlist]], dtype=np.float64)
    X = np.ascontiguousarray(X)

    proba = model.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)

    row_id = _next_row_id
    _next_row_id += 1
    _prediction_log[row_id] = X  # stored for /explain/{id} to reuse

    return PredictResponse(
        fail_probability=float(proba),
        predicted_label=label,
        row_id=row_id,
    )