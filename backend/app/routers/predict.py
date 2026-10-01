import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict

from app.config import settings
from app.db import get_db
from app.models import Prediction
from app.schemas import PredictRequest, PredictResponse
from app.dependencies import get_model, get_shortlist, get_medians, get_baseline_distribution
from app.services.preprocessing import impute_sensor_dict
from app.services.scoring import score_single_prediction

router = APIRouter(tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
def predict_single_wafer(
    body: PredictRequest,
    db: Session = Depends(get_db),
    model = Depends(get_model),
    shortlist_cols: List[str] = Depends(get_shortlist),
    medians: Dict[str, float] = Depends(get_medians),
    baseline_dist: List[float] = Depends(get_baseline_distribution)
):
    # 1. Impute missing sensor values
    imputed_dict, imputed_list = impute_sensor_dict(body.sensor_values, shortlist_cols, medians)

    # 2. Form contiguous numpy array in shortlist feature order
    X_row = np.array([[imputed_dict[c] for c in shortlist_cols]], dtype=np.float64)
    X_row = np.ascontiguousarray(X_row)

    # 3. Model Inference
    try:
        proba = float(model.predict_proba(X_row)[0, 1])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )

    # 4. Score relative to test batch distribution
    scoring_result = score_single_prediction(proba, baseline_dist, capacity_k=settings.INSPECTION_CAPACITY_K)

    # 5. Persist to DB
    pred_record = Prediction(
        upload_id=None,
        row_index=None,
        sensor_values=imputed_dict,
        imputed_sensors=imputed_list,
        fail_probability=scoring_result["fail_probability"],
        percentile_rank=scoring_result["percentile_rank"],
        risk_level=scoring_result["risk_level"],
        is_flagged_top_k=scoring_result["is_flagged_top_k"],
        model_version=settings.MODEL_VERSION
    )
    db.add(pred_record)
    db.commit()
    db.refresh(pred_record)

    return PredictResponse(
        prediction_id=pred_record.id,
        fail_probability=pred_record.fail_probability,
        percentile_rank=pred_record.percentile_rank,
        percentile_context="relative_to_test_batch",
        risk_level=pred_record.risk_level,
        is_flagged_top_k=pred_record.is_flagged_top_k,
        imputed_sensors=pred_record.imputed_sensors,
        model_version=pred_record.model_version,
        created_at=pred_record.created_at
    )
