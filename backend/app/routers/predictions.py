from typing import Optional, List
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Prediction, SHAPExplanation
from app.schemas import PredictionsListResponse, PredictionItemSchema, ExplainResponse, ContributorSchema
from app.dependencies import get_explainer, get_shortlist
from app.services.explainer import compute_shap_explanation

router = APIRouter(tags=["predictions"])

@router.get("/predictions", response_model=PredictionsListResponse)
def list_predictions(
    upload_id: Optional[int] = Query(None, description="Filter by upload ID"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (HIGH, MEDIUM, LOW)"),
    is_flagged_top_k: Optional[bool] = Query(None, description="Filter by top-K flagged status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    query = db.query(Prediction)
    if upload_id is not None:
        query = query.filter(Prediction.upload_id == upload_id)
    if risk_level is not None:
        query = query.filter(Prediction.risk_level == risk_level.upper())
    if is_flagged_top_k is not None:
        query = query.filter(Prediction.is_flagged_top_k == is_flagged_top_k)

    total = query.count()
    offset = (page - 1) * page_size
    items = query.order_by(Prediction.created_at.desc()).offset(offset).limit(page_size).all()

    pred_schemas = [
        PredictionItemSchema(
            id=p.id,
            upload_id=p.upload_id,
            row_index=p.row_index,
            fail_probability=p.fail_probability,
            percentile_rank=p.percentile_rank,
            risk_level=p.risk_level,
            is_flagged_top_k=p.is_flagged_top_k,
            imputed_sensors=p.imputed_sensors,
            model_version=p.model_version,
            created_at=p.created_at,
            sensor_values=p.sensor_values
        )
        for p in items
    ]

    return PredictionsListResponse(
        total=total,
        page=page,
        page_size=page_size,
        predictions=pred_schemas
    )


@router.get("/predictions/{prediction_id}/explain", response_model=ExplainResponse)
def explain_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    explainer = Depends(get_explainer),
    shortlist_cols: List[str] = Depends(get_shortlist)
):
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    # 1. Check if SHAP explanation is already cached in DB
    existing_shap = db.query(SHAPExplanation).filter(SHAPExplanation.prediction_id == prediction_id).first()
    if existing_shap:
        contribs = [
            ContributorSchema(sensor=c["sensor"], shap_value=c["shap_value"])
            for c in existing_shap.top_contributors
        ]
        return ExplainResponse(
            prediction_id=pred.id,
            row_index=pred.row_index,
            shap_values=existing_shap.shap_values,
            top_contributors=contribs
        )

    # 2. Compute SHAP dynamically if not cached
    sensor_vals = pred.sensor_values
    X_row = np.array([[sensor_vals[c] for c in shortlist_cols]], dtype=np.float64)
    X_row = np.ascontiguousarray(X_row)

    shap_results = compute_shap_explanation(explainer, X_row, shortlist_cols)
    shap_dict, top_contrib = shap_results[0]

    # 3. Cache computed SHAP explanation in DB
    new_shap_rec = SHAPExplanation(
        prediction_id=pred.id,
        shap_values=shap_dict,
        top_contributors=top_contrib
    )
    db.add(new_shap_rec)
    db.commit()
    db.refresh(new_shap_rec)

    contribs = [
        ContributorSchema(sensor=c["sensor"], shap_value=c["shap_value"])
        for c in top_contrib
    ]

    return ExplainResponse(
        prediction_id=pred.id,
        row_index=pred.row_index,
        shap_values=shap_dict,
        top_contributors=contribs
    )
