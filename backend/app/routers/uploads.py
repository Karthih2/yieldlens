import io
from typing import List, Dict
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Upload, Prediction, SHAPExplanation
from app.schemas import UploadSummary, UploadDetailResponse, PredictionItemSchema
from app.dependencies import get_model, get_explainer, get_shortlist, get_medians
from app.services.preprocessing import impute_sensor_dataframe
from app.services.scoring import score_batch_predictions
from app.services.explainer import compute_shap_explanation

router = APIRouter(tags=["uploads"])

@router.post("/uploads", response_model=UploadDetailResponse, status_code=status.HTTP_201_CREATED)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    model = Depends(get_model),
    explainer = Depends(get_explainer),
    shortlist_cols: List[str] = Depends(get_shortlist),
    medians: Dict[str, float] = Depends(get_medians)
):
    # 1. Size Limit Check
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB"
        )

    # 2. Parse CSV
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid CSV file format: {str(e)}"
        )

    if len(df_raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded CSV file is empty"
        )

    # 3. Shortlist Column Validation
    missing_cols = [col for col in shortlist_cols if col not in df_raw.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required shortlist sensor columns: {missing_cols}"
        )

    # 4. Check for non-numeric corrupt values
    for col in shortlist_cols:
        non_numeric = pd.to_numeric(df_raw[col], errors='coerce').isna() & df_raw[col].notna()
        if non_numeric.any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Column '{col}' contains non-numeric invalid values"
            )

    # 5. Impute Missing Telemetry Values
    clean_df, per_row_imputed = impute_sensor_dataframe(df_raw, shortlist_cols, medians)
    X_mat = np.ascontiguousarray(clean_df[shortlist_cols].to_numpy(dtype=np.float64))

    # 6. Model Batch Inference
    try:
        probas = model.predict_proba(X_mat)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch model prediction failed: {str(e)}"
        )

    # 7. Score & Rank Predictions
    scored_rows = score_batch_predictions(probas, capacity_k=settings.INSPECTION_CAPACITY_K)
    flagged_count = sum(1 for r in scored_rows if r["is_flagged_top_k"])

    # 8. Create Upload DB Record
    upload_record = Upload(
        filename=file.filename,
        row_count=len(clean_df),
        flagged_count=flagged_count,
        top_k=settings.INSPECTION_CAPACITY_K
    )
    db.add(upload_record)
    db.flush()

    # 9. Create Prediction Records & SHAP for Flagged Top-K Rows ONLY
    pred_records = []
    top_k_indices = []
    top_k_X = []

    for r_idx, scored in enumerate(scored_rows):
        row_sensor_dict = {col: float(clean_df.iloc[r_idx][col]) for col in shortlist_cols}
        pred_rec = Prediction(
            upload_id=upload_record.id,
            row_index=r_idx,
            sensor_values=row_sensor_dict,
            imputed_sensors=per_row_imputed[r_idx],
            fail_probability=scored["fail_probability"],
            percentile_rank=scored["percentile_rank"],
            risk_level=scored["risk_level"],
            is_flagged_top_k=scored["is_flagged_top_k"],
            model_version=settings.MODEL_VERSION
        )
        db.add(pred_rec)
        pred_records.append(pred_rec)
        
        if scored["is_flagged_top_k"]:
            top_k_indices.append(r_idx)
            top_k_X.append(X_mat[r_idx])

    db.flush()

    # Compute SHAP only for top-K flagged rows at upload time
    if top_k_X:
        top_k_mat = np.ascontiguousarray(np.array(top_k_X, dtype=np.float64))
        shap_results = compute_shap_explanation(explainer, top_k_mat, shortlist_cols)
        
        for idx_in_top, orig_row_idx in enumerate(top_k_indices):
            shap_dict, top_contrib = shap_results[idx_in_top]
            target_pred_rec = pred_records[orig_row_idx]
            
            shap_rec = SHAPExplanation(
                prediction_id=target_pred_rec.id,
                shap_values=shap_dict,
                top_contributors=top_contrib
            )
            db.add(shap_rec)

    db.commit()
    db.refresh(upload_record)
    for p in pred_records:
        db.refresh(p)

    upload_summary = UploadSummary(
        id=upload_record.id,
        filename=upload_record.filename,
        row_count=upload_record.row_count,
        flagged_count=upload_record.flagged_count,
        top_k=upload_record.top_k,
        created_at=upload_record.created_at
    )

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
        for p in pred_records
    ]

    return UploadDetailResponse(upload=upload_summary, predictions=pred_schemas)


@router.get("/uploads", response_model=List[UploadSummary])
def list_uploads(db: Session = Depends(get_db)):
    uploads = db.query(Upload).order_by(Upload.created_at.desc()).all()
    return [
        UploadSummary(
            id=u.id,
            filename=u.filename,
            row_count=u.row_count,
            flagged_count=u.flagged_count,
            top_k=u.top_k,
            created_at=u.created_at
        )
        for u in uploads
    ]


@router.get("/uploads/{upload_id}", response_model=UploadDetailResponse)
def get_upload_detail(upload_id: int, db: Session = Depends(get_db)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")

    predictions = db.query(Prediction).filter(Prediction.upload_id == upload_id).order_by(Prediction.row_index.asc()).all()

    upload_summary = UploadSummary(
        id=upload.id,
        filename=upload.filename,
        row_count=upload.row_count,
        flagged_count=upload.flagged_count,
        top_k=upload.top_k,
        created_at=upload.created_at
    )

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
        for p in predictions
    ]

    return UploadDetailResponse(upload=upload_summary, predictions=pred_schemas)
