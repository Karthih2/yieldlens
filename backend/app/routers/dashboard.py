from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.config import settings
from app.db import get_db
from app.models import Upload, Prediction
from app.schemas import DashboardResponse, UploadSummary

router = APIRouter(tags=["dashboard"])

@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard_summary(db: Session = Depends(get_db)):
    total_preds = db.query(Prediction).count()
    total_uploads = db.query(Upload).count()
    high_risk_count = db.query(Prediction).filter(Prediction.risk_level == "HIGH").count()
    
    high_risk_ratio = float(high_risk_count / total_preds) if total_preds > 0 else 0.0

    recent_uploads_raw = db.query(Upload).order_by(Upload.created_at.desc()).limit(5).all()
    recent_uploads = [
        UploadSummary(
            id=u.id,
            filename=u.filename,
            row_count=u.row_count,
            flagged_count=u.flagged_count,
            top_k=u.top_k,
            created_at=u.created_at
        )
        for u in recent_uploads_raw
    ]

    return DashboardResponse(
        total_predictions=total_preds,
        total_uploads=total_uploads,
        high_risk_count=high_risk_count,
        high_risk_ratio=round(high_risk_ratio, 4),
        model_version=settings.MODEL_VERSION,
        top_k_capacity=settings.INSPECTION_CAPACITY_K,
        recent_uploads=recent_uploads
    )
