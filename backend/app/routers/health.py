from fastapi import APIRouter, Depends, Request
from app.config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health_check(request: Request):
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    explainer_loaded = hasattr(request.app.state, "explainer") and request.app.state.explainer is not None
    shortlist_len = len(request.app.state.shortlist_cols) if hasattr(request.app.state, "shortlist_cols") else 0

    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "model_version": settings.MODEL_VERSION,
        "model_loaded": model_loaded,
        "explainer_loaded": explainer_loaded,
        "shortlist_features_count": shortlist_len
    }
