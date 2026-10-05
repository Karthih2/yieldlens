import json
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import joblib
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import engine, Base
from app.routers import health, predict, uploads, predictions, dashboard, sample, stability

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize DB tables
    Base.metadata.create_all(bind=engine)

    # 2. Verify artifacts directory
    artifacts_dir = settings.ARTIFACTS_DIR
    if not artifacts_dir.exists():
        raise RuntimeError(
            f"Artifacts directory not found at '{artifacts_dir}'. "
            "Please run 'python backend/scripts/export_artifacts.py' first."
        )

    # 3. Load Shortlist Sensors
    shortlist_path = artifacts_dir / "shortlist_sensors.csv"
    shortlist_df = pd.read_csv(shortlist_path)
    app.state.shortlist_cols = shortlist_df["sensor"].tolist()

    # 4. Load Model
    model_path = artifacts_dir / "random_forest_reduced_shortlist.pkl"
    app.state.model = joblib.load(model_path)
    assert app.state.model.n_features_in_ == len(app.state.shortlist_cols), "Model/shortlist feature mismatch"

    # 5. Load SHAP TreeExplainer
    app.state.explainer = shap.TreeExplainer(app.state.model)

    # 6. Load Medians
    medians_path = artifacts_dir / "medians.json"
    with open(medians_path, "r") as f:
        app.state.medians = json.load(f)

    # 7. Load Baseline Probability Distribution
    baseline_path = artifacts_dir / "baseline_distribution.json"
    with open(baseline_path, "r") as f:
        app.state.baseline_distribution = json.load(f)

    # 8. Load Stability Scores DataFrame
    stability_path = artifacts_dir / "stability_scores.csv"
    app.state.stability_df = pd.read_csv(stability_path)

    print(f"[{settings.APP_NAME}] Startup complete. Model '{settings.MODEL_VERSION}' loaded with {len(app.state.shortlist_cols)} features.")
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="YieldLens Backend API for Stability-Aware Semiconductor Yield Monitoring",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(uploads.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(sample.router)
app.include_router(stability.router)

# Mount Frontend UI static files if directory exists
from fastapi.staticfiles import StaticFiles
frontend_dir = settings.BASE_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "ui_url": "/ui/"
    }

