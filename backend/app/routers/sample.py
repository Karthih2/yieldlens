from typing import List, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException, status
import pandas as pd
from app.config import settings

router = APIRouter(tags=["sample"])

@router.get("/sample")
def get_sample_rows(limit: int = Query(20, ge=1, le=314)):
    sample_path = settings.ARTIFACTS_DIR / "sample_rows.csv"
    if not sample_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample data artifact not found. Please run export_artifacts.py."
        )

    df = pd.read_csv(sample_path)
    records = df.head(limit).to_dict(orient="records")
    return {
        "total_sample_rows": len(df),
        "returned_rows": len(records),
        "data": records
    }
