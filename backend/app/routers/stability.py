from typing import Optional
from fastapi import APIRouter, Depends, Query
import pandas as pd

from app.config import settings
from app.dependencies import get_stability_df, get_shortlist
from app.schemas import StabilityScoresResponse, StabilityScoreItem

router = APIRouter(tags=["stability"])

def get_tier(score: float) -> str:
    if score >= settings.STABILITY_TIER_HIGH:
        return "high"
    elif score >= settings.STABILITY_TIER_MEDIUM:
        return "medium"
    elif score > settings.STABILITY_TIER_LOW:
        return "low"
    else:
        return "zero"

@router.get("/stability-scores", response_model=StabilityScoresResponse)
def get_stability_scores(
    search: Optional[str] = Query(None, description="Search sensor name"),
    tier: Optional[str] = Query(None, description="Filter tier (high, medium, low, zero)"),
    shortlisted_only: Optional[bool] = Query(None, description="Filter shortlisted sensors only"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=600),
    stability_df: pd.DataFrame = Depends(get_stability_df),
    shortlist_cols = Depends(get_shortlist)
):
    shortlist_set = set(shortlist_cols)
    df = stability_df.copy()

    # Calculate tiers using config cut-offs
    df["tier"] = df["stability_score"].apply(get_tier)
    df["in_shortlist"] = df["sensor"].isin(shortlist_set)

    # Search filter
    if search:
        search_clean = search.strip().lower()
        df = df[df["sensor"].str.lower().str.contains(search_clean)]

    # Tier filter
    if tier:
        tier_clean = tier.strip().lower()
        df = df[df["tier"] == tier_clean]

    # Shortlisted filter
    if shortlisted_only is not None:
        df = df[df["in_shortlist"] == shortlisted_only]

    total = len(df)
    offset = (page - 1) * page_size
    paged_df = df.iloc[offset:offset + page_size]

    items = [
        StabilityScoreItem(
            sensor=row["sensor"],
            stability_score=float(row["stability_score"]),
            avg_importance=float(row["avg_importance"]),
            tier=row["tier"],
            in_shortlist=bool(row["in_shortlist"])
        )
        for _, row in paged_df.iterrows()
    ]

    return StabilityScoresResponse(
        scores=items,
        total=total,
        page=page,
        page_size=page_size
    )
