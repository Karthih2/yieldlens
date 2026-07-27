#api/routes/stability.py

import pandas as pd
from fastapi import APIRouter

from schemas import StabilityScoresResponse, StabilityScoreEntry

router = APIRouter(tags=["stability"])

_stability_df = pd.read_csv("../reports/stability_scores.csv")
_shortlist = set(pd.read_csv("../reports/shortlist_sensors.csv")["sensor"].tolist())


@router.get("/stability-scores", response_model=StabilityScoresResponse)
def get_stability_scores():
    entries = [
        StabilityScoreEntry(
            sensor=row["sensor"],
            stability_score=row["stability_score"],
            avg_importance=row["avg_importance"],
            in_shortlist=row["sensor"] in _shortlist,
        )
        for _, row in _stability_df.iterrows()
    ]
    return StabilityScoresResponse(scores=entries)