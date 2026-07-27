#api/schemas.py

from pydantic import BaseModel
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
    sensor_values: Dict[str, float]


class PredictResponse(BaseModel):
    fail_probability: float
    predicted_label: int
    row_id: int


class Contributor(BaseModel):
    sensor: str
    shap_value: float


class ExplainResponse(BaseModel):
    row_id: int
    actual_label: Optional[int] = None
    shap_values: Dict[str, float]
    top_contributors: List[Contributor]


class StabilityScoreEntry(BaseModel):
    sensor: str
    stability_score: float
    avg_importance: float
    in_shortlist: bool


class StabilityScoresResponse(BaseModel):
    scores: List[StabilityScoreEntry]