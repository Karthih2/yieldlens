import datetime
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class PredictRequest(BaseModel):
    sensor_values: Dict[str, Optional[float]] = Field(
        ...,
        description="Dictionary mapping sensor names to numeric values. Missing required shortlist sensors will be auto-imputed using medians."
    )

class ContributorSchema(BaseModel):
    sensor: str
    shap_value: float

class PredictResponse(BaseModel):
    prediction_id: int
    fail_probability: float
    percentile_rank: float
    percentile_context: str = "relative_to_test_batch"
    risk_level: str
    is_flagged_top_k: bool
    imputed_sensors: List[str]
    model_version: str
    created_at: datetime.datetime

class ExplainResponse(BaseModel):
    prediction_id: int
    row_index: Optional[int] = None
    shap_values: Dict[str, float]
    top_contributors: List[ContributorSchema]

class UploadSummary(BaseModel):
    id: int
    filename: str
    row_count: int
    flagged_count: int
    top_k: int
    created_at: datetime.datetime

class PredictionItemSchema(BaseModel):
    id: int
    upload_id: Optional[int] = None
    row_index: Optional[int] = None
    fail_probability: float
    percentile_rank: float
    risk_level: str
    is_flagged_top_k: bool
    imputed_sensors: List[str]
    model_version: str
    created_at: datetime.datetime
    sensor_values: Dict[str, float]

class UploadDetailResponse(BaseModel):
    upload: UploadSummary
    predictions: List[PredictionItemSchema]

class PredictionsListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    predictions: List[PredictionItemSchema]

class DashboardResponse(BaseModel):
    total_predictions: int
    total_uploads: int
    high_risk_count: int
    high_risk_ratio: float
    model_version: str
    top_k_capacity: int
    recent_uploads: List[UploadSummary]

class StabilityScoreItem(BaseModel):
    sensor: str
    stability_score: float
    avg_importance: float
    tier: str
    in_shortlist: bool

class StabilityScoresResponse(BaseModel):
    scores: List[StabilityScoreItem]
    total: int
    page: int
    page_size: int
