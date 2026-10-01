import pytest
import numpy as np
import pandas as pd
from app.services.preprocessing import impute_sensor_dict, impute_sensor_dataframe
from app.services.metrics import precision_at_k, recall_at_k
from app.services.scoring import score_batch_predictions, score_single_prediction

def test_impute_sensor_dict():
    shortlist = ["sensor_1", "sensor_2"]
    medians = {"sensor_1": 10.0, "sensor_2": 20.0}
    input_dict = {"sensor_1": None, "sensor_2": 5.5}

    clean, imputed = impute_sensor_dict(input_dict, shortlist, medians)
    assert clean["sensor_1"] == 10.0
    assert clean["sensor_2"] == 5.5
    assert imputed == ["sensor_1"]

def test_impute_sensor_dataframe():
    shortlist = ["sensor_1", "sensor_2"]
    medians = {"sensor_1": 1.0, "sensor_2": 2.0}
    raw_df = pd.DataFrame({"sensor_1": [None, 3.0], "other": [10, 20]})

    clean_df, imputed_per_row = impute_sensor_dataframe(raw_df, shortlist, medians)
    assert list(clean_df.columns) == shortlist
    assert clean_df.iloc[0]["sensor_1"] == 1.0
    assert clean_df.iloc[0]["sensor_2"] == 2.0
    assert "sensor_1" in imputed_per_row[0]
    assert "sensor_2" in imputed_per_row[0]

def test_metrics_precision_recall_at_k():
    y_true = np.array([1, 0, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.4, 0.1])
    
    # K=2 -> top 2 are idx 0 (fail) and idx 1 (pass) -> 1 fail / 2 = 0.5 precision, 1 fail / 2 total fails = 0.5 recall
    prec = precision_at_k(y_true, y_proba, k=2)
    rec = recall_at_k(y_true, y_proba, k=2)
    assert prec == 0.5
    assert rec == 0.5

    # K=3 -> top 3 are idx 0, 1, 2 -> 2 fails / 3 = 0.6667 precision, 2 fails / 2 total fails = 1.0 recall
    rec_all = recall_at_k(y_true, y_proba, k=3)
    assert rec_all == 1.0

def test_scoring_batch_and_single():
    probas = [0.1, 0.9, 0.4, 0.8]
    scored = score_batch_predictions(probas, capacity_k=2)
    
    # Highest prob is index 1 (0.9), second is index 3 (0.8)
    assert scored[1]["is_flagged_top_k"] is True
    assert scored[1]["risk_level"] == "HIGH"
    assert scored[3]["is_flagged_top_k"] is True
    assert scored[3]["risk_level"] == "HIGH"
    assert scored[0]["is_flagged_top_k"] is False
    assert scored[0]["risk_level"] == "LOW"

    # Single prediction scoring
    baseline_dist = [0.1] * 90 + [0.9] * 10
    single_res = score_single_prediction(0.95, baseline_dist, capacity_k=30)
    assert single_res["is_flagged_top_k"] is True
    assert single_res["risk_level"] == "HIGH"
