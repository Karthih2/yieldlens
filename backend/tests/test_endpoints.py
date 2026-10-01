import io
import pytest
import pandas as pd
from app.config import settings

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["explainer_loaded"] is True
    assert data["shortlist_features_count"] == 17

def test_predict_single_endpoint(client):
    # Single prediction with one missing sensor to test auto-imputation
    payload = {
        "sensor_values": {
            "sensor_21": 0.5,
            "sensor_0": None  # Should be imputed
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "fail_probability" in data
    assert data["percentile_context"] == "relative_to_test_batch"
    assert "sensor_0" in data["imputed_sensors"]

def test_upload_small_csv_5_rows(client):
    # Upload small CSV with 5 rows (< K=30 capacity)
    shortlist_cols = [
        'sensor_21', 'sensor_0', 'sensor_285', 'sensor_225', 'sensor_290',
        'sensor_71', 'sensor_59', 'sensor_571', 'sensor_103', 'sensor_406',
        'sensor_70', 'sensor_138', 'sensor_473', 'sensor_287', 'sensor_117',
        'sensor_100', 'sensor_95'
    ]
    df_data = {col: [0.1 * (i + 1) for i in range(5)] for col in shortlist_cols}
    csv_buf = io.StringIO()
    pd.DataFrame(df_data).to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    response = client.post(
        "/uploads",
        files={"file": ("small_test.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["upload"]["row_count"] == 5
    assert data["upload"]["flagged_count"] == 5  # When N < K=30, all 5 rows are flagged top K
    
    # Check predictions list in response
    preds = data["predictions"]
    assert len(preds) == 5
    for p in preds:
        assert p["is_flagged_top_k"] is True
        assert p["risk_level"] == "HIGH"

def test_upload_missing_columns_validation(client):
    # Missing required shortlist columns -> 422
    csv_bytes = b"sensor_21,sensor_0\n1.0,2.0\n"
    response = client.post(
        "/uploads",
        files={"file": ("bad_cols.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 422
    assert "Missing required shortlist sensor columns" in response.json()["detail"]

def test_upload_non_numeric_validation(client):
    # Non-numeric value in shortlist column -> 422
    shortlist_cols = [
        'sensor_21', 'sensor_0', 'sensor_285', 'sensor_225', 'sensor_290',
        'sensor_71', 'sensor_59', 'sensor_571', 'sensor_103', 'sensor_406',
        'sensor_70', 'sensor_138', 'sensor_473', 'sensor_287', 'sensor_117',
        'sensor_100', 'sensor_95'
    ]
    df_data = {col: [1.0] for col in shortlist_cols}
    df_data["sensor_21"] = ["invalid_text"]
    csv_buf = io.StringIO()
    pd.DataFrame(df_data).to_csv(csv_buf, index=False)

    response = client.post(
        "/uploads",
        files={"file": ("corrupt.csv", csv_buf.getvalue().encode("utf-8"), "text/csv")}
    )
    assert response.status_code == 422
    assert "non-numeric invalid values" in response.json()["detail"]

def test_explain_caching(client):
    # Create a single prediction
    payload = {"sensor_values": {"sensor_21": 1.2}}
    pred_res = client.post("/predict", json=payload).json()
    pred_id = pred_res["prediction_id"]

    # First call: computes and caches SHAP
    exp_res1 = client.get(f"/predictions/{pred_id}/explain")
    assert exp_res1.status_code == 200
    data1 = exp_res1.json()
    assert "shap_values" in data1
    assert len(data1["top_contributors"]) <= 5

    # Second call: fetches cached SHAP from DB
    exp_res2 = client.get(f"/predictions/{pred_id}/explain")
    assert exp_res2.status_code == 200
    assert exp_res2.json() == data1

def test_dashboard_endpoint(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "model_version" in data
    assert data["top_k_capacity"] == 30

def test_sample_endpoint(client):
    response = client.get("/sample?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["returned_rows"] == 5

def test_stability_scores_endpoint(client):
    response = client.get("/stability-scores?tier=high")
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    for item in data["scores"]:
        assert item["tier"] == "high"
