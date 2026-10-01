import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

def export_artifacts():
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    root_dir = backend_dir.parent
    artifacts_dir = backend_dir / "artifacts" / "v1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root directory: {root_dir}")
    print(f"Exporting artifacts to: {artifacts_dir}")

    # 1. Copy model file
    model_src = root_dir / "models" / "random_forest_reduced_shortlist.pkl"
    model_dst = artifacts_dir / "random_forest_reduced_shortlist.pkl"
    shutil.copy2(model_src, model_dst)
    print(f"Copied model -> {model_dst.name}")

    # 2. Copy CSV reports
    reports_to_copy = [
        "shortlist_sensors.csv",
        "stability_scores.csv",
        "tripwire_flagged_sensors.csv",
        "optbinning_rules.csv",
        "shap_global_importance.csv",
        "shap_per_prediction.csv",
    ]
    for rfile in reports_to_copy:
        src = root_dir / "reports" / rfile
        if src.exists():
            shutil.copy2(src, artifacts_dir / rfile)
            print(f"Copied report -> {rfile}")

    # 3. Validate model against shortlist
    shortlist_df = pd.read_csv(artifacts_dir / "shortlist_sensors.csv")
    shortlist_cols = shortlist_df["sensor"].tolist()
    model = joblib.load(model_dst)
    assert model.n_features_in_ == len(shortlist_cols), (
        f"Mismatch: model features = {model.n_features_in_}, shortlist len = {len(shortlist_cols)}"
    )
    print(f"Model validation passed: {len(shortlist_cols)} shortlist features")

    # 4. Compute medians from raw secom.parquet (BEFORE imputation)
    raw_secom_path = root_dir / "data" / "processed" / "secom.parquet"
    raw_df = pd.read_parquet(raw_secom_path)
    medians_dict = {}
    for col in shortlist_cols:
        if col in raw_df.columns:
            medians_dict[col] = float(raw_df[col].median())
        else:
            raise KeyError(f"Sensor {col} not found in raw secom.parquet")
    
    with open(artifacts_dir / "medians.json", "w") as f:
        json.dump(medians_dict, f, indent=2)
    print(f"Exported medians for {len(medians_dict)} sensors -> medians.json")

    # 5. Export baseline distribution from model_predictions.npz
    preds_path = root_dir / "data" / "processed" / "model_predictions.npz"
    preds_npz = np.load(preds_path)
    baseline_proba = preds_npz["proba_random_forest_reduced_shortlist"].tolist()
    with open(artifacts_dir / "baseline_distribution.json", "w") as f:
        json.dump(baseline_proba, f, indent=2)
    print(f"Exported baseline distribution ({len(baseline_proba)} probabilities) -> baseline_distribution.json")

    # 6. Export sample_rows.csv (test batch, batch_id == 4)
    batched_path = root_dir / "data" / "processed" / "secom_batched.parquet"
    batched_df = pd.read_parquet(batched_path)
    test_batch_id = batched_df["batch_id"].max()
    test_df = batched_df[batched_df["batch_id"] == test_batch_id].copy()
    
    sample_cols = shortlist_cols + ["labels", "timestamp", "batch_id"]
    available_cols = [c for c in sample_cols if c in test_df.columns]
    sample_df = test_df[available_cols]
    sample_df.to_csv(artifacts_dir / "sample_rows.csv", index=False)
    print(f"Exported sample test batch ({len(sample_df)} rows) -> sample_rows.csv")

    # 7. Export batch_summary.csv
    batch_summary = batched_df.groupby("batch_id").agg(
        n_rows=("labels", "size"),
        n_fail=("labels", "sum")
    ).reset_index()
    batch_summary["fail_rate"] = batch_summary["n_fail"] / batch_summary["n_rows"]
    batch_summary.to_csv(artifacts_dir / "batch_summary.csv", index=False)
    print(f"Exported batch summary -> batch_summary.csv")

    print("\nArtifact export complete!")

if __name__ == "__main__":
    export_artifacts()
