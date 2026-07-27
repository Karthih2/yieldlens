#src/modeling.py

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
sensor_columns = [c for c in df.columns if c.startswith("sensor_")]
shortlist = pd.read_csv("../reports/shortlist_sensors.csv")["sensor"].tolist()

# --- Chronological train/test split: hold out the FINAL batch as test ---
N_BATCHES = df["batch_id"].nunique()
TEST_BATCH = N_BATCHES - 1

train_df = df[df["batch_id"] < TEST_BATCH]
test_df = df[df["batch_id"] == TEST_BATCH]

print(f"Train: {len(train_df)} rows (batches 0-{TEST_BATCH - 1}), "
      f"Test: {len(test_df)} rows (batch {TEST_BATCH})")
print(f"Train fail rate: {train_df['labels'].mean():.3f}, "
      f"Test fail rate: {test_df['labels'].mean():.3f}")


def get_model(model_name):
    if model_name == "xgboost":
        return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1)
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=1
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def train_and_eval(feature_cols, model_name, feature_set_name):
    X_train = np.ascontiguousarray(train_df[feature_cols].astype(np.float64).to_numpy())
    y_train = np.ascontiguousarray(train_df["labels"].astype(np.int64).to_numpy())
    X_test = np.ascontiguousarray(test_df[feature_cols].astype(np.float64).to_numpy())
    y_test = np.ascontiguousarray(test_df["labels"].astype(np.int64).to_numpy())

    # SMOTE run manually (train data only — no leakage into test set)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    X_res = np.ascontiguousarray(X_res.astype(np.float64))
    y_res = np.ascontiguousarray(y_res.astype(np.int64))

    model = get_model(model_name)
    model.fit(X_res, y_res)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    tag = f"{model_name}_{feature_set_name}"
    joblib.dump(model, f"../models/{tag}.pkl")
    print(f"{tag}: AUC = {auc:.4f}")

    return y_test, y_proba


results = {}

# --- 2x2 runs: {xgboost, random_forest} x {baseline (590), reduced (shortlist)} ---
for model_name in ["xgboost", "random_forest"]:
    print(f"\n--- {model_name} | baseline (590 sensors) ---")
    y_test, proba = train_and_eval(sensor_columns, model_name, "baseline590")
    results[f"{model_name}_baseline590"] = {"y_test": y_test, "proba": proba}

    print(f"\n--- {model_name} | reduced (shortlist sensors) ---")
    y_test, proba = train_and_eval(shortlist, model_name, "reduced_shortlist")
    results[f"{model_name}_reduced_shortlist"] = {"y_test": y_test, "proba": proba}

# --- Save all predictions for metrics.py ---
save_dict = {"y_test": results["xgboost_baseline590"]["y_test"]}  # same y_test across all runs
for key, val in results.items():
    save_dict[f"proba_{key}"] = val["proba"]

np.savez("../data/processed/model_predictions.npz", **save_dict)
print("\nSaved all predictions to data/processed/model_predictions.npz for metrics.py")