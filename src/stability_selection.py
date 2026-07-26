#src/stability_selection.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
sensor_columns = [c for c in df.columns if c.startswith("sensor_")]

N_BATCHES = df["batch_id"].nunique()
selected_per_batch = {}   # batch_id -> set of selected sensors
coef_per_batch = {}       # batch_id -> {sensor: abs(coef)}

for batch_id in sorted(df["batch_id"].unique()):
    batch = df[df["batch_id"] == batch_id]
    X = batch[sensor_columns]
    y = batch["labels"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.05,             # inverse regularization strength — tune this
            max_iter=1000,
            class_weight="balanced"   # important: batches are still imbalanced
        ))
    ])
    pipeline.fit(X, y)

    coefs = pipeline.named_steps["lasso"].coef_[0]
    selected = [sensor_columns[i] for i, c in enumerate(coefs) if abs(c) > 1e-8]

    selected_per_batch[batch_id] = set(selected)
    coef_per_batch[batch_id] = dict(zip(sensor_columns, np.abs(coefs)))

    print(f"Batch {batch_id}: {len(selected)} sensors selected")

# --- Stability Score ---
stability_records = []
for sensor in sensor_columns:
    times_selected = sum(1 for b in selected_per_batch.values() if sensor in b)
    stability_score = times_selected / N_BATCHES
    avg_importance = np.mean([coef_per_batch[b][sensor] for b in coef_per_batch])
    stability_records.append({
        "sensor": sensor,
        "stability_score": stability_score,
        "avg_importance": avg_importance
    })

stability_df = pd.DataFrame(stability_records).sort_values(
    ["stability_score", "avg_importance"], ascending=False
)

stability_df.to_csv("../reports/stability_scores.csv", index=False)
print(stability_df.head(20))