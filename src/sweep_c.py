#src/sweep_c.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
sensor_columns = [c for c in df.columns if c.startswith("sensor_")]

for C in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    counts = []
    for batch_id in sorted(df["batch_id"].unique()):
        batch = df[df["batch_id"] == batch_id]
        X = batch[sensor_columns]
        y = batch["labels"]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LogisticRegression(
                penalty="l1", solver="liblinear", C=C,
                max_iter=1000, class_weight="balanced"
            ))
        ])
        pipeline.fit(X, y)
        n_selected = sum(abs(c) > 1e-8 for c in pipeline.named_steps["lasso"].coef_[0])
        counts.append(n_selected)

    print(f"C={C}: per-batch selected = {counts}")