#src/drift_injection.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
sensor_columns = [c for c in df.columns if c.startswith("sensor_")]

C = 0.05  # locked in from the sweep
TARGET_SENSOR = "sensor_21"  # highest-stability sensor from stability_selection.py

np.random.seed(42)


def get_selected_per_batch(data):
    """Runs LASSO per batch, returns {batch_id: set(selected_sensors)}."""
    selected_per_batch = {}
    for batch_id in sorted(data["batch_id"].unique()):
        batch = data[data["batch_id"] == batch_id]
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
        coefs = pipeline.named_steps["lasso"].coef_[0]
        selected = {sensor_columns[i] for i, c in enumerate(coefs) if abs(c) > 1e-8}
        selected_per_batch[batch_id] = selected
    return selected_per_batch


def stability_from_selected(selected_per_batch, sensor):
    n_batches = len(selected_per_batch)
    times = sum(sensor in s for s in selected_per_batch.values())
    return times / n_batches


# --- Step 1: Baseline run — find out which batches actually selected TARGET_SENSOR ---
baseline_selected = get_selected_per_batch(df)
baseline_stability = stability_from_selected(baseline_selected, TARGET_SENSOR)

print(f"\nBaseline selection per batch for {TARGET_SENSOR}:")
contributing_batches = []
for b, sset in baseline_selected.items():
    is_selected = TARGET_SENSOR in sset
    print(f"  Batch {b}: selected = {is_selected}")
    if is_selected:
        contributing_batches.append(b)

print(f"\n{TARGET_SENSOR} stability BEFORE injection: {baseline_stability:.2f}")
print(f"Batches where it was selected: {contributing_batches}")

if not contributing_batches:
    raise ValueError(
        f"{TARGET_SENSOR} was not selected in any batch — pick a different "
        f"TARGET_SENSOR from stability_scores.csv (one with stability_score > 0)."
    )

# --- Step 2: Inject noise into ONE batch that actually contributed to the score ---
INJECT_BATCH = contributing_batches[0]  # guaranteed to matter to the stability score
print(f"\nInjecting noise into batch {INJECT_BATCH} (a batch that contributed to stability)")

df_drifted = df.copy()
mask = df_drifted["batch_id"] == INJECT_BATCH
original_std = df_drifted.loc[mask, TARGET_SENSOR].std()
overall_mean = df_drifted[TARGET_SENSOR].mean()

noise = np.random.normal(0, original_std, size=mask.sum())
df_drifted.loc[mask, TARGET_SENSOR] = overall_mean + noise

# --- Step 3: Re-run stability selection on the drifted data ---
drifted_selected = get_selected_per_batch(df_drifted)
drifted_stability = stability_from_selected(drifted_selected, TARGET_SENSOR)

print(f"\n{TARGET_SENSOR} stability AFTER injection:  {drifted_stability:.2f}")
print(f"Selected in injected batch ({INJECT_BATCH}) after noise: "
      f"{TARGET_SENSOR in drifted_selected[INJECT_BATCH]}")

# --- Step 4: Save full before/after comparison for all sensors ---
result = pd.DataFrame({
    "sensor": sensor_columns,
    "stability_before": [stability_from_selected(baseline_selected, s) for s in sensor_columns],
    "stability_after": [stability_from_selected(drifted_selected, s) for s in sensor_columns]
})
result["delta"] = result["stability_after"] - result["stability_before"]
result.to_csv("../reports/drift_injection_validation.csv", index=False)

print(f"\nFull comparison saved to reports/drift_injection_validation.csv")
print(result[result["sensor"] == TARGET_SENSOR])