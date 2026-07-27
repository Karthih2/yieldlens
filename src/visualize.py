#src/visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../reports/figures", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

# 1. Stability vs Importance scatter (matches the React frontend spec)
stability_df = pd.read_csv("../reports/stability_scores.csv")
shortlist = pd.read_csv("../reports/shortlist_sensors.csv")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(stability_df["avg_importance"], stability_df["stability_score"],
           alpha=0.3, s=20, color="gray", label="All sensors")
shortlisted = stability_df[stability_df["sensor"].isin(shortlist["sensor"])]
ax.scatter(shortlisted["avg_importance"], shortlisted["stability_score"],
           color="crimson", s=60, label="Shortlisted (~17)")
for _, row in shortlisted.iterrows():
    ax.annotate(row["sensor"].replace("sensor_", "s"),
                (row["avg_importance"], row["stability_score"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Average LASSO Importance (|coef|)")
ax.set_ylabel("Stability Score")
ax.set_title("Sensor Stability vs Importance")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/stability_vs_importance.png")
plt.close()
print("Saved: stability_vs_importance.png")

# 2. SHAP global importance bar chart
shap_df = pd.read_csv("../reports/shap_global_importance.csv").head(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(shap_df["sensor"], shap_df["mean_abs_shap"], color="steelblue")
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top Sensors by SHAP Importance (Reduced Model)")
plt.tight_layout()
plt.savefig("../reports/figures/shap_importance.png")
plt.close()
print("Saved: shap_importance.png")

# 3. Batch fail-rate drift chart
batched_df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
batch_summary = batched_df.groupby("batch_id").agg(
    n_rows=("labels", "size"), n_fail=("labels", "sum")
).reset_index()
batch_summary["fail_rate"] = batch_summary["n_fail"] / batch_summary["n_rows"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(batch_summary["batch_id"], batch_summary["fail_rate"], marker="o", color="darkorange", linewidth=2)
ax.axhline(batch_summary["fail_rate"].mean(), linestyle="--", color="gray",
           label=f"Mean fail rate ({batch_summary['fail_rate'].mean():.1%})")
ax.set_xlabel("Chronological Batch (time order)")
ax.set_ylabel("Fail Rate")
ax.set_title("Process Drift: Fail Rate Across Time Batches")
ax.set_xticks(batch_summary["batch_id"])
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/batch_drift.png")
plt.close()
print("Saved: batch_drift.png")

# 4. Model comparison: AUC / Precision@K / Recall@K, baseline vs reduced
from sklearn.metrics import roc_auc_score
data = np.load("../data/processed/model_predictions.npz")
K = 30
y_test = data["y_test"]

def precision_at_k(y_true, y_proba, k):
    idx = np.argsort(y_proba)[::-1][:k]
    return y_true[idx].sum() / k

def recall_at_k(y_true, y_proba, k):
    idx = np.argsort(y_proba)[::-1][:k]
    return y_true[idx].sum() / y_true.sum()

runs = ["xgboost_baseline590", "xgboost_reduced_shortlist",
        "random_forest_baseline590", "random_forest_reduced_shortlist"]
metrics = {"AUC": [], "Precision@K": [], "Recall@K": []}
for r in runs:
    proba = data[f"proba_{r}"]
    metrics["AUC"].append(roc_auc_score(y_test, proba))
    metrics["Precision@K"].append(precision_at_k(y_test, proba, K))
    metrics["Recall@K"].append(recall_at_k(y_test, proba, K))

labels = ["XGB\nFull", "XGB\nReduced", "RF\nFull", "RF\nReduced"]
x = np.arange(len(labels))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 6))
for i, (name, vals) in enumerate(metrics.items()):
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_title(f"Model Comparison: Full (590) vs Reduced (~17 sensors), K={K}")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/model_comparison.png")
plt.close()
print("Saved: model_comparison.png")

# 5. Risk bins for the top sensor
bins_df = pd.read_csv("../reports/optbinning_rules.csv")
top_sensor = shap_df.iloc[0]["sensor"]
sensor_bins = bins_df[(bins_df["sensor"] == top_sensor) &
                       (~bins_df["Bin"].isin(["Special", "Missing", "Totals"])) &
                       (bins_df["Bin"].notna())]
if not sensor_bins.empty and "Event rate" in sensor_bins.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sensor_bins["Bin"].astype(str), sensor_bins["Event rate"], color="teal")
    ax.set_xlabel("Value Range")
    ax.set_ylabel("Event Rate (Fail Probability)")
    ax.set_title(f"Risk by Value Range: {top_sensor}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"../reports/figures/{top_sensor}_risk_bins.png")
    plt.close()
    print(f"Saved: {top_sensor}_risk_bins.png")
else:
    print(f"Skipped risk-bin chart — 'Event rate' column not visible. Add "
          f"pd.set_option('display.max_columns', None) to explain_binning.py and re-run first.")

print("\nAll figures saved to reports/figures/")