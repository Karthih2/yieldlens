#src/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score

data = np.load("../data/processed/model_predictions.npz")
y_test = data["y_test"]

N_FAILS_TOTAL = y_test.sum()
N_TEST = len(y_test)
print(f"Test set: {N_TEST} rows, {N_FAILS_TOTAL} actual fails "
      f"({N_FAILS_TOTAL / N_TEST:.1%} fail rate)")

# K = fixed daily inspection capacity — how many wafers can realistically be
# manually inspected per day. Tune this to a realistic operational number.
K = 30


def precision_at_k(y_true, y_proba, k):
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return y_true[top_k_idx].sum() / k


def recall_at_k(y_true, y_proba, k):
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    total_fails = y_true.sum()
    if total_fails == 0:
        return 0.0
    return y_true[top_k_idx].sum() / total_fails


run_names = [
    "xgboost_baseline590",
    "xgboost_reduced_shortlist",
    "random_forest_baseline590",
    "random_forest_reduced_shortlist",
]

rows = []
for name in run_names:
    proba = data[f"proba_{name}"]
    auc = roc_auc_score(y_test, proba)
    prec_k = precision_at_k(y_test, proba, K)
    rec_k = recall_at_k(y_test, proba, K)
    rows.append({
        "run": name, "auc": auc,
        f"precision_at_{K}": prec_k, f"recall_at_{K}": rec_k,
    })
    print(f"{name}: AUC={auc:.4f}  Precision@{K}={prec_k:.3f}  Recall@{K}={rec_k:.3f}")


def yield_loss_avoided(baseline_recall, reduced_recall):
    if baseline_recall == 0:
        return None
    return (reduced_recall - baseline_recall) / baseline_recall


xgb_base = next(r for r in rows if r["run"] == "xgboost_baseline590")
xgb_red = next(r for r in rows if r["run"] == "xgboost_reduced_shortlist")
rf_base = next(r for r in rows if r["run"] == "random_forest_baseline590")
rf_red = next(r for r in rows if r["run"] == "random_forest_reduced_shortlist")

xgb_yield = yield_loss_avoided(xgb_base[f"recall_at_{K}"], xgb_red[f"recall_at_{K}"])
rf_yield = yield_loss_avoided(rf_base[f"recall_at_{K}"], rf_red[f"recall_at_{K}"])

with open("../reports/final_summary.md", "w") as f:
    f.write("# YieldLens — Final Results Summary\n\n")
    f.write(f"Test set: {N_TEST} rows (final chronological batch), "
            f"{N_FAILS_TOTAL} actual fails ({N_FAILS_TOTAL / N_TEST:.1%}).\n")
    f.write(f"Inspection capacity K = {K} wafers/day.\n\n")

    f.write("## Comparison Table\n\n")
    f.write(f"| Model | Feature Set | AUC | Precision@{K} | Recall@{K} |\n")
    f.write("|---|---|---|---|---|\n")
    for r in rows:
        feature_set = "Full (590)" if "baseline590" in r["run"] else "Reduced (~17)"
        model_fam = "XGBoost" if "xgboost" in r["run"] else "Random Forest"
        f.write(f"| {model_fam} | {feature_set} | {r['auc']:.4f} | "
                f"{r[f'precision_at_{K}']:.3f} | {r[f'recall_at_{K}']:.3f} |\n")

    f.write("\n## Headline Metric: % Change in Recall@K (Reduced vs Full)\n\n")
    if xgb_yield is not None:
        f.write(f"- XGBoost: {xgb_yield:+.1%} change in Recall@{K} using ~17 sensors vs 590\n")
    if rf_yield is not None:
        f.write(f"- Random Forest: {rf_yield:+.1%} change in Recall@{K} using ~17 sensors vs 590\n")

    f.write("\n## Interpretation\n\n")
    f.write("A positive percentage means the ~17-sensor shortlist model catches "
            "*more* true failures within the same fixed daily inspection budget "
            "than the full 590-sensor model — i.e., yield loss avoided, not just "
            "monitoring overhead reduced.\n\n")
    f.write("## Limitations\n\n")
    f.write("- Single chronological train/test split on a small test set "
            "(~314 rows); results should be treated as indicative, not "
            "statistically definitive.\n")
    f.write("- K=30 is illustrative; should be replaced with the fab's actual "
            "daily inspection capacity for operational deployment.\n")

print("\nSaved comparison table to reports/final_summary.md")