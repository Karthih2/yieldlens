#src/explain_shap.py

import pandas as pd
import numpy as np
import joblib
import shap

df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
shortlist = pd.read_csv("../reports/shortlist_sensors.csv")["sensor"].tolist()

N_BATCHES = df["batch_id"].nunique()
TEST_BATCH = N_BATCHES - 1
test_df = df[df["batch_id"] == TEST_BATCH]

model = joblib.load("../models/random_forest_reduced_shortlist.pkl")

X_test = np.ascontiguousarray(test_df[shortlist].astype(np.float64).to_numpy())
y_test = np.ascontiguousarray(test_df["labels"].astype(np.int64).to_numpy())

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    # Older shap versions: list of arrays, one per class
    shap_vals_fail = shap_values[1]
elif shap_values.ndim == 3:
    # Newer shap versions: single array (n_samples, n_features, n_classes)
    shap_vals_fail = shap_values[:, :, 1]
else:
    # Already 2D (n_samples, n_features)
    shap_vals_fail = shap_values

print(f"shap_vals_fail shape: {shap_vals_fail.shape}  "
      f"(expected: ({X_test.shape[0]}, {len(shortlist)}))")

global_importance = pd.DataFrame({
    "sensor": shortlist,
    "mean_abs_shap": np.abs(shap_vals_fail).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

global_importance.to_csv("../reports/shap_global_importance.csv", index=False)
print("\nGlobal SHAP importance (top 10):")
print(global_importance.head(10))

shap_per_prediction = pd.DataFrame(shap_vals_fail, columns=shortlist)
shap_per_prediction.insert(0, "row_id", range(len(shap_per_prediction)))
shap_per_prediction.insert(1, "actual_label", y_test)
shap_per_prediction.to_csv("../reports/shap_per_prediction.csv", index=False)
print(f"\nSaved per-prediction SHAP values for {len(shap_per_prediction)} test rows")
print("\nDone. Run explain_binning.py next.")