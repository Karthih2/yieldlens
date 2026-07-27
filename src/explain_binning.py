#src/explain_binning.py

from optbinning import OptimalBinning
import pandas as pd

df = pd.read_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")
print("Step 4: data loaded")
shortlist = pd.read_csv("../reports/shortlist_sensors.csv")["sensor"].tolist()
print("Step 5: shortlist loaded")

N_BATCHES = df["batch_id"].nunique()
TEST_BATCH = N_BATCHES - 1
train_df = df[df["batch_id"] < TEST_BATCH]
train_X, train_y = train_df[shortlist], train_df["labels"]
print("Step 6: train data prepared")

global_importance = pd.read_csv("../reports/shap_global_importance.csv")
top_sensors = global_importance.head(5)["sensor"].tolist()
print(f"Step 7: top sensors = {top_sensors}")

binning_results = []
for sensor in top_sensors:
    print(f"  Binning {sensor}...")
    try:
        optb = OptimalBinning(name=sensor, dtype="numerical", solver="mip", max_n_bins=5)
        optb.fit(train_X[sensor].values, train_y.values)
        print(f"  {sensor} fit complete")
        table = optb.binning_table.build()
        table["sensor"] = sensor
        binning_results.append(table)
        print(f"\n--- {sensor} risk bins ---\n{table}")
    except Exception as e:
        print(f"  OptBinning failed for {sensor}: {e}")

if binning_results:
    pd.concat(binning_results, ignore_index=True).to_csv("../reports/optbinning_rules.csv", index=False)
    print("\nSaved to reports/optbinning_rules.csv")