#src/batching.py

import pandas as pd

df = pd.read_parquet("../data/processed/secom_clean.parquet", engine='pyarrow')
N_BATCHES = 5 # tune based on dataset size / stability tradeoff

# Data must already be sorted by timestamp (done in data_loader.py) — confirm:
assert df["timestamp"].is_monotonic_increasing, "Data is not sorted by timestamp!"

df["batch_id"] = pd.cut(range(len(df)), bins = N_BATCHES, labels = False)

batch_summary = df.groupby("batch_id").agg(
    n_rows = ("labels", "size"),
    n_fail = ("labels", "sum")
)

batch_summary["fail_rate"] = batch_summary["n_fail"] / batch_summary["n_rows"]
print(batch_summary)

df.to_parquet("../data/processed/secom_batched.parquet", engine="pyarrow")