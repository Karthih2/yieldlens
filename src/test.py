import pandas as pd
import numpy as np
import lightgbm as lgb

df = pd.read_parquet("../data/processed/secom_batched.parquet")

sensor_columns = [c for c in df.columns if c.startswith("sensor_")]

N_BATCHES = df["batch_id"].nunique()
TEST_BATCH = N_BATCHES - 1

train = df[df["batch_id"] < TEST_BATCH]

X = train[sensor_columns].to_numpy(dtype=np.float32)
y = train["labels"].to_numpy(dtype=np.float32)

print(X.shape)
print(y.shape)

dataset = lgb.Dataset(X, label=y)

print("Dataset created successfully")

dataset.construct()

print("Dataset constructed successfully")