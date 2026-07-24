#src/data_loader.py

import pandas as pd

sensor_data = pd.read_csv("../data/raw/secom.data", sep = r"\s+", header = None)
sensor_data.columns = [f"sensor_{i}" for i in range(sensor_data.shape[1])]

labels = pd.read_csv(
    "../data/raw/secom_labels.data", 
    sep = r"\s+", header = None,
    names = ["labels_raw", "timestamp"],
    parse_dates=["timestamp"]
)

df = pd.concat([sensor_data, labels], axis=1)
#Note: The labels in original dataset is -1 means Pass and 1 means Fail.
# The labels - after the below step it wil be 0 is Pass and 1 is Fail.
df["labels"] = (df["labels_raw"] == 1).astype(int)
df= df.drop(columns = ["labels_raw"]) # keeping only cleaned label column
df= df.sort_values(by="timestamp").reset_index(drop=True) 

df.to_parquet("../data/processed/secom.parquet")

