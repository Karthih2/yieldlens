#src/preprocessing.py

import pandas as pd
df = pd.read_parquet("../data/processed/secom.parquet", engine='pyarrow')
sensor_columns = [col for col in df.columns if col.startswith("sensor_")]

# # 1. Drop fully-missing columns (none in this case, but keep the logic for safety)
# fully_missing = df[sensor_columns].columns[df[sensor_columns].isnull().all()]
# print(f"Columns with all missing values: {fully_missing.tolist()}")
# df = df.drop(columns=fully_missing)
# sensor_columns = [c for c in sensor_columns if c not in fully_missing]

# 2. Check how much missingness remains (useful sanity check before imputing)
missing_pct = df[sensor_columns].isnull().mean().sort_values(ascending=False)
print(f"Top 5 sensors by missing %:\n{missing_pct.head()}")


# 3. Median-impute remaining missing values, per sensor
df[sensor_columns] = df[sensor_columns].fillna(df[sensor_columns].median())

# Confirm no missing values remain
assert df[sensor_columns].isnull().sum().sum() == 0, "Missing values remain after imputation"

df.to_parquet("../data/processed/secom_clean.parquet", engine="pyarrow")
print("Saved cleaned data to secom_clean.parquet")