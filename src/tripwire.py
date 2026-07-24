#src/tripwire.py

import pandas as pd

df = pd.read_parquet("../data/processed/secom_clean.parquet", engine='pyarrow')
sensor_columns = [col for col in df.columns if col.startswith("sensor_")]

#tune as needed - defines "near constant"
VARIANCE_THRESHOLD = 1e-6  # Variance threshold for tripwire detection 1e-6 is 0.000001

pass_df = df[df["labels"] == 0] # Pass samples
fail_df = df[df["labels"] == 1] # Fail samples

results = []
for col in sensor_columns:
    variance = df[col].var()
    if variance <= VARIANCE_THRESHOLD:
        pass_mean = pass_df[col].mean()
        fail_mean = fail_df[col].mean()
        overall_std = df[col].std()
        lift = abs(fail_mean - pass_mean) / overall_std if overall_std > 0 else 0
        results.append({
            "sensor": col,
            "variance": variance,
            "pass_mean": pass_mean,
            "fail_mean": fail_mean,
            "overall_std": overall_std,
            "failure_lift": lift
        })

tripwire_df = pd.DataFrame(results).sort_values("failure_lift", ascending=False)
LIFT_THRESHOLD = 0.2

tripwire_df["flagged"] = tripwire_df["failure_lift"] > LIFT_THRESHOLD
tripwire_df.to_csv("../reports/tripwire_flagged_sensors.csv", index=False)
print(f"Checked {len(sensor_columns)} sensors, found {len(tripwire_df)} near-constant, flagged {tripwire_df['flagged'].sum()}")
