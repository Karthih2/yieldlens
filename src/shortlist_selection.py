#src/shortlist_selection.py

import pandas as pd

stability_df = pd.read_csv("../reports/stability_scores.csv")
tripwire_df = pd.read_csv("../reports/tripwire_flagged_sensors.csv")

TARGET_SHORTLIST_SIZE = 15

# 1. Rank by stability first, importance second (already sorted this way, but be explicit)
ranked = stability_df.sort_values(
    ["stability_score", "avg_importance"], ascending=False
).reset_index(drop=True)

# 2. Take top N from the ranked list
shortlist = ranked.head(TARGET_SHORTLIST_SIZE).copy()
shortlist["source"] = "stability_ranking"

# 3. Force-include tripwire-flagged sensors, even if they didn't make the ranked cut
tripwire_whitelist = tripwire_df[tripwire_df["flagged"] == True]["sensor"].tolist()
print(f"Tripwire whitelist sensors: {tripwire_whitelist}")

for sensor in tripwire_whitelist:
    if sensor not in shortlist["sensor"].values:
        row = ranked[ranked["sensor"] == sensor].copy()
        if row.empty:
            # sensor wasn't in stability_scores at all (shouldn't normally happen)
            row = pd.DataFrame([{"sensor": sensor, "stability_score": None, "avg_importance": None}])
        row["source"] = "tripwire_whitelist"
        shortlist = pd.concat([shortlist, row], ignore_index=True)

# 4. Manual correlation screen — flag (don't auto-drop) highly correlated pairs for review
import numpy as np
df = pd.read_parquet("../data/processed/secom_clean.parquet", engine="pyarrow")
sensor_subset = df[shortlist["sensor"].tolist()]
corr_matrix = sensor_subset.corr().abs()

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:  # threshold — tune as needed
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("\n⚠ High-correlation pairs found in shortlist (documented as future work, not auto-dropped):")
    for s1, s2, corr in high_corr_pairs:
        print(f"  {s1} <-> {s2}: corr = {corr:.3f}")
else:
    print("\nNo highly correlated pairs (>0.9) found in shortlist.")

shortlist.to_csv("../reports/shortlist_sensors.csv", index=False)
print(f"\nFinal shortlist size: {len(shortlist)}")
print(shortlist[["sensor", "stability_score", "avg_importance", "source"]])