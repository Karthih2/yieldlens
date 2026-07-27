# YieldLens — Final Results Summary

Test set: 314 rows (final chronological batch), 17 actual fails (5.4%).
Inspection capacity K = 30 wafers/day.

## Comparison Table

| Model | Feature Set | AUC | Precision@30 | Recall@30 |
|---|---|---|---|---|
| XGBoost | Full (590) | 0.6053 | 0.133 | 0.235 |
| XGBoost | Reduced (~17) | 0.6922 | 0.167 | 0.294 |
| Random Forest | Full (590) | 0.5925 | 0.100 | 0.176 |
| Random Forest | Reduced (~17) | 0.6547 | 0.200 | 0.353 |

## Headline Metric: % Change in Recall@K (Reduced vs Full)

- XGBoost: +25.0% change in Recall@30 using ~17 sensors vs 590
- Random Forest: +100.0% change in Recall@30 using ~17 sensors vs 590

## Interpretation

A positive percentage means the ~17-sensor shortlist model catches *more* true failures within the same fixed daily inspection budget than the full 590-sensor model — i.e., yield loss avoided, not just monitoring overhead reduced.

## Limitations

- Single chronological train/test split on a small test set (~314 rows); results should be treated as indicative, not statistically definitive.
- K=30 is illustrative; should be replaced with the fab's actual daily inspection capacity for operational deployment.
