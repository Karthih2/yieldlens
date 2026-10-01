from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

def impute_sensor_dict(
    sensor_values: Dict[str, Optional[float]],
    shortlist_cols: List[str],
    medians: Dict[str, float]
) -> Tuple[Dict[str, float], List[str]]:
    """
    Imputes missing/NaN values in a single sensor input dictionary using training medians.
    Returns:
        imputed_dict: Dict with all shortlist sensors populated with float values.
        imputed_list: List of sensor names that were imputed.
    """
    imputed_dict = {}
    imputed_list = []

    for col in shortlist_cols:
        val = sensor_values.get(col)
        if val is None or pd.isna(val):
            imputed_dict[col] = float(medians[col])
            imputed_list.append(col)
        else:
            try:
                imputed_dict[col] = float(val)
            except (ValueError, TypeError):
                imputed_dict[col] = float(medians[col])
                imputed_list.append(col)

    return imputed_dict, imputed_list


def impute_sensor_dataframe(
    df: pd.DataFrame,
    shortlist_cols: List[str],
    medians: Dict[str, float]
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Imputes missing values in a pandas DataFrame for the required shortlist columns.
    Returns:
        imputed_df: Clean DataFrame containing all shortlist columns in exact order.
        per_row_imputed: List of lists containing imputed sensor names for each row.
    """
    clean_df = pd.DataFrame(index=df.index)
    per_row_imputed = [[] for _ in range(len(df))]

    for col in shortlist_cols:
        if col not in df.columns:
            # Entire column missing
            median_val = float(medians[col])
            clean_df[col] = median_val
            for r_idx in range(len(df)):
                per_row_imputed[r_idx].append(col)
        else:
            col_series = pd.to_numeric(df[col], errors='coerce')
            median_val = float(medians[col])
            is_null = col_series.isna()
            
            for r_idx, null_flag in enumerate(is_null):
                if null_flag:
                    per_row_imputed[r_idx].append(col)
            
            clean_df[col] = col_series.fillna(median_val).astype(float)

    return clean_df[shortlist_cols], per_row_imputed
