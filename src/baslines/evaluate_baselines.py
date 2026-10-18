from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "events_features_final_2026-08-12.parquet"
)

OUTPUT = (
    ROOT
    / "results"
    / "baseline_results_2026-08-12.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_parquet(INPUT)

print("Rows:", len(df))
print("Trips:", df["trip_uid"].nunique())


# ==================================================
# METRIC FUNCTION
# ==================================================

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return mae, rmse


# ==================================================
# RESULTS
# ==================================================

results = []


# ==================================================
# BASELINE 1: ZERO-DELAY
# ==================================================
#
# Rule:
# For the future event being evaluated,
# always predict delay = 0.
# ==================================================

for k in [1, 2, 3]:

    target_col = f"target_delay_t{k}"
    mask_col = f"target_mask_t{k}"

    # Only use rows where this future target exists
    valid = (
        (df[mask_col] == 1)
        & df[target_col].notna()
    )

    y_true = df.loc[
        valid,
        target_col
    ].to_numpy()

    # Zero-delay prediction
    y_pred = np.zeros(
        len(y_true)
    )

    mae, rmse = evaluate(
        y_true,
        y_pred
    )

    results.append(
        {
            "model": "Zero-delay",
            "horizon": f"t+{k}",
            "mae_sec": mae,
            "rmse_sec": rmse,
            "n": len(y_true),
        }
    )


# ==================================================
# BASELINE 2: PERSISTENCE
# ==================================================
#
# Rule:
# Future delay = current delay.
#
# Example:
# current delay = 90 sec
# prediction for t+1 = 90 sec
# prediction for t+2 = 90 sec
# prediction for t+3 = 90 sec
# ==================================================

for k in [1, 2, 3]:

    target_col = f"target_delay_t{k}"
    mask_col = f"target_mask_t{k}"

    # Need:
    # 1. future target exists
    # 2. current delay exists
    valid = (
        (df[mask_col] == 1)
        & df[target_col].notna()
        & df["current_delay_sec"].notna()
    )

    y_true = df.loc[
        valid,
        target_col
    ].to_numpy()

    # Persistence prediction:
    # keep the current delay
    y_pred = df.loc[
        valid,
        "current_delay_sec"
    ].to_numpy()

    mae, rmse = evaluate(
        y_true,
        y_pred
    )

    results.append(
        {
            "model": "Persistence",
            "horizon": f"t+{k}",
            "mae_sec": mae,
            "rmse_sec": rmse,
            "n": len(y_true),
        }
    )


# ==================================================
# RESULTS TABLE
# ==================================================

results_df = pd.DataFrame(results)

results_df["mae_min"] = (
    results_df["mae_sec"] / 60
)

results_df["rmse_min"] = (
    results_df["rmse_sec"] / 60
)


print("\n" + "=" * 80)
print("BASELINE RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "model",
            "horizon",
            "mae_sec",
            "rmse_sec",
            "mae_min",
            "rmse_min",
            "n",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==================================================
# SAVE
# ==================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)