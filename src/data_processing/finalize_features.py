from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "events_features_2026-08-12.parquet"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "events_features_final_2026-08-12.parquet"
)

df = pd.read_parquet(INPUT)

print("Input rows:", len(df))
print("Input trips:", df["trip_uid"].nunique())


# ==================================================
# 1. RUN-TIME VALIDITY
# ==================================================
#
# Extremely long adjacent-stop run times are usually
# timestamp gaps / service-day artifacts, not real
# train movement.
#
# We KEEP the event and graph topology.
# We only mark the run-time feature unavailable.
# ==================================================

MAX_RUN_SEC = 20 * 60  # 20 minutes

invalid_run = (
    (df["observed_run_sec"] < 0)
    | (df["observed_run_sec"] > MAX_RUN_SEC)
)

print("\nInvalid observed run-time rows:")
print(
    invalid_run.sum(),
    f"({invalid_run.mean() * 100:.4f}%)"
)

df["run_time_valid"] = (
    ~invalid_run
    & df["observed_run_sec"].notna()
).astype(int)

df.loc[
    invalid_run,
    [
        "observed_run_sec",
        "run_delay_sec",
    ]
] = np.nan


# ==================================================
# 2. HEADWAY VALIDITY
# ==================================================
#
# Only connect trains that are reasonably close in
# operational time.
#
# A 20-hour "headway" is simply a dataset/window
# boundary, not a meaningful train interaction.
# ==================================================

MAX_HEADWAY_SEC = 30 * 60  # 30 minutes

invalid_prev_headway = (
    (df["previous_headway_sec"] <= 0)
    | (df["previous_headway_sec"] > MAX_HEADWAY_SEC)
)

invalid_following_headway = (
    (df["following_headway_sec"] <= 0)
    | (df["following_headway_sec"] > MAX_HEADWAY_SEC)
)

print("\nInvalid previous headways:")
print(
    invalid_prev_headway.sum(),
    f"({invalid_prev_headway.mean() * 100:.4f}%)"
)

df["previous_headway_valid"] = (
    ~invalid_prev_headway
    & df["previous_headway_sec"].notna()
).astype(int)

df.loc[
    invalid_prev_headway,
    "previous_headway_sec"
] = np.nan

# following_headway is retained for graph construction /
# diagnostics but MUST NOT be a model node feature.
df.loc[
    invalid_following_headway,
    "following_headway_sec"
] = np.nan


# ==================================================
# 3. STATION CONTEXT
# ==================================================
#
# For station_mean_delay_10m, no historical event may
# exist inside the window. Keep NaN rather than
# pretending the station delay is zero.
# ==================================================

df["station_context_available"] = (
    df["station_mean_delay_10m"].notna()
).astype(int)


# ==================================================
# 4. MODEL INPUT POLICY
# ==================================================
#
# These are candidate numeric node features.
#
# IMPORTANT:
# following_headway_sec is intentionally excluded.
# Targets are also excluded.
# ==================================================

model_feature_cols = [
    "current_delay_sec",
    "prev_arrival_delay_sec",
    "prev_departure_delay_sec",
    "delay_change_sec",

    "observed_dwell_sec",
    "scheduled_dwell_sec",
    "dwell_delay_sec",
    "has_positive_dwell",

    "observed_run_sec",
    "scheduled_run_sec",
    "run_delay_sec",
    "run_time_valid",

    "stop_sequence",
    "trip_progress",
    "stops_remaining",

    "hour_sin",
    "hour_cos",
    "rush_hour",

    "previous_headway_sec",
    "previous_headway_valid",

    "station_train_count_10m",
    "station_mean_delay_10m",
    "station_context_available",
]


# ==================================================
# 5. FINAL QC
# ==================================================

print("\n" + "=" * 70)
print("FINAL FEATURE TABLE")
print("=" * 70)

print("Rows:", len(df))
print("Trips:", df["trip_uid"].nunique())

print("\nObserved run-time after cleanup:")
print(df["observed_run_sec"].describe())

print("\nPrevious headway after cleanup:")
print(df["previous_headway_sec"].describe())

print("\nTarget availability:")

for k in [1, 2, 3]:
    count = df[f"target_mask_t{k}"].sum()

    print(
        f"t+{k}:",
        count,
        f"({count / len(df) * 100:.2f}%)"
    )

print("\nCandidate model features:")
for col in model_feature_cols:
    print(
        f"{col:30s}",
        f"missing={df[col].isna().mean() * 100:6.2f}%"
    )


# ==================================================
# 6. SAVE
# ==================================================

df.to_parquet(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\nFinal shape:", df.shape)
