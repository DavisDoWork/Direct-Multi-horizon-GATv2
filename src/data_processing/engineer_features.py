from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "events_2026-08-12_clean.parquet"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "events_features_2026-08-12.parquet"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_parquet(INPUT)

print("Input rows:", len(df))
print("Input trips:", df["trip_uid"].nunique())

df = df.sort_values(
    ["trip_uid", "stop_sequence"]
).reset_index(drop=True)


# ==================================================
# 1. CORE DELAY FEATURES
# ==================================================

# Current event delay
df["current_delay_sec"] = df["arrival_delay_sec"]

# Previous event delays within the same trip
df["prev_arrival_delay_sec"] = (
    df.groupby("trip_uid")["arrival_delay_sec"]
    .shift(1)
)

df["prev_departure_delay_sec"] = (
    df.groupby("trip_uid")["departure_delay_sec"]
    .shift(1)
)

# Change in delay from previous stop
df["delay_change_sec"] = (
    df["arrival_delay_sec"]
    - df["prev_arrival_delay_sec"]
)


# ==================================================
# 2. DWELL FEATURES
# ==================================================

# Difference between observed and scheduled dwell
df["dwell_delay_sec"] = (
    df["observed_dwell_sec"]
    - df["scheduled_dwell_sec"]
)

# Whether this stop has measurable positive dwell
df["has_positive_dwell"] = (
    df["observed_dwell_sec"] > 0
).astype(int)


# ==================================================
# 3. RUN-TIME FEATURES
# ==================================================

# Previous stop timestamps on the same train
prev_departure = (
    df.groupby("trip_uid")["observed_departure"]
    .shift(1)
)

prev_sched_departure = (
    df.groupby("trip_uid")["scheduled_departure_final"]
    .shift(1)
)

# Observed running time:
# previous stop departure -> current stop arrival
df["observed_run_sec"] = (
    df["observed_arrival"]
    - prev_departure
).dt.total_seconds()

# Scheduled running time
df["scheduled_run_sec"] = (
    df["scheduled_arrival_final"]
    - prev_sched_departure
).dt.total_seconds()

# Extra running time relative to schedule
df["run_delay_sec"] = (
    df["observed_run_sec"]
    - df["scheduled_run_sec"]
)


# ==================================================
# 4. TRIP PROGRESS FEATURES
# ==================================================

trip_max_seq = (
    df.groupby("trip_uid")["stop_sequence"]
    .transform("max")
)

trip_min_seq = (
    df.groupby("trip_uid")["stop_sequence"]
    .transform("min")
)

denom = (
    trip_max_seq
    - trip_min_seq
).replace(0, np.nan)

df["trip_progress"] = (
    (df["stop_sequence"] - trip_min_seq)
    / denom
)

# Number of stops remaining in this trip
df["stops_remaining"] = (
    trip_max_seq
    - df["stop_sequence"]
)


# ==================================================
# 5. TEMPORAL FEATURES
# ==================================================

# Prefer observed arrival time for operational context.
# Fall back to scheduled arrival where necessary.
df["event_time"] = (
    df["observed_arrival"]
    .fillna(df["scheduled_arrival_final"])
)

df["hour"] = (
    df["event_time"].dt.hour
    + df["event_time"].dt.minute / 60.0
)

# Cyclic encoding of hour
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["weekday"] = df["event_time"].dt.dayofweek

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

# NYC subway rush-hour proxy
morning_rush = (
    (df["hour"] >= 7)
    & (df["hour"] < 10)
)

evening_rush = (
    (df["hour"] >= 16)
    & (df["hour"] < 19)
)

df["rush_hour"] = (
    morning_rush | evening_rush
).astype(int)


# ==================================================
# 6. HEADWAY FEATURES
# ==================================================
# Same directional stop + route.
# We sort trains by event time and calculate the
# time gap from the previous train.

headway_sort_cols = [
    "route_id",
    "direction_id",
    "stop_id",
    "event_time",
]

headway_df = (
    df[
        [
            "event_id",
            "route_id",
            "direction_id",
            "stop_id",
            "event_time",
        ]
    ]
    .dropna(subset=["event_time"])
    .sort_values(headway_sort_cols)
    .copy()
)

headway_df["previous_train_time"] = (
    headway_df
    .groupby(
        [
            "route_id",
            "direction_id",
            "stop_id",
        ]
    )["event_time"]
    .shift(1)
)

headway_df["next_train_time"] = (
    headway_df
    .groupby(
        [
            "route_id",
            "direction_id",
            "stop_id",
        ]
    )["event_time"]
    .shift(-1)
)

headway_df["previous_headway_sec"] = (
    headway_df["event_time"]
    - headway_df["previous_train_time"]
).dt.total_seconds()

headway_df["following_headway_sec"] = (
    headway_df["next_train_time"]
    - headway_df["event_time"]
).dt.total_seconds()

df = df.merge(
    headway_df[
        [
            "event_id",
            "previous_headway_sec",
            "following_headway_sec",
        ]
    ],
    on="event_id",
    how="left",
    validate="one_to_one",
)


# ==================================================
# 7. LOCAL STATION CONGESTION FEATURES
# ==================================================
# Compute trailing 10-minute context at each
# directional stop.
#
# IMPORTANT:
# only past/current events are used.
# No future information should enter the features.

station_groups = []

temp = df[
    [
        "event_id",
        "stop_id",
        "event_time",
        "arrival_delay_sec",
    ]
].dropna(
    subset=["event_time"]
).copy()

temp = temp.sort_values(
    ["stop_id", "event_time"]
)

for stop_id, group in temp.groupby("stop_id"):

    g = group.sort_values("event_time").copy()

    g = g.set_index("event_time")

    # Number of events in the previous 10 minutes
    g["station_train_count_10m"] = (
        g["event_id"]
        .rolling("10min")
        .count()
    )

    # Mean delay in previous 10 minutes
    g["station_mean_delay_10m"] = (
        g["arrival_delay_sec"]
        .rolling("10min")
        .mean()
    )

    g = g.reset_index()

    station_groups.append(
        g[
            [
                "event_id",
                "station_train_count_10m",
                "station_mean_delay_10m",
            ]
        ]
    )

station_features = pd.concat(
    station_groups,
    ignore_index=True
)

df = df.merge(
    station_features,
    on="event_id",
    how="left",
    validate="one_to_one",
)


# ==================================================
# 8. OPTIONAL TRACK FEATURE
# ==================================================
# Keep the raw track field for later inspection.
# Do NOT assume it is reliable infrastructure data yet.

df["track_available"] = (
    df["track"].notna()
).astype(int)


# ==================================================
# 9. BASIC FEATURE CLEANING
# ==================================================

# Impossible negative running times are invalid.
df.loc[
    df["observed_run_sec"] < 0,
    "observed_run_sec"
] = np.nan

df.loc[
    df["scheduled_run_sec"] < 0,
    "scheduled_run_sec"
] = np.nan

df.loc[
    df["previous_headway_sec"] < 0,
    "previous_headway_sec"
] = np.nan

df.loc[
    df["following_headway_sec"] < 0,
    "following_headway_sec"
] = np.nan


# ==================================================
# 10. MULTI-HORIZON TARGETS
# ==================================================
# Forecast the next 1, 2, and 3 train-stop events
# on the SAME trip.

grouped_delay = df.groupby(
    "trip_uid"
)["arrival_delay_sec"]

for k in [1, 2, 3]:

    df[f"target_delay_t{k}"] = (
        grouped_delay.shift(-k)
    )

    # Regression mask:
    # target exists
    df[f"target_mask_t{k}"] = (
        df[f"target_delay_t{k}"].notna()
    ).astype(int)


# --------------------------------------------------
# Classification targets
# --------------------------------------------------
# Placeholder threshold:
# delay >= 60 sec = delayed.
#
# Keep this explicit so we can change the threshold
# later to match the anchor paper exactly.

DELAY_THRESHOLD_SEC = 60

for k in [1, 2, 3]:

    target_col = f"target_delay_t{k}"

    df[f"target_delayed_t{k}"] = np.where(
        df[target_col].notna(),
        (
            df[target_col]
            >= DELAY_THRESHOLD_SEC
        ).astype(int),
        np.nan,
    )


# ==================================================
# 11. QUALITY CHECKS
# ==================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING QUALITY CHECKS")
print("=" * 70)

feature_cols = [
    "current_delay_sec",
    "prev_arrival_delay_sec",
    "delay_change_sec",
    "observed_dwell_sec",
    "dwell_delay_sec",
    "observed_run_sec",
    "scheduled_run_sec",
    "run_delay_sec",
    "trip_progress",
    "hour_sin",
    "hour_cos",
    "previous_headway_sec",
    "station_train_count_10m",
    "station_mean_delay_10m",
]

print("\nMissing feature percentages:")

for col in feature_cols:
    pct = df[col].isna().mean() * 100
    print(f"{col:30s}: {pct:6.2f}%")


print("\nHeadway summary:")
print(
    df["previous_headway_sec"]
    .describe()
)

print("\nObserved run-time summary:")
print(
    df["observed_run_sec"]
    .describe()
)

print("\nTrip progress summary:")
print(
    df["trip_progress"]
    .describe()
)


print("\nTarget availability:")

for k in [1, 2, 3]:
    available = df[
        f"target_mask_t{k}"
    ].sum()

    print(
        f"t+{k}:",
        available,
        f"({available / len(df) * 100:.2f}%)"
    )


print("\nTarget delay distributions:")

for k in [1, 2, 3]:
    print(f"\nt+{k}:")
    print(
        df[f"target_delay_t{k}"]
        .describe()
    )


# ==================================================
# 12. SAVE
# ==================================================

df.to_parquet(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\nFinal shape:", df.shape)
