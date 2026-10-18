from pathlib import Path
from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SERVICE_DATE = pd.Timestamp("2026-08-12")

GTFS = ROOT / "data" / "raw" / "gtfs_static"
OBS = ROOT / "data" / "raw" / "subwaydata_2026-08-12"
OUT = ROOT / "data" / "processed"

OUTPUT = OUT / "events_pregraph_cleaned.parquet"

MAX_MATCH_TIME_SEC = 5 * 60
BAD_DELAY_SEC = 2 * 3600
DELAY_THRESHOLD_SEC = 60


# ============================================================
# HELPERS
# ============================================================

def hhmmss_to_seconds(value):
    if pd.isna(value):
        return np.nan
    value = str(value).zfill(6)
    try:
        return int(value[:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:6])
    except ValueError:
        return np.nan


def observed_start_seconds(trip_id):
    match = re.match(r"^(\d{6})_", str(trip_id))
    return hhmmss_to_seconds(match.group(1)) if match else np.nan


def static_start_seconds(trip_id):
    matches = re.findall(r"_(\d{6})_", str(trip_id))
    return hhmmss_to_seconds(matches[-1]) if matches else np.nan


def sequence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def gtfs_time_to_timestamp(value):
    if pd.isna(value):
        return pd.NaT
    try:
        h, m, s = map(int, str(value).split(":"))
        return (
            SERVICE_DATE
            + pd.Timedelta(days=h // 24)
            + pd.Timedelta(hours=h % 24, minutes=m, seconds=s)
        )
    except Exception:
        return pd.NaT


def unix_to_timestamp(series):
    return (
        pd.to_datetime(series, unit="s", utc=True, errors="coerce")
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )


def choose_day_shift(group):
    valid = group.dropna(subset=["observed_arrival", "scheduled_arrival"])
    if valid.empty:
        return 0

    scores = {}
    for shift in [-1, 0, 1]:
        scheduled = valid["scheduled_arrival"] + pd.Timedelta(days=shift)
        delay = (valid["observed_arrival"] - scheduled).dt.total_seconds()
        scores[shift] = delay.abs().median()

    return min(scores, key=scores.get)


def nearest_schedule_day(observed, scheduled):
    if pd.isna(observed) or pd.isna(scheduled):
        return scheduled

    candidates = [
        scheduled - pd.Timedelta(days=1),
        scheduled,
        scheduled + pd.Timedelta(days=1),
    ]
    return min(candidates, key=lambda x: abs((observed - x).total_seconds()))


def make_mask(df, value_col, mask_col):
    if mask_col not in df.columns:
        df[mask_col] = df[value_col].notna().astype("int8")
    else:
        df[mask_col] = df[mask_col].astype("int8")


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("Loading raw data...")

observed_trips = pd.read_csv(OBS / "trips.csv")
observed_stops = pd.read_csv(OBS / "stop_times.csv")

static_trips = pd.read_csv(GTFS / "trips.txt")
static_stop_times = pd.read_csv(GTFS / "stop_times.txt")
static_stops = pd.read_csv(GTFS / "stops.txt")

# August 12, 2026 is a weekday.
static_trips = static_trips[
    static_trips["service_id"].str.startswith("Weekday", na=False)
].copy()


# ============================================================
# 2. MATCH OBSERVED TRIPS TO STATIC GTFS
# ============================================================

# ---------- Exact suffix match ----------

exact = observed_trips.merge(
    static_trips,
    on=["route_id", "direction_id"],
    suffixes=("_obs", "_static"),
)

exact["suffix_match"] = exact.apply(
    lambda row: str(row["trip_id_static"]).endswith(str(row["trip_id_obs"])),
    axis=1,
)

exact = exact[exact["suffix_match"]].copy()

counts = exact.groupby("trip_uid").size()
unique_uids = counts[counts == 1].index
exact = exact[exact["trip_uid"].isin(unique_uids)]

exact = exact[
    [
        "trip_uid",
        "trip_id_obs",
        "trip_id_static",
        "route_id",
        "direction_id",
        "start_time",
        "vehicle_id",
    ]
].rename(
    columns={
        "trip_id_obs": "observed_trip_id",
        "trip_id_static": "static_trip_id",
    }
)

exact["match_method"] = "exact_suffix"
exact["time_diff_sec"] = 0.0
exact["sequence_similarity"] = 1.0

# ---------- Prepare fallback matching ----------

observed_sequences = (
    observed_stops.dropna(subset=["stop_id"])
    .groupby("trip_uid")["stop_id"]
    .apply(list)
    .to_dict()
)

static_sequences = (
    static_stop_times.dropna(subset=["stop_id"])
    .groupby("trip_id")["stop_id"]
    .apply(list)
    .to_dict()
)

static_trips["start_seconds"] = static_trips["trip_id"].apply(
    static_start_seconds
)

def fallback_match(unmatched, min_similarity, min_margin=None, method="fallback"):
    rows = []

    unmatched = unmatched.copy()
    unmatched["start_seconds"] = unmatched["trip_id"].apply(
        observed_start_seconds
    )

    for _, obs in unmatched.iterrows():
        obs_sequence = observed_sequences.get(obs["trip_uid"], [])

        if not obs_sequence or pd.isna(obs["start_seconds"]):
            continue

        candidates = static_trips[
            (static_trips["route_id"] == obs["route_id"])
            & (static_trips["direction_id"] == obs["direction_id"])
        ].copy()

        candidates["time_diff_sec"] = (
            candidates["start_seconds"] - obs["start_seconds"]
        ).abs()

        candidates = candidates[
            candidates["time_diff_sec"] <= MAX_MATCH_TIME_SEC
        ]

        scored = []

        for _, candidate in candidates.iterrows():
            sid = candidate["trip_id"]
            seq = static_sequences.get(sid, [])

            if not seq:
                continue

            sim = sequence_similarity(obs_sequence, seq)

            if min_margin is None and sim < min_similarity:
                continue

            scored.append(
                {
                    "static_trip_id": sid,
                    "time_diff_sec": candidate["time_diff_sec"],
                    "sequence_similarity": sim,
                }
            )

        if not scored:
            continue

        scored.sort(
            key=lambda x: (-x["sequence_similarity"], x["time_diff_sec"])
        )

        best = scored[0]

        if best["sequence_similarity"] < min_similarity:
            continue

        if min_margin is not None:
            second = (
                scored[1]["sequence_similarity"]
                if len(scored) > 1
                else 0.0
            )
            if best["sequence_similarity"] - second < min_margin:
                continue

        rows.append(
            {
                "trip_uid": obs["trip_uid"],
                "observed_trip_id": obs["trip_id"],
                "static_trip_id": best["static_trip_id"],
                "route_id": obs["route_id"],
                "direction_id": obs["direction_id"],
                "start_time": obs.get("start_time", np.nan),
                "vehicle_id": obs.get("vehicle_id", np.nan),
                "match_method": method,
                "time_diff_sec": best["time_diff_sec"],
                "sequence_similarity": best["sequence_similarity"],
            }
        )

    return pd.DataFrame(rows)


# Fallback 1: similarity >= 0.80 and margin >= 0.05
unmatched = observed_trips[
    ~observed_trips["trip_uid"].isin(exact["trip_uid"])
].copy()

fallback1 = fallback_match(
    unmatched,
    min_similarity=0.80,
    min_margin=0.05,
    method="time_stop_sequence",
)

matches = pd.concat([exact, fallback1], ignore_index=True)

# Fallback 2: remaining trips, similarity >= 0.90, no margin rule
unmatched = observed_trips[
    ~observed_trips["trip_uid"].isin(matches["trip_uid"])
].copy()

fallback2 = fallback_match(
    unmatched,
    min_similarity=0.90,
    min_margin=None,
    method="sequence_time_relaxed",
)

matches = pd.concat([matches, fallback2], ignore_index=True)
matches = matches.drop_duplicates("trip_uid")

print(
    f"Matched trips: {matches['trip_uid'].nunique()} / "
    f"{observed_trips['trip_uid'].nunique()}"
)


# ============================================================
# 3. BUILD TRAIN-STOP EVENTS
# ============================================================

trip_map = matches[
    [
        "trip_uid",
        "static_trip_id",
        "route_id",
        "direction_id",
        "match_method",
        "sequence_similarity",
        "time_diff_sec",
    ]
].drop_duplicates("trip_uid")

events = observed_stops.merge(
    trip_map,
    on="trip_uid",
    how="inner",
    validate="many_to_one",
)

static_schedule = static_stop_times.rename(
    columns={
        "trip_id": "static_trip_id",
        "arrival_time": "scheduled_arrival_raw",
        "departure_time": "scheduled_departure_raw",
    }
)[
    [
        "static_trip_id",
        "stop_id",
        "stop_sequence",
        "scheduled_arrival_raw",
        "scheduled_departure_raw",
    ]
]

events = events.merge(
    static_schedule,
    on=["static_trip_id", "stop_id"],
    how="left",
    validate="many_to_one",
)

events = events.rename(
    columns={
        "arrival_time": "observed_arrival_raw",
        "departure_time": "observed_departure_raw",
    }
)

events["observed_arrival"] = unix_to_timestamp(
    events["observed_arrival_raw"]
)
events["observed_departure"] = unix_to_timestamp(
    events["observed_departure_raw"]
)

events["scheduled_arrival"] = events[
    "scheduled_arrival_raw"
].apply(gtfs_time_to_timestamp)

events["scheduled_departure"] = events[
    "scheduled_departure_raw"
].apply(gtfs_time_to_timestamp)

events["arrival_delay_sec"] = (
    events["observed_arrival"] - events["scheduled_arrival"]
).dt.total_seconds()

events["departure_delay_sec"] = (
    events["observed_departure"] - events["scheduled_departure"]
).dt.total_seconds()

events["observed_dwell_sec"] = (
    events["observed_departure"] - events["observed_arrival"]
).dt.total_seconds()

events["scheduled_dwell_sec"] = (
    events["scheduled_departure"] - events["scheduled_arrival"]
).dt.total_seconds()

station_cols = [
    c
    for c in ["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"]
    if c in static_stops.columns
]

events = events.merge(
    static_stops[station_cols].drop_duplicates("stop_id"),
    on="stop_id",
    how="left",
    validate="many_to_one",
)

trip_meta_cols = [
    c
    for c in [
        "trip_uid",
        "trip_id",
        "start_time",
        "vehicle_id",
        "last_observed",
        "marked_past",
        "num_updates",
        "num_schedule_changes",
        "num_schedule_rewrites",
    ]
    if c in observed_trips.columns
]

trip_meta = observed_trips[trip_meta_cols].copy()

trip_meta = trip_meta.rename(
    columns={
        "trip_id": "observed_trip_id",
        "last_observed": "trip_last_observed",
        "marked_past": "trip_marked_past",
    }
)

# Avoid duplicate columns already supplied through the match table.
for col in ["start_time", "vehicle_id"]:
    if col in trip_meta.columns and col in events.columns:
        trip_meta = trip_meta.drop(columns=col)

events = events.merge(
    trip_meta,
    on="trip_uid",
    how="left",
    validate="many_to_one",
)

events = events.sort_values(
    ["trip_uid", "stop_sequence"],
    na_position="last",
).reset_index(drop=True)

events["event_id"] = (
    events["trip_uid"].astype(str)
    + "_"
    + events["stop_id"].astype(str)
)


# ============================================================
# 4. CORRECT SERVICE-DAY / MIDNIGHT TIMESTAMPS
# ============================================================

trip_shifts = (
    events.groupby("trip_uid", group_keys=False)
    .apply(choose_day_shift)
    .rename("schedule_day_shift")
)

events = events.merge(
    trip_shifts,
    on="trip_uid",
    how="left",
)

events["scheduled_arrival_corrected"] = (
    events["scheduled_arrival"]
    + pd.to_timedelta(events["schedule_day_shift"], unit="D")
)

events["scheduled_departure_corrected"] = (
    events["scheduled_departure"]
    + pd.to_timedelta(events["schedule_day_shift"], unit="D")
)

events["arrival_delay_sec_corrected"] = (
    events["observed_arrival"]
    - events["scheduled_arrival_corrected"]
).dt.total_seconds()

events["departure_delay_sec_corrected"] = (
    events["observed_departure"]
    - events["scheduled_departure_corrected"]
).dt.total_seconds()

# Event-level ±1 day correction for remaining midnight ambiguity.
events["scheduled_arrival_final"] = [
    nearest_schedule_day(obs, sched)
    for obs, sched in zip(
        events["observed_arrival"],
        events["scheduled_arrival_corrected"],
    )
]

events["scheduled_departure_final"] = [
    nearest_schedule_day(obs, sched)
    for obs, sched in zip(
        events["observed_departure"],
        events["scheduled_departure_corrected"],
    )
]

events["arrival_delay_sec"] = (
    events["observed_arrival"]
    - events["scheduled_arrival_final"]
).dt.total_seconds()

events["departure_delay_sec"] = (
    events["observed_departure"]
    - events["scheduled_departure_final"]
).dt.total_seconds()


# ============================================================
# 5. CLEAN BAD MATCHES / BAD EVENTS
# ============================================================

trip_quality = (
    events.groupby("trip_uid")["arrival_delay_sec"]
    .apply(lambda x: x.abs().median())
)

bad_trip_ids = trip_quality[
    trip_quality > BAD_DELAY_SEC
].index

events = events[
    ~events["trip_uid"].isin(bad_trip_ids)
].copy()

events = events[
    events["arrival_delay_sec"].abs() <= BAD_DELAY_SEC
].copy()

events = events.dropna(
    subset=[
        "trip_uid",
        "stop_id",
        "stop_sequence",
        "scheduled_arrival_final",
    ]
).copy()

events = events.sort_values(
    ["trip_uid", "stop_sequence"]
).reset_index(drop=True)


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

events["current_delay_sec"] = events["arrival_delay_sec"]

events["prev_arrival_delay_sec"] = (
    events.groupby("trip_uid")["arrival_delay_sec"].shift(1)
)

events["prev_departure_delay_sec"] = (
    events.groupby("trip_uid")["departure_delay_sec"].shift(1)
)

events["delay_change_sec"] = (
    events["arrival_delay_sec"]
    - events["prev_arrival_delay_sec"]
)

events["dwell_delay_sec"] = (
    events["observed_dwell_sec"]
    - events["scheduled_dwell_sec"]
)

events["has_positive_dwell"] = (
    events["observed_dwell_sec"] > 0
).astype(int)

prev_departure = (
    events.groupby("trip_uid")["observed_departure"].shift(1)
)

prev_sched_departure = (
    events.groupby("trip_uid")["scheduled_departure_final"].shift(1)
)

events["observed_run_sec"] = (
    events["observed_arrival"] - prev_departure
).dt.total_seconds()

events["scheduled_run_sec"] = (
    events["scheduled_arrival_final"] - prev_sched_departure
).dt.total_seconds()

events["run_delay_sec"] = (
    events["observed_run_sec"]
    - events["scheduled_run_sec"]
)

trip_max_seq = (
    events.groupby("trip_uid")["stop_sequence"].transform("max")
)
trip_min_seq = (
    events.groupby("trip_uid")["stop_sequence"].transform("min")
)

denom = (trip_max_seq - trip_min_seq).replace(0, np.nan)

events["trip_progress"] = (
    events["stop_sequence"] - trip_min_seq
) / denom

events["stops_remaining"] = (
    trip_max_seq - events["stop_sequence"]
)

events["event_time"] = (
    events["observed_arrival"]
    .fillna(events["scheduled_arrival_final"])
)

events["hour"] = (
    events["event_time"].dt.hour
    + events["event_time"].dt.minute / 60.0
)

events["hour_sin"] = np.sin(
    2 * np.pi * events["hour"] / 24
)
events["hour_cos"] = np.cos(
    2 * np.pi * events["hour"] / 24
)

events["weekday"] = events["event_time"].dt.dayofweek
events["is_weekend"] = (
    events["weekday"] >= 5
).astype(int)

events["rush_hour"] = (
    ((events["hour"] >= 7) & (events["hour"] < 10))
    | ((events["hour"] >= 16) & (events["hour"] < 19))
).astype(int)

# ---------- Headway ----------

headway = (
    events[
        [
            "event_id",
            "route_id",
            "direction_id",
            "stop_id",
            "event_time",
        ]
    ]
    .dropna(subset=["event_time"])
    .sort_values(
        ["route_id", "direction_id", "stop_id", "event_time"]
    )
    .copy()
)

group_cols = ["route_id", "direction_id", "stop_id"]

headway["previous_train_time"] = (
    headway.groupby(group_cols)["event_time"].shift(1)
)
headway["next_train_time"] = (
    headway.groupby(group_cols)["event_time"].shift(-1)
)

headway["previous_headway_sec"] = (
    headway["event_time"] - headway["previous_train_time"]
).dt.total_seconds()

headway["following_headway_sec"] = (
    headway["next_train_time"] - headway["event_time"]
).dt.total_seconds()

events = events.merge(
    headway[
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

# ---------- Trailing 10-minute station context ----------

station_groups = []

temp = (
    events[
        [
            "event_id",
            "stop_id",
            "event_time",
            "arrival_delay_sec",
        ]
    ]
    .dropna(subset=["event_time"])
    .sort_values(["stop_id", "event_time"])
)

for stop_id, group in temp.groupby("stop_id"):
    g = group.sort_values("event_time").set_index("event_time")

    g["station_train_count_10m"] = (
        g["event_id"].rolling("10min").count()
    )

    g["station_mean_delay_10m"] = (
        g["arrival_delay_sec"].rolling("10min").mean()
    )

    station_groups.append(
        g.reset_index()[
            [
                "event_id",
                "station_train_count_10m",
                "station_mean_delay_10m",
            ]
        ]
    )

station_features = pd.concat(
    station_groups,
    ignore_index=True,
)

events = events.merge(
    station_features,
    on="event_id",
    how="left",
    validate="one_to_one",
)

events["track_available"] = (
    events["track"].notna().astype(int)
    if "track" in events.columns
    else 0
)

# Impossible negative values are unavailable.
events.loc[
    events["observed_run_sec"] < 0,
    "observed_run_sec",
] = np.nan

events.loc[
    events["scheduled_run_sec"] < 0,
    "scheduled_run_sec",
] = np.nan

events.loc[
    events["previous_headway_sec"] < 0,
    "previous_headway_sec",
] = np.nan

events.loc[
    events["following_headway_sec"] < 0,
    "following_headway_sec",
] = np.nan


# ============================================================
# 7. INITIAL T+1 / T+2 / T+3 TARGETS
# ============================================================

grouped_delay = events.groupby(
    "trip_uid"
)["arrival_delay_sec"]

for k in [1, 2, 3]:
    target_col = f"target_delay_t{k}"

    events[target_col] = grouped_delay.shift(-k)

    events[f"target_mask_t{k}"] = (
        events[target_col].notna().astype("int8")
    )

    events[f"target_delayed_t{k}"] = np.where(
        events[target_col].notna(),
        (events[target_col] >= DELAY_THRESHOLD_SEC).astype(int),
        np.nan,
    )


# ============================================================
# 8. FINAL PRE-GRAPH CLEANING / VALIDITY MASKS
# ============================================================

required_graph_cols = [
    "event_id",
    "trip_uid",
    "stop_id",
    "stop_sequence",
    "route_id",
    "direction_id",
    "event_time",
]

if events[required_graph_cols].isna().sum().sum() > 0:
    raise ValueError(
        "Required graph identifiers contain missing values."
    )

if events["event_id"].duplicated().any():
    raise ValueError("event_id is not unique.")

if events.duplicated(
    subset=["trip_uid", "stop_sequence"]
).any():
    raise ValueError(
        "Duplicate (trip_uid, stop_sequence) events detected."
    )

make_mask(
    events,
    "current_delay_sec",
    "current_delay_valid",
)
make_mask(
    events,
    "prev_arrival_delay_sec",
    "prev_arrival_delay_valid",
)
make_mask(
    events,
    "prev_departure_delay_sec",
    "prev_departure_delay_valid",
)
make_mask(
    events,
    "observed_run_sec",
    "observed_run_valid",
)
make_mask(
    events,
    "previous_headway_sec",
    "previous_headway_valid",
)
make_mask(
    events,
    "following_headway_sec",
    "following_headway_valid",
)
make_mask(
    events,
    "station_mean_delay_10m",
    "station_mean_delay_10m_valid",
)

events["delay_change_valid"] = (
    events["current_delay_valid"].eq(1)
    & events["prev_arrival_delay_valid"].eq(1)
).astype("int8")

valid = events["delay_change_valid"].eq(1)

if not np.allclose(
    events.loc[valid, "delay_change_sec"],
    (
        events.loc[valid, "current_delay_sec"]
        - events.loc[valid, "prev_arrival_delay_sec"]
    ),
    equal_nan=True,
):
    raise ValueError(
        "delay_change_sec consistency check failed."
    )

events["dwell_valid"] = (
    events["observed_dwell_sec"].notna()
    & events["scheduled_dwell_sec"].notna()
    & events["dwell_delay_sec"].notna()
).astype("int8")

valid = events["dwell_valid"].eq(1)

if not np.allclose(
    events.loc[valid, "dwell_delay_sec"],
    (
        events.loc[valid, "observed_dwell_sec"]
        - events.loc[valid, "scheduled_dwell_sec"]
    ),
    equal_nan=True,
):
    raise ValueError(
        "dwell_delay_sec consistency check failed."
    )

run_available = (
    events["observed_run_sec"].notna()
    & events["scheduled_run_sec"].notna()
    & events["run_delay_sec"].notna()
)

expected_run_delay = (
    events["observed_run_sec"]
    - events["scheduled_run_sec"]
)

run_consistent = (
    events["run_delay_sec"] - expected_run_delay
).abs() <= 1e-6

events["run_time_valid"] = (
    run_available & run_consistent
).astype("int8")

run_mismatch = run_available & ~run_consistent
events.loc[
    run_mismatch,
    "run_delay_sec",
] = 0.0

# Only terminal single-stop trips may have undefined progress.
missing_progress = events["trip_progress"].isna()

unresolved = (
    missing_progress
    & events["stops_remaining"].ne(0)
)

if unresolved.any():
    raise ValueError(
        "trip_progress missing for a non-terminal event."
    )

events.loc[
    missing_progress
    & events["stops_remaining"].eq(0),
    "trip_progress",
] = 1.0

# Recreate availability before zero filling.
events["station_mean_delay_10m_valid"] = (
    events["station_mean_delay_10m"]
    .notna()
    .astype("int8")
)

events["following_headway_valid"] = (
    events["following_headway_sec"]
    .notna()
    .astype("int8")
)

masked_features = [
    "current_delay_sec",
    "prev_arrival_delay_sec",
    "prev_departure_delay_sec",
    "delay_change_sec",
    "observed_dwell_sec",
    "dwell_delay_sec",
    "observed_run_sec",
    "run_delay_sec",
    "previous_headway_sec",
    "following_headway_sec",
    "station_mean_delay_10m",
]

events[masked_features] = (
    events[masked_features].fillna(0)
)

for h in [1, 2, 3]:
    target_col = f"target_delay_t{h}"
    mask_col = f"target_mask_t{h}"
    class_col = f"target_delayed_t{h}"

    events[mask_col] = (
        events[mask_col]
        .fillna(0)
        .astype("int8")
    )

    if (
        events[mask_col].eq(1)
        & events[target_col].isna()
    ).any():
        raise ValueError(
            f"{target_col} missing while {mask_col}=1."
        )

    events[target_col] = (
        events[target_col].fillna(0)
    )

    events[class_col] = (
        events[class_col]
        .fillna(0)
        .astype("int8")
    )

events = events.sort_values(
    ["trip_uid", "stop_sequence"]
).reset_index(drop=True)

mask_cols = [
    "delay_change_valid",
    "prev_arrival_delay_valid",
    "observed_run_valid",
    "run_time_valid",
    "prev_departure_delay_valid",
    "dwell_valid",
    "previous_headway_valid",
    "following_headway_valid",
    "current_delay_valid",
    "station_mean_delay_10m_valid",
    "target_mask_t1",
    "target_mask_t2",
    "target_mask_t3",
]

events[mask_cols] = events[mask_cols].astype("int8")

core_features = [
    "current_delay_sec",
    "prev_arrival_delay_sec",
    "prev_departure_delay_sec",
    "delay_change_sec",
    "observed_run_sec",
    "run_delay_sec",
    "observed_dwell_sec",
    "dwell_delay_sec",
    "previous_headway_sec",
    "trip_progress",
    "stops_remaining",
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "rush_hour",
    "station_train_count_10m",
    "station_mean_delay_10m",
    "current_delay_valid",
    "prev_arrival_delay_valid",
    "prev_departure_delay_valid",
    "delay_change_valid",
    "observed_run_valid",
    "run_time_valid",
    "dwell_valid",
    "previous_headway_valid",
    "station_mean_delay_10m_valid",
]

if events[core_features].isna().sum().sum() > 0:
    missing = events[core_features].isna().sum()
    raise ValueError(
        "NaN remains in core features:\n"
        + str(missing[missing > 0])
    )

numeric_core = events[core_features].select_dtypes(
    include=np.number
)

if np.isinf(numeric_core.to_numpy()).any():
    raise ValueError(
        "Infinite values remain in core features."
    )


# ============================================================
# 9. SAVE FINAL CLEANED DATASET
# ============================================================

OUT.mkdir(parents=True, exist_ok=True)

events.to_parquet(
    OUTPUT,
    index=False,
)

print("\n" + "=" * 70)
print("FINAL CLEANED DATASET")
print("=" * 70)
print("Rows:", len(events))
print("Trips:", events["trip_uid"].nunique())
print("Stops:", events["stop_id"].nunique())
print("Routes:", events["route_id"].nunique())
print("Duplicate event IDs:", events["event_id"].duplicated().sum())
print("Bad >2h arrival delays:", (events["arrival_delay_sec"].abs() > BAD_DELAY_SEC).sum())
print("\nSaved to:")
print(OUTPUT)
