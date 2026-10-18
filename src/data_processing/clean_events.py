from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "interim"
    / "events_2026-08-12_timefixed.parquet"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "events_2026-08-12_clean.parquet"
)

df = pd.read_parquet(INPUT)

print("Starting rows:", len(df))
print("Starting trips:", df["trip_uid"].nunique())


# --------------------------------------------------
# 1. Event-level service-day normalization
# --------------------------------------------------
# The previous correction fixed the schedule per trip.
#
# A small number of trips cross the midnight boundary in
# a way that still leaves individual events ±24 hours off.
#
# For each event, test schedule:
#     -1 day
#      0 day
#     +1 day
#
# Choose the version closest to the observed timestamp.
# This ONLY fixes calendar-day ambiguity.
# It cannot turn a 4-hour bad trip match into a good match.
# --------------------------------------------------

def nearest_schedule_day(row, time_col):
    observed = row[time_col]
    
    if time_col == "observed_arrival":
        scheduled = row["scheduled_arrival_corrected"]
    else:
        scheduled = row["scheduled_departure_corrected"]

    if pd.isna(observed) or pd.isna(scheduled):
        return scheduled

    candidates = [
        scheduled - pd.Timedelta(days=1),
        scheduled,
        scheduled + pd.Timedelta(days=1),
    ]

    best = min(
        candidates,
        key=lambda x: abs(
            (observed - x).total_seconds()
        )
    )

    return best


df["scheduled_arrival_final"] = df.apply(
    lambda row: nearest_schedule_day(
        row,
        "observed_arrival"
    ),
    axis=1
)

df["scheduled_departure_final"] = df.apply(
    lambda row: nearest_schedule_day(
        row,
        "observed_departure"
    ),
    axis=1
)


# --------------------------------------------------
# 2. Recompute final delays
# --------------------------------------------------

df["arrival_delay_sec"] = (
    df["observed_arrival"]
    - df["scheduled_arrival_final"]
).dt.total_seconds()

df["departure_delay_sec"] = (
    df["observed_departure"]
    - df["scheduled_departure_final"]
).dt.total_seconds()


# --------------------------------------------------
# 3. Detect genuinely bad trip matches
# --------------------------------------------------
# Use trip-level median absolute delay.
#
# A whole trip with median error > 2 hours is very
# unlikely to represent an actual NYC subway delay.
# --------------------------------------------------

trip_quality = (
    df.groupby("trip_uid")
    ["arrival_delay_sec"]
    .apply(lambda x: x.abs().median())
    .rename("median_abs_delay")
)

bad_trip_ids = trip_quality[
    trip_quality > 2 * 3600
].index

print("\nBad trips to remove:", len(bad_trip_ids))

for trip in bad_trip_ids:
    print(
        trip,
        "median abs delay:",
        trip_quality.loc[trip]
    )


df = df[
    ~df["trip_uid"].isin(bad_trip_ids)
].copy()


# --------------------------------------------------
# 4. Remove isolated invalid events
# --------------------------------------------------
# After trip-level cleaning, remaining >2-hour events
# are treated as individual observation anomalies.
# --------------------------------------------------

bad_event_mask = (
    df["arrival_delay_sec"].abs()
    > 2 * 3600
)

print(
    "\nIsolated bad events to remove:",
    bad_event_mask.sum()
)

df = df[
    ~bad_event_mask
].copy()


# --------------------------------------------------
# 5. Require usable graph-event information
# --------------------------------------------------

df = df.dropna(
    subset=[
        "trip_uid",
        "stop_id",
        "stop_sequence",
        "scheduled_arrival_final",
    ]
).copy()


# --------------------------------------------------
# 6. Sort
# --------------------------------------------------

df = df.sort_values(
    [
        "trip_uid",
        "stop_sequence"
    ]
).reset_index(drop=True)


# --------------------------------------------------
# 7. Final diagnostics
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CLEAN DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Trips:", df["trip_uid"].nunique())

print("\nArrival delay:")
print(df["arrival_delay_sec"].describe())

print("\nDeparture delay:")
print(df["departure_delay_sec"].describe())

print("\nObserved dwell:")
print(df["observed_dwell_sec"].describe())

print(
    "\nRemaining |arrival delay| > 2h:",
    (
        df["arrival_delay_sec"].abs()
        > 2 * 3600
    ).sum()
)

print(
    "Duplicate event IDs:",
    df["event_id"].duplicated().sum()
)

print(
    "Missing stop sequence:",
    df["stop_sequence"].isna().sum()
)


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
