from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "interim" / "events_2026-08-12.parquet"
OUTPUT = ROOT / "data" / "interim" / "events_2026-08-12_timefixed.parquet"

df = pd.read_parquet(INPUT)

DAY = 86400


# --------------------------------------------------
# 1. Find best service-day offset PER TRIP
# --------------------------------------------------

def choose_day_shift(group):
    """
    Test schedule shifts of -1, 0, +1 day.
    Choose the shift giving the smallest median
    absolute arrival delay.
    """

    valid = group.dropna(
        subset=["observed_arrival", "scheduled_arrival"]
    )

    if len(valid) == 0:
        return 0

    scores = {}

    for shift in [-1, 0, 1]:

        shifted_schedule = (
            valid["scheduled_arrival"]
            + pd.Timedelta(days=shift)
        )

        delay = (
            valid["observed_arrival"]
            - shifted_schedule
        ).dt.total_seconds()

        scores[shift] = delay.abs().median()

    return min(scores, key=scores.get)


trip_shifts = (
    df.groupby("trip_uid")
    .apply(choose_day_shift)
    .rename("schedule_day_shift")
)

df = df.merge(
    trip_shifts,
    on="trip_uid",
    how="left"
)


# --------------------------------------------------
# 2. Correct scheduled timestamps
# --------------------------------------------------

df["scheduled_arrival_corrected"] = (
    df["scheduled_arrival"]
    + pd.to_timedelta(
        df["schedule_day_shift"],
        unit="D"
    )
)

df["scheduled_departure_corrected"] = (
    df["scheduled_departure"]
    + pd.to_timedelta(
        df["schedule_day_shift"],
        unit="D"
    )
)


# --------------------------------------------------
# 3. Recompute delays
# --------------------------------------------------

df["arrival_delay_sec_corrected"] = (
    df["observed_arrival"]
    - df["scheduled_arrival_corrected"]
).dt.total_seconds()

df["departure_delay_sec_corrected"] = (
    df["observed_departure"]
    - df["scheduled_departure_corrected"]
).dt.total_seconds()


# --------------------------------------------------
# 4. Trip-level quality
# --------------------------------------------------

trip_quality = (
    df.groupby("trip_uid")
    .agg(
        median_abs_delay=(
            "arrival_delay_sec_corrected",
            lambda x: x.abs().median()
        ),
        median_delay=(
            "arrival_delay_sec_corrected",
            "median"
        ),
        num_events=("event_id", "count"),
    )
    .reset_index()
)


# Flag trips still suspicious after date correction
trip_quality["suspicious_trip"] = (
    trip_quality["median_abs_delay"] > 2 * 3600
)

df = df.merge(
    trip_quality[
        [
            "trip_uid",
            "median_abs_delay",
            "suspicious_trip",
        ]
    ],
    on="trip_uid",
    how="left"
)


# --------------------------------------------------
# 5. Diagnostics
# --------------------------------------------------

print("Schedule day shifts:")
print(
    df[
        ["trip_uid", "schedule_day_shift"]
    ]
    .drop_duplicates()
    ["schedule_day_shift"]
    .value_counts()
    .sort_index()
)

print("\nCorrected arrival delay:")
print(
    df["arrival_delay_sec_corrected"]
    .describe()
)

print("\nRemaining > 2 hour rows:")

bad_rows = (
    df["arrival_delay_sec_corrected"].abs()
    > 2 * 3600
).sum()

print(
    bad_rows,
    f"({bad_rows / len(df) * 100:.2f}%)"
)

print("\nSuspicious trips:")
print(
    trip_quality["suspicious_trip"].sum(),
    "/",
    len(trip_quality)
)

print("\nSuspicious by match method:")

print(
    df[
        ["trip_uid", "match_method", "sequence_similarity",
         "suspicious_trip"]
    ]
    .drop_duplicates("trip_uid")
    .groupby(["match_method", "suspicious_trip"])
    .size()
)


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_parquet(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
