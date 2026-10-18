from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PATH = (
    ROOT
    / "data"
    / "interim"
    / "events_2026-08-12_timefixed.parquet"
)

df = pd.read_parquet(PATH)

bad = df[
    df["arrival_delay_sec_corrected"].abs() > 2 * 3600
].copy()

print("Bad rows:", len(bad))
print("Bad trips:", bad["trip_uid"].nunique())

print("\nRows per bad trip:")
print(
    bad.groupby(
        [
            "trip_uid",
            "observed_trip_id",
            "match_method",
            "sequence_similarity",
        ],
        dropna=False,
    )
    .size()
    .sort_values(ascending=False)
    .head(30)
)

cols = [
    "trip_uid",
    "observed_trip_id",
    "static_trip_id",
    "route_id",
    "stop_id",
    "stop_sequence",
    "observed_arrival",
    "scheduled_arrival_corrected",
    "arrival_delay_sec_corrected",
    "match_method",
    "sequence_similarity",
    "time_diff_sec",
]

print("\nBad event details:")
print(
    bad[cols]
    .sort_values(
        ["trip_uid", "stop_sequence"]
    )
    .to_string(index=False)
)
