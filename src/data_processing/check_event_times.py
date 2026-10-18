from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EVENTS = (
    ROOT
    / "data"
    / "interim"
    / "events_2026-08-12.parquet"
)

df = pd.read_parquet(EVENTS)

print("Rows:", len(df))

print("\nDelay counts by absolute size:")

for hours in [1, 2, 4, 8, 12, 20]:
    threshold = hours * 3600

    count = (
        df["arrival_delay_sec"].abs()
        > threshold
    ).sum()

    print(
        f"> {hours} hr:",
        count,
        f"({count / len(df) * 100:.2f}%)"
    )

bad = df[
    df["arrival_delay_sec"].abs() > 6 * 3600
].copy()

cols = [
    "trip_uid",
    "observed_trip_id",
    "static_trip_id",
    "route_id",
    "stop_id",
    "stop_sequence",
    "observed_arrival",
    "scheduled_arrival",
    "arrival_delay_sec",
    "match_method",
    "time_diff_sec",
    "sequence_similarity",
]

print("\nSuspicious negative rows:")
print(
    bad[cols]
    .sort_values("arrival_delay_sec")
    .head(25)
    .to_string(index=False)
)

print("\nSuspicious positive rows:")
print(
    bad[cols]
    .sort_values(
        "arrival_delay_sec",
        ascending=False
    )
    .head(25)
    .to_string(index=False)
)

clean = df[
    df["arrival_delay_sec"].abs() <= 6 * 3600
]

print("\nDelay summary after temporary ±6h filter:")
print(clean["arrival_delay_sec"].describe())
