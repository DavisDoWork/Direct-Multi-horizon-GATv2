from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PATH = (
    ROOT
    / "data"
    / "processed"
    / "events_features_2026-08-12.parquet"
)

df = pd.read_parquet(PATH)


# ==================================================
# 1. RUN-TIME OUTLIERS
# ==================================================

print("=" * 70)
print("RUN-TIME CHECK")
print("=" * 70)

for minutes in [5, 10, 20, 60]:
    threshold = minutes * 60

    count = (
        df["observed_run_sec"] > threshold
    ).sum()

    print(
        f"> {minutes:2d} min:",
        count,
        f"({count / len(df) * 100:.4f}%)"
    )


run_bad = df[
    df["observed_run_sec"] > 20 * 60
].copy()

print("\nTrips with >20 min run time:")
print(
    run_bad.groupby("trip_uid")
    .size()
    .sort_values(ascending=False)
    .head(20)
)


print("\nLargest run times:")

cols = [
    "trip_uid",
    "route_id",
    "stop_id",
    "stop_sequence",
    "observed_arrival",
    "observed_departure",
    "observed_run_sec",
    "scheduled_run_sec",
    "run_delay_sec",
]

print(
    run_bad[cols]
    .sort_values(
        "observed_run_sec",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ==================================================
# 2. HEADWAY OUTLIERS
# ==================================================

print("\n" + "=" * 70)
print("HEADWAY CHECK")
print("=" * 70)

for minutes in [10, 20, 30, 60]:
    threshold = minutes * 60

    count = (
        df["previous_headway_sec"] > threshold
    ).sum()

    print(
        f"> {minutes:2d} min:",
        count,
        f"({count / len(df) * 100:.4f}%)"
    )


headway_bad = df[
    df["previous_headway_sec"] > 30 * 60
].copy()

print("\nLargest headways:")

cols = [
    "event_id",
    "route_id",
    "direction_id",
    "stop_id",
    "event_time",
    "previous_headway_sec",
]

print(
    headway_bad[cols]
    .sort_values(
        "previous_headway_sec",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ==================================================
# 3. TARGET CHECK
# ==================================================

print("\n" + "=" * 70)
print("TARGET CONSISTENCY")
print("=" * 70)

for k in [1, 2, 3]:

    mask = df[f"target_mask_t{k}"] == 1

    missing = (
        df.loc[
            mask,
            f"target_delay_t{k}"
        ].isna().sum()
    )

    print(
        f"t+{k} mask=1 but target missing:",
        missing
    )


# ==================================================
# 4. SINGLE-STOP TRIPS
# ==================================================

trip_sizes = (
    df.groupby("trip_uid")
    .size()
)

print("\nSingle-event trips:")
print((trip_sizes == 1).sum())
