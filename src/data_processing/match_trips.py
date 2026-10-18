from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GTFS = ROOT / "data" / "raw" / "gtfs_static"
OBS = ROOT / "data" / "raw" / "subwaydata_2026-08-12"
OUT = ROOT / "data" / "interim"

observed = pd.read_csv(OBS / "trips.csv")
static = pd.read_csv(GTFS / "trips.txt")

# Aug 12, 2026 is a weekday
static = static[static["service_id"].str.startswith("Weekday")].copy()

# Observed trip_id appears as suffix inside static GTFS trip_id
matches = observed.merge(
    static,
    on=["route_id", "direction_id"],
    suffixes=("_obs", "_static")
)

matches["suffix_match"] = matches.apply(
    lambda row: str(row["trip_id_static"]).endswith(str(row["trip_id_obs"])),
    axis=1
)

matched = matches[matches["suffix_match"]].copy()

# Keep only observed trips that matched exactly once
counts = matched.groupby("trip_uid").size()
unique_trip_uids = counts[counts == 1].index

matched = matched[matched["trip_uid"].isin(unique_trip_uids)]

result = matched[
    [
        "trip_uid",
        "trip_id_obs",
        "trip_id_static",
        "route_id",
        "direction_id",
        "start_time",
        "vehicle_id",
    ]
].copy()

result = result.rename(
    columns={
        "trip_id_obs": "observed_trip_id",
        "trip_id_static": "static_trip_id",
    }
)

result["match_method"] = "exact_suffix"

print("Observed trips:", observed["trip_uid"].nunique())
print("Matched trips:", result["trip_uid"].nunique())
print(
    "Match rate:",
    round(result["trip_uid"].nunique() / observed["trip_uid"].nunique() * 100, 2),
    "%"
)

print("\nSample:")
print(result.head(10))

OUT.mkdir(parents=True, exist_ok=True)

result.to_parquet(
    OUT / "trip_matches_2026-08-12.parquet",
    index=False
)

print("\nSaved:")
print(OUT / "trip_matches_2026-08-12.parquet")

