from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GTFS = ROOT / "data" / "raw" / "gtfs_static"
OBS = ROOT / "data" / "raw" / "subwaydata_2026-08-12"
INTERIM = ROOT / "data" / "interim"


def hhmmss_to_seconds(value):
    if pd.isna(value):
        return None

    value = str(value).zfill(6)

    try:
        h = int(value[:2])
        m = int(value[2:4])
        s = int(value[4:6])
        return h * 3600 + m * 60 + s
    except ValueError:
        return None


def observed_start_seconds(trip_id):
    match = re.match(r"^(\d{6})_", str(trip_id))

    if not match:
        return None

    return hhmmss_to_seconds(match.group(1))


def static_start_seconds(trip_id):
    matches = re.findall(r"_(\d{6})_", str(trip_id))

    if not matches:
        return None

    return hhmmss_to_seconds(matches[-1])


def sequence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------
# Load
# --------------------------------------------------

observed_trips = pd.read_csv(OBS / "trips.csv")
observed_stops = pd.read_csv(OBS / "stop_times.csv")

static_trips = pd.read_csv(GTFS / "trips.txt")
static_stops = pd.read_csv(GTFS / "stop_times.txt")

previous_matches = pd.read_parquet(
    INTERIM / "trip_matches_final_2026-08-12.parquet"
)


# August 12 = weekday
static_trips = static_trips[
    static_trips["service_id"].str.startswith("Weekday", na=False)
].copy()


# --------------------------------------------------
# Only currently unmatched trips
# --------------------------------------------------

matched_uids = set(previous_matches["trip_uid"])

unmatched = observed_trips[
    ~observed_trips["trip_uid"].isin(matched_uids)
].copy()

print("Previous matches:", len(matched_uids))
print("Still unmatched:", unmatched["trip_uid"].nunique())


# --------------------------------------------------
# Start times
# --------------------------------------------------

unmatched["start_seconds"] = unmatched["trip_id"].apply(
    observed_start_seconds
)

static_trips["start_seconds"] = static_trips["trip_id"].apply(
    static_start_seconds
)


# --------------------------------------------------
# Stop sequences
# --------------------------------------------------

observed_sequences = (
    observed_stops
    .dropna(subset=["stop_id"])
    .groupby("trip_uid")["stop_id"]
    .apply(list)
    .to_dict()
)

static_sequences = (
    static_stops
    .dropna(subset=["stop_id"])
    .groupby("trip_id")["stop_id"]
    .apply(list)
    .to_dict()
)


# --------------------------------------------------
# Fallback 2
# --------------------------------------------------

MAX_TIME_DIFFERENCE = 5 * 60
MIN_SEQUENCE_SIMILARITY = 0.90

fallback2_matches = []


for _, obs in unmatched.iterrows():

    obs_uid = obs["trip_uid"]
    obs_sequence = observed_sequences.get(obs_uid, [])

    if not obs_sequence:
        continue

    if pd.isna(obs["start_seconds"]):
        continue

    candidates = static_trips[
        (static_trips["route_id"] == obs["route_id"])
        &
        (static_trips["direction_id"] == obs["direction_id"])
    ].copy()

    if candidates.empty:
        continue

    candidates["time_diff_sec"] = (
        candidates["start_seconds"] - obs["start_seconds"]
    ).abs()

    candidates = candidates[
        candidates["time_diff_sec"] <= MAX_TIME_DIFFERENCE
    ]

    if candidates.empty:
        continue

    scored = []

    for _, candidate in candidates.iterrows():

        static_trip_id = candidate["trip_id"]

        static_sequence = static_sequences.get(
            static_trip_id,
            []
        )

        if not static_sequence:
            continue

        similarity = sequence_similarity(
            obs_sequence,
            static_sequence
        )

        if similarity >= MIN_SEQUENCE_SIMILARITY:

            scored.append(
                {
                    "static_trip_id": static_trip_id,
                    "time_diff_sec": candidate["time_diff_sec"],
                    "sequence_similarity": similarity,
                }
            )

    if not scored:
        continue

    # First choose highest stop similarity.
    # If tied, choose closest scheduled start time.
    scored = sorted(
        scored,
        key=lambda x: (
            -x["sequence_similarity"],
            x["time_diff_sec"]
        )
    )

    best = scored[0]

    fallback2_matches.append(
        {
            "trip_uid": obs_uid,
            "observed_trip_id": obs["trip_id"],
            "static_trip_id": best["static_trip_id"],
            "route_id": obs["route_id"],
            "direction_id": obs["direction_id"],
            "start_time": obs["start_time"],
            "vehicle_id": obs["vehicle_id"],
            "match_method": "sequence_time_relaxed",
            "time_diff_sec": best["time_diff_sec"],
            "sequence_similarity": best["sequence_similarity"],
        }
    )


fallback2 = pd.DataFrame(fallback2_matches)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nFallback 2 matches:", len(fallback2))

if len(fallback2) > 0:

    combined = pd.concat(
        [
            previous_matches,
            fallback2
        ],
        ignore_index=True
    )

    print(
        "Total matched:",
        len(combined),
        "/",
        len(observed_trips)
    )

    print(
        "Overall match rate:",
        round(
            len(combined) / len(observed_trips) * 100,
            2
        ),
        "%"
    )

    print("\nFallback 2 similarity:")
    print(
        fallback2["sequence_similarity"].describe()
    )

    print("\nFallback 2 time difference:")
    print(
        fallback2["time_diff_sec"].describe()
    )

    print("\nSample:")
    print(fallback2.head(10))

    fallback2.to_parquet(
        INTERIM / "trip_matches_fallback2_2026-08-12.parquet",
        index=False
    )

    combined.to_parquet(
        INTERIM / "trip_matches_final_v2_2026-08-12.parquet",
        index=False
    )

    print("\nSaved:")
    print(
        INTERIM /
        "trip_matches_final_v2_2026-08-12.parquet"
    )

else:
    print("No fallback 2 matches found.")

