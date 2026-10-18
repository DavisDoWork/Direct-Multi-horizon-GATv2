from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GTFS = ROOT / "data" / "raw" / "gtfs_static"
OBS = ROOT / "data" / "raw" / "subwaydata_2026-08-12"
INTERIM = ROOT / "data" / "interim"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def hhmmss_to_seconds(value):
    """Convert HHMMSS string to seconds after midnight."""
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
    """
    Example:
        024200_5..S32R -> 024200 -> seconds
    """
    match = re.match(r"^(\d{6})_", str(trip_id))

    if not match:
        return None

    return hhmmss_to_seconds(match.group(1))


def static_start_seconds(trip_id):
    """
    Example:
        ASP26GEN-5114-Weekday-00_024200_5..S32R
                             ^^^^^^
    """
    matches = re.findall(r"_(\d{6})_", str(trip_id))

    if not matches:
        return None

    return hhmmss_to_seconds(matches[-1])


def sequence_similarity(a, b):
    """
    Compare ordered stop sequences.
    1.0 = identical
    """
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------
# Load
# --------------------------------------------------

observed_trips = pd.read_csv(OBS / "trips.csv")
observed_stops = pd.read_csv(OBS / "stop_times.csv")

static_trips = pd.read_csv(GTFS / "trips.txt")
static_stops = pd.read_csv(GTFS / "stop_times.txt")

exact_matches = pd.read_parquet(
    INTERIM / "trip_matches_2026-08-12.parquet"
)


# August 12, 2026 = weekday
static_trips = static_trips[
    static_trips["service_id"].str.startswith("Weekday", na=False)
].copy()


# --------------------------------------------------
# Find unmatched observed trips
# --------------------------------------------------

matched_uids = set(exact_matches["trip_uid"])

unmatched = observed_trips[
    ~observed_trips["trip_uid"].isin(matched_uids)
].copy()

print("Exact matches:", len(matched_uids))
print("Unmatched trips:", unmatched["trip_uid"].nunique())


# --------------------------------------------------
# Extract start times
# --------------------------------------------------

unmatched["start_seconds"] = unmatched["trip_id"].apply(
    observed_start_seconds
)

static_trips["start_seconds"] = static_trips["trip_id"].apply(
    static_start_seconds
)


# --------------------------------------------------
# Build stop sequences
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
# Fallback matching
# --------------------------------------------------

fallback_matches = []

MAX_TIME_DIFFERENCE = 5 * 60       # 5 minutes
MIN_SEQUENCE_SIMILARITY = 0.80
MIN_MARGIN = 0.05


for i, obs in unmatched.iterrows():

    obs_uid = obs["trip_uid"]
    obs_sequence = observed_sequences.get(obs_uid, [])

    if not obs_sequence:
        continue

    if pd.isna(obs["start_seconds"]):
        continue

    # Candidate static trips:
    # same route + same direction
    candidates = static_trips[
        (static_trips["route_id"] == obs["route_id"])
        &
        (static_trips["direction_id"] == obs["direction_id"])
    ].copy()

    if candidates.empty:
        continue

    candidates["time_diff"] = (
        candidates["start_seconds"] - obs["start_seconds"]
    ).abs()

    candidates = candidates[
        candidates["time_diff"] <= MAX_TIME_DIFFERENCE
    ]

    if candidates.empty:
        continue

    scored = []

    for _, candidate in candidates.iterrows():

        static_trip_id = candidate["trip_id"]
        static_sequence = static_sequences.get(static_trip_id, [])

        if not static_sequence:
            continue

        similarity = sequence_similarity(
            obs_sequence,
            static_sequence
        )

        scored.append(
            {
                "static_trip_id": static_trip_id,
                "time_diff_sec": candidate["time_diff"],
                "sequence_similarity": similarity,
            }
        )

    if not scored:
        continue

    scored = sorted(
        scored,
        key=lambda x: (
            -x["sequence_similarity"],
            x["time_diff_sec"],
        )
    )

    best = scored[0]

    second_similarity = (
        scored[1]["sequence_similarity"]
        if len(scored) > 1
        else 0
    )

    margin = (
        best["sequence_similarity"]
        - second_similarity
    )

    # Conservative acceptance
    if (
        best["sequence_similarity"] >= MIN_SEQUENCE_SIMILARITY
        and margin >= MIN_MARGIN
    ):
        fallback_matches.append(
            {
                "trip_uid": obs_uid,
                "observed_trip_id": obs["trip_id"],
                "static_trip_id": best["static_trip_id"],
                "route_id": obs["route_id"],
                "direction_id": obs["direction_id"],
                "start_time": obs["start_time"],
                "vehicle_id": obs["vehicle_id"],
                "match_method": "time_stop_sequence",
                "time_diff_sec": best["time_diff_sec"],
                "sequence_similarity": best["sequence_similarity"],
            }
        )


fallback = pd.DataFrame(fallback_matches)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nFallback matches:", len(fallback))

if len(fallback) > 0:

    total = len(exact_matches) + len(fallback)

    print(
        "Total matched:",
        total,
        "/",
        len(observed_trips)
    )

    print(
        "Overall match rate:",
        round(
            total / len(observed_trips) * 100,
            2
        ),
        "%"
    )

    print("\nFallback similarity:")
    print(fallback["sequence_similarity"].describe())

    print("\nSample fallback matches:")
    print(fallback.head(10))


    # Make schemas compatible
    exact_matches["time_diff_sec"] = 0
    exact_matches["sequence_similarity"] = 1.0

    combined = pd.concat(
        [
            exact_matches,
            fallback
        ],
        ignore_index=True
    )

    combined.to_parquet(
        INTERIM / "trip_matches_final_2026-08-12.parquet",
        index=False
    )

    fallback.to_parquet(
        INTERIM / "trip_matches_fallback_2026-08-12.parquet",
        index=False
    )

    print("\nSaved final matches:")
    print(
        INTERIM
        / "trip_matches_final_2026-08-12.parquet"
    )

else:
    print("No reliable fallback matches found.")

