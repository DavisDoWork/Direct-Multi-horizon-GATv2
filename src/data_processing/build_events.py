from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GTFS = ROOT / "data" / "raw" / "gtfs_static"
OBS = ROOT / "data" / "raw" / "subwaydata_2026-08-12"
INTERIM = ROOT / "data" / "interim"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

SERVICE_DATE = pd.Timestamp("2026-08-12")


def gtfs_time_to_timestamp(value):
    """
    Convert GTFS HH:MM:SS into timestamp on SERVICE_DATE.
    Handles GTFS times > 24:00:00.
    """
    if pd.isna(value):
        return pd.NaT

    value = str(value)

    try:
        h, m, s = map(int, value.split(":"))

        day_offset = h // 24
        hour = h % 24

        return (
            SERVICE_DATE
            + pd.Timedelta(days=day_offset)
            + pd.Timedelta(hours=hour, minutes=m, seconds=s)
        )

    except Exception:
        return pd.NaT


def unix_to_timestamp(value):
    """
    Subway Data NYC observed timestamps are Unix timestamps.
    Convert UTC epoch -> America/New_York local time.
    """
    return pd.to_datetime(
        value,
        unit="s",
        utc=True,
        errors="coerce"
    ).dt.tz_convert("America/New_York").dt.tz_localize(None)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading data...")

observed_trips = pd.read_csv(
    OBS / "trips.csv"
)

observed_stops = pd.read_csv(
    OBS / "stop_times.csv"
)

static_stop_times = pd.read_csv(
    GTFS / "stop_times.txt"
)

static_stops = pd.read_csv(
    GTFS / "stops.txt"
)

matches = pd.read_parquet(
    INTERIM / "trip_matches_final_v2_2026-08-12.parquet"
)


print("Observed trips:", len(observed_trips))
print("Observed stop rows:", len(observed_stops))
print("Matched trips:", len(matches))


# --------------------------------------------------
# 1. Attach static_trip_id to observed stops
# --------------------------------------------------

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
    validate="many_to_one"
)

print("\nObserved stop rows after matched-trip filtering:")
print(len(events))


# --------------------------------------------------
# 2. Prepare static schedule
# --------------------------------------------------

static_schedule = static_stop_times.rename(
    columns={
        "trip_id": "static_trip_id",
        "arrival_time": "scheduled_arrival_raw",
        "departure_time": "scheduled_departure_raw",
    }
)

static_schedule = static_schedule[
    [
        "static_trip_id",
        "stop_id",
        "stop_sequence",
        "scheduled_arrival_raw",
        "scheduled_departure_raw",
    ]
].copy()


# Check duplicate static trip-stop keys
dup_static = static_schedule.duplicated(
    subset=["static_trip_id", "stop_id"],
    keep=False
).sum()

print("Duplicate static trip-stop keys:", dup_static)


# --------------------------------------------------
# 3. Merge observed event with scheduled event
# --------------------------------------------------

events = events.merge(
    static_schedule,
    on=["static_trip_id", "stop_id"],
    how="left",
    validate="many_to_one"
)


# --------------------------------------------------
# 4. Rename observed fields
# --------------------------------------------------

events = events.rename(
    columns={
        "arrival_time": "observed_arrival_raw",
        "departure_time": "observed_departure_raw",
    }
)


# --------------------------------------------------
# 5. Convert timestamps
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Calculate delay
# --------------------------------------------------

events["arrival_delay_sec"] = (
    events["observed_arrival"]
    - events["scheduled_arrival"]
).dt.total_seconds()

events["departure_delay_sec"] = (
    events["observed_departure"]
    - events["scheduled_departure"]
).dt.total_seconds()


# --------------------------------------------------
# 7. Calculate dwell
# --------------------------------------------------

events["observed_dwell_sec"] = (
    events["observed_departure"]
    - events["observed_arrival"]
).dt.total_seconds()

events["scheduled_dwell_sec"] = (
    events["scheduled_departure"]
    - events["scheduled_arrival"]
).dt.total_seconds()


# --------------------------------------------------
# 8. Add station metadata
# --------------------------------------------------

station_meta = static_stops[
    [
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "parent_station",
    ]
].copy()

events = events.merge(
    station_meta,
    on="stop_id",
    how="left",
    validate="many_to_one"
)


# --------------------------------------------------
# 9. Add observed trip metadata
# --------------------------------------------------

trip_meta = observed_trips[
    [
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
].copy()

trip_meta = trip_meta.rename(
    columns={
        "trip_id": "observed_trip_id",
        "last_observed": "trip_last_observed",
        "marked_past": "trip_marked_past",
    }
)

events = events.merge(
    trip_meta,
    on="trip_uid",
    how="left",
    validate="many_to_one"
)


# --------------------------------------------------
# 10. Sort into event order
# --------------------------------------------------

events = events.sort_values(
    ["trip_uid", "stop_sequence"],
    na_position="last"
).reset_index(drop=True)


# --------------------------------------------------
# 11. Create event id
# --------------------------------------------------

events["event_id"] = (
    events["trip_uid"].astype(str)
    + "_"
    + events["stop_id"].astype(str)
)


# --------------------------------------------------
# 12. Quality checks
# --------------------------------------------------

print("\n" + "=" * 70)
print("QUALITY CHECKS")
print("=" * 70)

print("Final event rows:", len(events))
print("Unique matched trips:", events["trip_uid"].nunique())

matched_schedule = events["stop_sequence"].notna().mean() * 100

print(
    "Observed stop rows matched to static schedule:",
    round(matched_schedule, 2),
    "%"
)

print(
    "Missing observed arrival:",
    round(events["observed_arrival"].isna().mean() * 100, 2),
    "%"
)

print(
    "Missing observed departure:",
    round(events["observed_departure"].isna().mean() * 100, 2),
    "%"
)

print(
    "Missing scheduled arrival:",
    round(events["scheduled_arrival"].isna().mean() * 100, 2),
    "%"
)

print(
    "Missing stop_sequence:",
    round(events["stop_sequence"].isna().mean() * 100, 2),
    "%"
)


print("\nArrival delay summary:")
print(events["arrival_delay_sec"].describe())

print("\nDeparture delay summary:")
print(events["departure_delay_sec"].describe())

print("\nObserved dwell summary:")
print(events["observed_dwell_sec"].describe())

print(
    "\nNegative observed dwell count:",
    (events["observed_dwell_sec"] < 0).sum()
)

print(
    "Duplicate event_id count:",
    events["event_id"].duplicated().sum()
)


# --------------------------------------------------
# 13. Save
# --------------------------------------------------

output_path = (
    INTERIM / "events_2026-08-12.parquet"
)

events.to_parquet(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)


# --------------------------------------------------
# 14. Preview important columns
# --------------------------------------------------

preview_cols = [
    "event_id",
    "trip_uid",
    "route_id",
    "direction_id",
    "stop_id",
    "stop_sequence",
    "observed_arrival",
    "scheduled_arrival",
    "arrival_delay_sec",
    "observed_departure",
    "scheduled_departure",
    "departure_delay_sec",
    "observed_dwell_sec",
    "scheduled_dwell_sec",
]

print("\nPreview:")
print(events[preview_cols].head(15))
