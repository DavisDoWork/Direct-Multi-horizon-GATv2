from pathlib import Path
import io
import tarfile
import shutil

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

START_DATE = "2026-05-26"
END_DATE = "2026-08-12"


# Columns that identify each table.
TRIP_REQUIRED = {
    "trip_uid",
    "trip_id",
    "route_id",
    "direction_id",
}

STOP_REQUIRED = {
    "trip_uid",
    "stop_id",
    "stop_sequence",
}


def identify_csv(path):
    """
    Determine whether a CSV is the observed trips table
    or observed stop-times table based on its columns.
    """

    try:
        sample = pd.read_csv(
            path,
            nrows=5,
        )
    except Exception:
        return None

    cols = set(sample.columns)

    if TRIP_REQUIRED.issubset(cols):
        return "trips"

    if STOP_REQUIRED.issubset(cols):
        return "stop_times"

    return None


def download_day(date):
    date_str = date.strftime("%Y-%m-%d")

    output_dir = (
        RAW_DIR
        / f"subwaydata_{date_str}"
    )

    trips_out = output_dir / "trips.csv"
    stops_out = output_dir / "stop_times.csv"

    # Already complete.
    if trips_out.exists() and stops_out.exists():
        print(f"{date_str}: already exists")
        return True

    url = (
        "https://subwaydata.nyc/data/"
        f"subwaydatanyc_{date_str}_csv.tar.xz"
    )

    print(f"{date_str}: downloading...")

    response = requests.get(
        url,
        timeout=120,
    )

    if response.status_code == 404:
        print(f"{date_str}: not available")
        return False

    response.raise_for_status()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Open archive directly from memory.
    with tarfile.open(
        fileobj=io.BytesIO(response.content),
        mode="r:xz",
    ) as tar:

        temp_dir = (
            output_dir
            / "_temp_extract"
        )

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tar.extractall(
            temp_dir,
            filter="data",
        )

    csv_files = list(
        temp_dir.rglob("*.csv")
    )

    if not csv_files:
        print(
            f"{date_str}: no CSV files found"
        )
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )
        return False

    found_trips = None
    found_stops = None

    for csv_path in csv_files:

        table_type = identify_csv(
            csv_path
        )

        if table_type == "trips":
            found_trips = csv_path

        elif table_type == "stop_times":
            found_stops = csv_path

    if found_trips is None:
        print(
            f"{date_str}: observed trips table not found"
        )

    if found_stops is None:
        print(
            f"{date_str}: observed stop-times table not found"
        )

    if (
        found_trips is None
        or found_stops is None
    ):
        print(
            f"{date_str}: files inside archive:"
        )

        for file in csv_files:
            print(
                "   ",
                file.relative_to(temp_dir)
            )

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        return False

    shutil.copy2(
        found_trips,
        trips_out,
    )

    shutil.copy2(
        found_stops,
        stops_out,
    )

    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    print(
        f"{date_str}: complete"
    )

    return True


dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)


success = 0
failed = 0

for date in dates:

    try:

        ok = download_day(date)

        if ok:
            success += 1
        else:
            failed += 1

    except Exception as error:

        print(
            f"{date.date()}: ERROR: {error}"
        )

        failed += 1


print("\n" + "=" * 60)
print("FINISHED")
print("=" * 60)
print("Successful days:", success)
print("Failed/unavailable days:", failed)