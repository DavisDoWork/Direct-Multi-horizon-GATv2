from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GTFS = ROOT / "data" / "raw" / "gtfs_static"
OBS = ROOT / "data" / "raw" / "subwaydata_2026-08-12"

files = {
    "static_trips": GTFS / "trips.txt",
    "static_stop_times": GTFS / "stop_times.txt",
    "static_stops": GTFS / "stops.txt",
    "static_routes": GTFS / "routes.txt",
    "observed_trips": OBS / "trips.csv",
    "observed_stop_times": OBS / "stop_times.csv",
}

for name, path in files.items():
    print("\n" + "=" * 80)
    print(name)
    print(path)

    df = pd.read_csv(path)

    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("\nhead:")
    print(df.head())
