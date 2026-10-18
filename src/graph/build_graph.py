# ============================================================
# graph_construction.py
#
# EVENT-BASED TRAIN GRAPH
#
# Node:
#   ARRIVAL event
#   DEPARTURE event
#
# Edge types:
#   0 = DWELL
#   1 = RUN
#   2 = HEADWAY
#
# Forecast targets:
#   next 1 / 2 / 3 operational events within the same trip
#
# ============================================================

import pandas as pd
import numpy as np
import torch

from pathlib import Path
from torch_geometric.data import Data


# ============================================================
# 0. CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    DATA_DIR
    / "events_pregraph_cleaned.parquet"
)

OUTPUT_GRAPH_PATH = (
    DATA_DIR
    / "event_graph_arr_dep.pt"
)

OUTPUT_NODE_PATH = (
    DATA_DIR
    / "event_nodes_arr_dep.parquet"
)


# ------------------------------------------------------------
# Maximum meaningful headway relationship
# ------------------------------------------------------------

MAX_HEADWAY_SEC = 30 * 60   # 30 minutes


# ============================================================
# 1. LOAD ORIGINAL TRAIN-STOP TABLE
# ============================================================

print("=" * 72)
print("ARRIVAL / DEPARTURE EVENT GRAPH CONSTRUCTION")
print("=" * 72)

print(f"\nLoading: {INPUT_PATH.name}")

df = pd.read_parquet(INPUT_PATH)

df = df.reset_index(drop=True)

print("Original rows:", len(df))
print("Original columns:", df.shape[1])


# ============================================================
# 2. VERIFY REQUIRED COLUMNS
# ============================================================

required_cols = [

    # IDs / topology
    "event_id",
    "trip_uid",
    "route_id",
    "direction_id",
    "stop_id",
    "stop_sequence",

    # Event times
    "observed_arrival",
    "observed_departure",
    "scheduled_arrival_final",
    "scheduled_departure_final",

    # Event delays
    "arrival_delay_sec_corrected",
    "departure_delay_sec_corrected",

    # Operational state
    "observed_dwell_sec",
    "dwell_delay_sec",

    "observed_run_sec",
    "run_delay_sec",

    # Trip context
    "trip_progress",
    "stops_remaining",

    # Station context
    "station_train_count_10m",
    "station_mean_delay_10m",
    "station_mean_delay_10m_valid",

    # Track
    "track_available",

    # Existing target columns only used to infer
    # the classification threshold
    "target_delay_t1",
    "target_delayed_t1",
    "target_mask_t1",
]


missing_cols = [
    col
    for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns:\n{missing_cols}"
    )

print("✓ Required columns available")


# ============================================================
# 3. NORMALIZE DATETIMES
# ============================================================

datetime_cols = [
    "observed_arrival",
    "observed_departure",
    "scheduled_arrival_final",
    "scheduled_departure_final",
]

for col in datetime_cols:
    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )


# ============================================================
# 4. INFER EXISTING DELAY CLASSIFICATION THRESHOLD
# ============================================================
#
# Instead of inventing a new delayed/not-delayed threshold,
# infer which common threshold best reproduces the labels
# already produced by your preprocessing pipeline.
#
# This keeps classification semantics consistent.
# ============================================================

print("\n" + "=" * 72)
print("[1] INFER DELAY CLASSIFICATION THRESHOLD")
print("=" * 72)


candidate_thresholds = [
    0,
    30,
    60,
    120,
    180,
    300,
]


valid_old_target = (
    (df["target_mask_t1"] == 1)
    &
    df["target_delay_t1"].notna()
    &
    df["target_delayed_t1"].notna()
)


best_threshold = None
best_accuracy = -1


for threshold in candidate_thresholds:

    pred = (
        df.loc[
            valid_old_target,
            "target_delay_t1"
        ]
        > threshold
    ).astype(int)

    actual = (
        df.loc[
            valid_old_target,
            "target_delayed_t1"
        ].astype(int)
    )

    accuracy = (
        pred.to_numpy()
        ==
        actual.to_numpy()
    ).mean()

    print(
        f"Threshold > {threshold:>3}s "
        f"agreement = {accuracy:.6f}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold


DELAY_THRESHOLD_SEC = best_threshold


print(
    "\nSelected delay threshold:",
    DELAY_THRESHOLD_SEC,
    "seconds"
)

print(
    "Agreement with old target labels:",
    best_accuracy
)


# ============================================================
# 5. BUILD ARRIVAL NODES
# ============================================================

print("\n" + "=" * 72)
print("[2] CREATE ARRIVAL NODES")
print("=" * 72)


arrival = df.copy()


arrival["event_type"] = "arrival"

arrival["event_order"] = 0

arrival["is_arrival"] = 1.0

arrival["is_departure"] = 0.0


# ------------------------------------------------------------
# Unique event-node ID
# ------------------------------------------------------------

arrival["node_id"] = (
    arrival["event_id"].astype(str)
    + "_ARR"
)


# ------------------------------------------------------------
# Event time
#
# Prefer observed time.
# Fall back to scheduled time if realtime observation missing.
# ------------------------------------------------------------

arrival["node_event_time"] = (
    arrival["observed_arrival"]
    .fillna(
        arrival["scheduled_arrival_final"]
    )
)


# ------------------------------------------------------------
# Current delay of ARRIVAL event
# ------------------------------------------------------------

arrival["node_delay_sec"] = (
    arrival["arrival_delay_sec_corrected"]
)


arrival["current_delay_valid"] = (
    arrival["node_delay_sec"]
    .notna()
    .astype(np.float32)
)


# ------------------------------------------------------------
# ARRIVAL uses RUN information.
#
# Dwell has not occurred yet.
# ------------------------------------------------------------

arrival["node_run_sec"] = (
    arrival["observed_run_sec"]
)

arrival["node_run_delay_sec"] = (
    arrival["run_delay_sec"]
)


arrival["node_dwell_sec"] = 0.0
arrival["node_dwell_delay_sec"] = 0.0


arrival["run_valid"] = (
    arrival["observed_run_sec"]
    .notna()
    .astype(np.float32)
)

arrival["dwell_valid_node"] = 0.0


print(
    "Arrival nodes prepared:",
    len(arrival)
)


# ============================================================
# 6. BUILD DEPARTURE NODES
# ============================================================

print("\n" + "=" * 72)
print("[3] CREATE DEPARTURE NODES")
print("=" * 72)


departure = df.copy()


departure["event_type"] = "departure"

departure["event_order"] = 1

departure["is_arrival"] = 0.0

departure["is_departure"] = 1.0


departure["node_id"] = (
    departure["event_id"].astype(str)
    + "_DEP"
)


departure["node_event_time"] = (
    departure["observed_departure"]
    .fillna(
        departure["scheduled_departure_final"]
    )
)


departure["node_delay_sec"] = (
    departure[
        "departure_delay_sec_corrected"
    ]
)


departure["current_delay_valid"] = (
    departure["node_delay_sec"]
    .notna()
    .astype(np.float32)
)


# ------------------------------------------------------------
# DEPARTURE uses DWELL information.
#
# Run to next station has not occurred yet.
# ------------------------------------------------------------

departure["node_run_sec"] = 0.0
departure["node_run_delay_sec"] = 0.0


departure["node_dwell_sec"] = (
    departure["observed_dwell_sec"]
)

departure["node_dwell_delay_sec"] = (
    departure["dwell_delay_sec"]
)


departure["run_valid"] = 0.0


departure["dwell_valid_node"] = (
    departure["observed_dwell_sec"]
    .notna()
    .astype(np.float32)
)


print(
    "Departure nodes prepared:",
    len(departure)
)


# ============================================================
# 7. COMBINE ARRIVAL + DEPARTURE NODES
# ============================================================

print("\n" + "=" * 72)
print("[4] COMBINE EVENT NODES")
print("=" * 72)


nodes = pd.concat(
    [
        arrival,
        departure
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# Remove nodes that have no usable event time.
#
# Usually very few/none because scheduled time provides
# a fallback.
# ------------------------------------------------------------

missing_event_time = (
    nodes["node_event_time"]
    .isna()
)


print(
    "Nodes without any event time:",
    missing_event_time.sum()
)


nodes = (
    nodes.loc[
        ~missing_event_time
    ]
    .copy()
)


# ============================================================
# 8. SORT INTO TRUE OPERATIONAL EVENT ORDER
# ============================================================

nodes = (
    nodes.sort_values(
        [
            "trip_uid",
            "stop_sequence",
            "event_order",
        ],
        kind="stable"
    )
    .reset_index(drop=True)
)


# New integer node index
nodes["node_idx"] = np.arange(
    len(nodes),
    dtype=np.int64
)


print(
    "Total event nodes:",
    len(nodes)
)

print(
    "Arrival nodes:",
    (nodes["event_type"] == "arrival").sum()
)

print(
    "Departure nodes:",
    (nodes["event_type"] == "departure").sum()
)


# ============================================================
# 9. RECOMPUTE EVENT-SPECIFIC TEMPORAL FEATURES
# ============================================================
#
# Do NOT simply reuse the old row-level hour.
#
# Arrival and departure have their own timestamps.
# ============================================================

print("\n" + "=" * 72)
print("[5] RECOMPUTE EVENT TEMPORAL FEATURES")
print("=" * 72)


nodes["hour"] = (

    nodes["node_event_time"].dt.hour

    +

    nodes["node_event_time"].dt.minute
    / 60.0

    +

    nodes["node_event_time"].dt.second
    / 3600.0
)


nodes["hour_sin"] = np.sin(
    2
    * np.pi
    * nodes["hour"]
    / 24
)


nodes["hour_cos"] = np.cos(
    2
    * np.pi
    * nodes["hour"]
    / 24
)


nodes["weekday"] = (
    nodes["node_event_time"]
    .dt.weekday
    .astype(np.float32)
)


nodes["is_weekend"] = (
    nodes["weekday"] >= 5
).astype(np.float32)


# Keep your existing rush-hour definition if available.
# Here we use typical weekday commuting windows.
nodes["rush_hour"] = (

    (nodes["is_weekend"] == 0)

    &

    (
        (
            (nodes["hour"] >= 7)
            &
            (nodes["hour"] < 10)
        )

        |

        (
            (nodes["hour"] >= 16)
            &
            (nodes["hour"] < 19)
        )
    )

).astype(np.float32)


# ============================================================
# 10. EVENT-SPECIFIC PREVIOUS DELAY
# ============================================================
#
# For ARR:
# previous ARR delay in same trip
#
# For DEP:
# previous DEP delay in same trip
# ============================================================

print("\n" + "=" * 72)
print("[6] BUILD EVENT-SPECIFIC DELAY HISTORY")
print("=" * 72)


nodes["previous_same_type_delay_sec"] = (

    nodes.groupby(
        [
            "trip_uid",
            "event_type"
        ],
        sort=False
    )["node_delay_sec"]
    .shift(1)
)


nodes["previous_delay_valid"] = (

    nodes[
        "previous_same_type_delay_sec"
    ]
    .notna()
    .astype(np.float32)
)


nodes["delay_change_sec"] = (

    nodes["node_delay_sec"]

    -

    nodes[
        "previous_same_type_delay_sec"
    ]
)


nodes["delay_change_valid"] = (

    nodes["node_delay_sec"].notna()

    &

    nodes[
        "previous_same_type_delay_sec"
    ].notna()

).astype(np.float32)


# ============================================================
# 11. RECOMPUTE EVENT-SPECIFIC PREVIOUS HEADWAY
# ============================================================
#
# Same:
# route
# direction
# station
# event type
#
# Example:
#
# ARR train A @ station X
#          ↓
# ARR train B @ station X
#
# or
#
# DEP train A @ station X
#          ↓
# DEP train B @ station X
# ============================================================

print("\n" + "=" * 72)
print("[7] COMPUTE EVENT HEADWAYS")
print("=" * 72)


headway_sorted = (
    nodes.sort_values(
        [
            "route_id",
            "direction_id",
            "stop_id",
            "event_type",
            "node_event_time",
        ]
    )
    .copy()
)


group_cols = [
    "route_id",
    "direction_id",
    "stop_id",
    "event_type",
]


headway_sorted[
    "previous_event_time"
] = (

    headway_sorted
    .groupby(
        group_cols,
        sort=False
    )["node_event_time"]
    .shift(1)
)


headway_sorted[
    "previous_event_node"
] = (

    headway_sorted
    .groupby(
        group_cols,
        sort=False
    )["node_idx"]
    .shift(1)
)


headway_sorted[
    "previous_event_trip"
] = (

    headway_sorted
    .groupby(
        group_cols,
        sort=False
    )["trip_uid"]
    .shift(1)
)


headway_sorted[
    "previous_headway_sec_event"
] = (

    headway_sorted[
        "node_event_time"
    ]

    -

    headway_sorted[
        "previous_event_time"
    ]

).dt.total_seconds()


# ------------------------------------------------------------
# Headway is meaningful only within 30 minutes
# ------------------------------------------------------------

headway_sorted[
    "previous_headway_valid_event"
] = (

    headway_sorted[
        "previous_event_node"
    ].notna()

    &

    (
        headway_sorted[
            "previous_event_trip"
        ]
        !=
        headway_sorted[
            "trip_uid"
        ]
    )

    &

    (
        headway_sorted[
            "previous_headway_sec_event"
        ]
        > 0
    )

    &

    (
        headway_sorted[
            "previous_headway_sec_event"
        ]
        <= MAX_HEADWAY_SEC
    )

).astype(np.float32)


too_long = (

    headway_sorted[
        "previous_headway_sec_event"
    ]
    > MAX_HEADWAY_SEC
)


print(
    "Headways >30 min excluded:",
    too_long.sum()
)


# ------------------------------------------------------------
# For the node feature:
#
# no meaningful previous headway -> 0
# plus validity mask = 0
# ------------------------------------------------------------

headway_sorted.loc[
    headway_sorted[
        "previous_headway_valid_event"
    ] == 0,
    "previous_headway_sec_event"
] = 0.0


# Put headway feature back into original node ordering
headway_feature_map = (
    headway_sorted[
        [
            "node_idx",
            "previous_headway_sec_event",
            "previous_headway_valid_event",
        ]
    ]
    .set_index("node_idx")
)


nodes[
    "previous_headway_sec"
] = (

    nodes["node_idx"]
    .map(
        headway_feature_map[
            "previous_headway_sec_event"
        ]
    )
)


nodes[
    "previous_headway_valid"
] = (

    nodes["node_idx"]
    .map(
        headway_feature_map[
            "previous_headway_valid_event"
        ]
    )
)


# ============================================================
# 12. REGENERATE T+1 / T+2 / T+3 EVENT TARGETS
# ============================================================
#
# Sequence example:
#
# ARR stop 1
# DEP stop 1
# ARR stop 2
# DEP stop 2
# ARR stop 3
# ...
#
# T+1 means NEXT OPERATIONAL EVENT,
# not next original dataframe row.
# ============================================================

print("\n" + "=" * 72)
print("[8] REGENERATE EVENT-LEVEL TARGETS")
print("=" * 72)


for horizon in [1, 2, 3]:

    target_col = (
        f"target_delay_t{horizon}"
    )

    mask_col = (
        f"target_mask_t{horizon}"
    )

    class_col = (
        f"target_delayed_t{horizon}"
    )


    nodes[target_col] = (

        nodes.groupby(
            "trip_uid",
            sort=False
        )["node_delay_sec"]
        .shift(-horizon)
    )


    nodes[mask_col] = (

        nodes[target_col]
        .notna()
        .astype(np.float32)
    )


    nodes[class_col] = (

        (
            nodes[target_col]
            > DELAY_THRESHOLD_SEC
        )

        &

        (
            nodes[mask_col] == 1
        )

    ).astype(np.float32)


    print(
        f"T+{horizon} valid:",
        int(
            nodes[mask_col].sum()
        )
    )


# ============================================================
# 13. BUILD DWELL EDGES
# ============================================================
#
# ARRIVAL(stop i) -> DEPARTURE(stop i)
# ============================================================

print("\n" + "=" * 72)
print("[9] BUILD DWELL EDGES")
print("=" * 72)


node_lookup = dict(
    zip(
        nodes["node_id"],
        nodes["node_idx"]
    )
)


dwell_sources = []
dwell_targets = []
dwell_durations = []
dwell_duration_valids = []

for _, row in df.iterrows():

    arr_id = (
        str(row["event_id"])
        + "_ARR"
    )

    dep_id = (
        str(row["event_id"])
        + "_DEP"
    )

    if (
        arr_id not in node_lookup
        or
        dep_id not in node_lookup
    ):
        continue

    source = node_lookup[arr_id]
    target = node_lookup[dep_id]

    dwell_sec = row["observed_dwell_sec"]

    if pd.isna(dwell_sec):
        dwell_duration_valid = 0.0
        dwell_sec = 0.0
    else:
        dwell_duration_valid = 1.0
        dwell_sec = max(float(dwell_sec), 0.0)

    dwell_sources.append(source)
    dwell_targets.append(target)

    dwell_durations.append(dwell_sec)
    dwell_duration_valids.append(dwell_duration_valid)

print(
    "Dwell edges:",
    len(dwell_sources)
)


# ============================================================
# 14. BUILD RUN EDGES
# ============================================================
#
# DEPARTURE(stop i) -> ARRIVAL(stop i+1)
# ============================================================

print("\n" + "=" * 72)
print("[10] BUILD RUN EDGES")
print("=" * 72)


run_sources = []
run_targets = []
run_durations = []
run_duration_valids = []

skipped_nonconsecutive = 0


for trip_uid, group in df.groupby(
    "trip_uid",
    sort=False
):

    group = (
        group.sort_values(
            "stop_sequence"
        )
        .reset_index(drop=True)
    )


    if len(group) < 2:
        continue


    for i in range(
        len(group) - 1
    ):

        current_row = group.iloc[i]

        next_row = group.iloc[i + 1]


        current_seq = (
            current_row[
                "stop_sequence"
            ]
        )

        next_seq = (
            next_row[
                "stop_sequence"
            ]
        )


        if (
            next_seq
            -
            current_seq
            != 1
        ):

            skipped_nonconsecutive += 1
            continue


        source_id = (
            str(
                current_row[
                    "event_id"
                ]
            )
            + "_DEP"
        )


        target_id = (
            str(
                next_row[
                    "event_id"
                ]
            )
            + "_ARR"
        )


        if (
            source_id not in node_lookup
            or
            target_id not in node_lookup
        ):
            continue


        source = (
            node_lookup[
                source_id
            ]
        )

        target = (
            node_lookup[
                target_id
            ]
        )


        # observed_run_sec belongs to arrival
        # at the next station
        run_sec = (
            next_row[
                "observed_run_sec"
            ]
        )


        if pd.isna(run_sec) or float(run_sec) <= 0:
            run_duration_valid = 0.0
            run_sec = 0.0
        else:
            run_duration_valid = 1.0
            run_sec = float(run_sec)

        run_sources.append(source)
        run_targets.append(target)

        run_durations.append(run_sec)
        run_duration_valids.append(run_duration_valid)


print(
    "Run edges:",
    len(run_sources)
)

print(
    "Skipped non-consecutive stop pairs:",
    skipped_nonconsecutive
)


# ============================================================
# 15. BUILD HEADWAY EDGES
# ============================================================
#
# Previous ARR -> current ARR
# Previous DEP -> current DEP
#
# Same:
# route
# direction
# stop
# event type
#
# Maximum 30 minutes.
# ============================================================

print("\n" + "=" * 72)
print("[11] BUILD HEADWAY EDGES")
print("=" * 72)


valid_headway_rows = (

    headway_sorted.loc[
        headway_sorted[
            "previous_headway_valid_event"
        ] == 1
    ]
)


headway_sources = (

    valid_headway_rows[
        "previous_event_node"
    ]
    .astype(np.int64)
    .tolist()
)


headway_targets = (

    valid_headway_rows[
        "node_idx"
    ]
    .astype(np.int64)
    .tolist()
)


headway_durations = (

    valid_headway_rows[
        "previous_headway_sec_event"
    ]
    .astype(np.float32)
    .tolist()
)

# Every retained headway duration is known and passed the
# 0 < headway <= MAX_HEADWAY_SEC validity rule.
headway_duration_valids = (
    [1.0] * len(headway_sources)
)


print(
    "Headway edges:",
    len(headway_sources)
)


if len(headway_durations) > 0:

    print(
        "Maximum retained headway:",
        max(headway_durations),
        "seconds"
    )

    assert (
        max(headway_durations)
        <= MAX_HEADWAY_SEC
    )


# ============================================================
# 16. COMBINE GRAPH EDGES
# ============================================================

print("\n" + "=" * 72)
print("[12] COMBINE EDGE TYPES")
print("=" * 72)


# Edge type:
#
# 0 = DWELL
# 1 = RUN
# 2 = HEADWAY


all_sources = (
    dwell_sources
    +
    run_sources
    +
    headway_sources
)


all_targets = (
    dwell_targets
    +
    run_targets
    +
    headway_targets
)


edge_types = (

    [0] * len(dwell_sources)

    +

    [1] * len(run_sources)

    +

    [2] * len(headway_sources)
)


# ============================================================
# 17. BUILD EDGE ATTRIBUTES
# ============================================================
#
# Edge attributes:
#
# [
#   is_dwell,
#   is_run,
#   is_headway,
#   duration_sec,
#   duration_valid
# ]
#
# duration_valid = 0 means duration_sec is only a zero placeholder,
# not a genuine observed zero duration.
#
# ============================================================


dwell_attr = np.column_stack(
    [
        np.ones(
            len(dwell_sources),
            dtype=np.float32
        ),

        np.zeros(
            len(dwell_sources),
            dtype=np.float32
        ),

        np.zeros(
            len(dwell_sources),
            dtype=np.float32
        ),

        np.asarray(
            dwell_durations,
            dtype=np.float32
        ),

        np.asarray(
            dwell_duration_valids,
            dtype=np.float32
        ),
    ]
)


run_attr = np.column_stack(
    [
        np.zeros(
            len(run_sources),
            dtype=np.float32
        ),

        np.ones(
            len(run_sources),
            dtype=np.float32
        ),

        np.zeros(
            len(run_sources),
            dtype=np.float32
        ),

        np.asarray(
            run_durations,
            dtype=np.float32
        ),

        np.asarray(
            run_duration_valids,
            dtype=np.float32
        ),
    ]
)


headway_attr = np.column_stack(
    [
        np.zeros(
            len(headway_sources),
            dtype=np.float32
        ),

        np.zeros(
            len(headway_sources),
            dtype=np.float32
        ),

        np.ones(
            len(headway_sources),
            dtype=np.float32
        ),

        np.asarray(
            headway_durations,
            dtype=np.float32
        ),

        np.asarray(
            headway_duration_valids,
            dtype=np.float32
        ),
    ]
)


edge_attr_np = np.vstack(
    [
        dwell_attr,
        run_attr,
        headway_attr,
    ]
)


edge_index = torch.tensor(
    [
        all_sources,
        all_targets
    ],
    dtype=torch.long
)


edge_type = torch.tensor(
    edge_types,
    dtype=torch.long
)


edge_attr = torch.tensor(
    edge_attr_np,
    dtype=torch.float32
)


print(
    "edge_index:",
    edge_index.shape
)

print(
    "edge_type:",
    edge_type.shape
)

print(
    "edge_attr:",
    edge_attr.shape
)


# ============================================================
# 18. NODE FEATURE MATRIX X
# ============================================================

print("\n" + "=" * 72)
print("[13] BUILD NODE FEATURE MATRIX X")
print("=" * 72)


# ------------------------------------------------------------
# Common node feature schema.
#
# ARR and DEP have SAME dimensionality.
#
# Non-applicable operational features are zero with validity
# indicators.
# ------------------------------------------------------------

node_feature_cols = [

    # Current delay state
    "node_delay_sec",

    # Previous same-event-type delay
    "previous_same_type_delay_sec",

    # Change from previous same type
    "delay_change_sec",

    # RUN information
    "node_run_sec",
    "node_run_delay_sec",

    # DWELL information
    "node_dwell_sec",
    "node_dwell_delay_sec",

    # Trip state
    "trip_progress",
    "stops_remaining",

    # Temporal context
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "rush_hour",

    # Headway
    "previous_headway_sec",

    # Station context
    "station_train_count_10m",
    "station_mean_delay_10m",

    # Track availability
    "track_available",

    # Event identity
    "is_arrival",
    "is_departure",

    # Validity masks
    "current_delay_valid",
    "previous_delay_valid",
    "delay_change_valid",
    "run_valid",
    "dwell_valid_node",
    "previous_headway_valid",
    "station_mean_delay_10m_valid",
]


# ------------------------------------------------------------
# Fill unavailable numerical values with 0.
#
# Corresponding validity mask lets the model distinguish:
#
# true 0
# vs
# unavailable value.
# ------------------------------------------------------------

nodes[node_feature_cols] = (

    nodes[node_feature_cols]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)


X_np = (

    nodes[node_feature_cols]
    .to_numpy(
        dtype=np.float32
    )
)


if np.isnan(X_np).any():
    raise ValueError(
        "NaN remains in X"
    )


if np.isinf(X_np).any():
    raise ValueError(
        "Infinite value remains in X"
    )


x = torch.tensor(
    X_np,
    dtype=torch.float32
)


print(
    "X shape:",
    x.shape
)

print(
    "Number of node features:",
    len(node_feature_cols)
)


print("\nNode features:")

for i, col in enumerate(
    node_feature_cols
):

    print(
        f"{i:>2}: {col}"
    )


# ============================================================
# 19. BUILD TARGET TENSORS
# ============================================================

print("\n" + "=" * 72)
print("[14] BUILD MULTI-HORIZON TARGETS")
print("=" * 72)


regression_target_cols = [
    "target_delay_t1",
    "target_delay_t2",
    "target_delay_t3",
]


classification_target_cols = [
    "target_delayed_t1",
    "target_delayed_t2",
    "target_delayed_t3",
]


target_mask_cols = [
    "target_mask_t1",
    "target_mask_t2",
    "target_mask_t3",
]


# Missing regression targets become 0 placeholders.
# target_mask prevents them from contributing to loss.

y_delay = torch.tensor(

    nodes[
        regression_target_cols
    ]
    .fillna(0.0)
    .to_numpy(
        dtype=np.float32
    ),

    dtype=torch.float32
)


y_delayed = torch.tensor(

    nodes[
        classification_target_cols
    ]
    .to_numpy(
        dtype=np.float32
    ),

    dtype=torch.float32
)


target_mask = torch.tensor(

    nodes[
        target_mask_cols
    ]
    .to_numpy(
        dtype=np.bool_
    ),

    dtype=torch.bool
)


print(
    "y_delay:",
    y_delay.shape
)

print(
    "y_delayed:",
    y_delayed.shape
)

print(
    "target_mask:",
    target_mask.shape
)


# ============================================================
# 20. CREATE PYTORCH GEOMETRIC GRAPH
# ============================================================

print("\n" + "=" * 72)
print("[15] CREATE PYTORCH GEOMETRIC DATA")
print("=" * 72)


data = Data(

    x=x,

    edge_index=edge_index,

    edge_type=edge_type,

    edge_attr=edge_attr,

    y_delay=y_delay,

    y_delayed=y_delayed,

    target_mask=target_mask,
)


# ============================================================
# 21. STORE NODE METADATA
# ============================================================
#
# Metadata is NOT part of X.
# ============================================================

data.node_id = (
    nodes["node_id"]
    .astype(str)
    .tolist()
)

data.event_type = (
    nodes["event_type"]
    .astype(str)
    .tolist()
)

data.original_event_id = (
    nodes["event_id"]
    .astype(str)
    .tolist()
)

data.trip_uid = (
    nodes["trip_uid"]
    .astype(str)
    .tolist()
)

data.stop_id = (
    nodes["stop_id"]
    .astype(str)
    .tolist()
)

data.route_id = (
    nodes["route_id"]
    .astype(str)
    .tolist()
)


data.stop_sequence = torch.tensor(

    nodes[
        "stop_sequence"
    ]
    .to_numpy(
        dtype=np.int64
    ),

    dtype=torch.long
)


# Convert event times to Unix nanoseconds
data.event_time_ns = torch.tensor(

    nodes[
        "node_event_time"
    ]
    .astype("int64")
    .to_numpy(),

    dtype=torch.long
)


# ============================================================
# 22. STORE GRAPH METADATA
# ============================================================

data.feature_names = (
    node_feature_cols
)


data.edge_type_names = [
    "dwell",
    "run",
    "headway",
]


data.edge_attr_names = [
    "is_dwell",
    "is_run",
    "is_headway",
    "duration_sec",
    "duration_valid",
]


data.regression_target_names = (
    regression_target_cols
)


data.classification_target_names = (
    classification_target_cols
)


data.delay_threshold_sec = (
    float(
        DELAY_THRESHOLD_SEC
    )
)


data.max_headway_sec = (
    float(
        MAX_HEADWAY_SEC
    )
)


# ============================================================
# 23. VALIDATE GRAPH
# ============================================================

print("\n" + "=" * 72)
print("[16] GRAPH VALIDATION")
print("=" * 72)


num_nodes = (
    data.x.shape[0]
)

num_edges = (
    data.edge_index.shape[1]
)


print(
    "Nodes:",
    num_nodes
)

print(
    "Edges:",
    num_edges
)

print(
    "Features:",
    data.x.shape[1]
)


assert (
    data.edge_index.shape[0]
    == 2
)


assert (
    data.edge_index.shape[1]
    ==
    data.edge_type.shape[0]
)


assert (
    data.edge_index.shape[1]
    ==
    data.edge_attr.shape[0]
)


assert (
    data.edge_attr.shape[1]
    == 5
)


assert (
    data.y_delay.shape
    ==
    (num_nodes, 3)
)


assert (
    data.y_delayed.shape
    ==
    (num_nodes, 3)
)


assert (
    data.target_mask.shape
    ==
    (num_nodes, 3)
)


assert not torch.isnan(
    data.x
).any()


assert not torch.isinf(
    data.x
).any()


assert not torch.isnan(
    data.edge_attr
).any()


assert not torch.isinf(
    data.edge_attr
).any()


assert (
    data.edge_index.min()
    >= 0
)


assert (
    data.edge_index.max()
    < num_nodes
)


print("✓ Graph indices valid")
print("✓ Node features valid")
print("✓ Edge attributes valid")
print("✓ Multi-horizon targets valid")


# ============================================================
# 24. EDGE-TYPE SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("[17] EDGE SUMMARY")
print("=" * 72)


unique_types, counts = torch.unique(
    data.edge_type,
    return_counts=True
)


for edge_id, count in zip(
    unique_types.tolist(),
    counts.tolist()
):

    print(
        f"{data.edge_type_names[edge_id]:<10}"
        f": {count}"
    )


# ============================================================
# 25. SAMPLE EVENT SEQUENCE
# ============================================================

print("\n" + "=" * 72)
print("[18] SAMPLE EVENT SEQUENCE")
print("=" * 72)


sample_trip = (
    nodes[
        "trip_uid"
    ]
    .iloc[0]
)


sample = (

    nodes.loc[
        nodes["trip_uid"]
        == sample_trip,
        [
            "node_idx",
            "trip_uid",
            "stop_sequence",
            "event_type",
            "node_event_time",
            "node_delay_sec",
            "target_delay_t1",
            "target_delay_t2",
            "target_delay_t3",
        ]
    ]
    .head(12)
)


print(
    sample.to_string(
        index=False
    )
)


# ============================================================
# 26. SAMPLE EDGES
# ============================================================

print("\n" + "=" * 72)
print("[19] SAMPLE EDGES")
print("=" * 72)


for edge_type_id, edge_name in enumerate(
    data.edge_type_names
):

    print(
        f"\n--- {edge_name.upper()} ---"
    )


    indices = torch.where(
        data.edge_type
        == edge_type_id
    )[0]


    for idx in indices[:5]:

        src = (
            data.edge_index[
                0,
                idx
            ].item()
        )

        dst = (
            data.edge_index[
                1,
                idx
            ].item()
        )

        duration = (
            data.edge_attr[
                idx,
                3
            ].item()
        )

        duration_valid = (
            data.edge_attr[
                idx,
                4
            ].item()
        )

        print(
            f"{src} "
            f"({data.event_type[src]}, "
            f"stop={data.stop_id[src]})"
            f" -> "
            f"{dst} "
            f"({data.event_type[dst]}, "
            f"stop={data.stop_id[dst]})"
            f" | duration={duration:.1f}s"
            f" | duration_valid={int(duration_valid)}"
        )


# ============================================================
# 27. SAVE NODE TABLE
# ============================================================

node_save_cols = [

    "node_idx",
    "node_id",
    "event_id",

    "trip_uid",
    "route_id",
    "direction_id",

    "stop_id",
    "stop_sequence",

    "event_type",
    "node_event_time",

    "node_delay_sec",

    *node_feature_cols,

    *regression_target_cols,
    *classification_target_cols,
    *target_mask_cols,
]


# remove duplicates while preserving order
node_save_cols = list(
    dict.fromkeys(
        node_save_cols
    )
)


nodes[
    node_save_cols
].to_parquet(
    OUTPUT_NODE_PATH,
    index=False
)


# ============================================================
# 28. SAVE GRAPH
# ============================================================

torch.save(
    data,
    OUTPUT_GRAPH_PATH
)


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("EVENT GRAPH CONSTRUCTION COMPLETE")
print("=" * 72)


print(
    "\nOriginal train-stop rows:",
    len(df)
)


print(
    "Event nodes:",
    num_nodes
)


print(
    "\nArrival nodes:",
    sum(
        e == "arrival"
        for e in data.event_type
    )
)


print(
    "Departure nodes:",
    sum(
        e == "departure"
        for e in data.event_type
    )
)


print(
    "\nDwell edges:",
    len(dwell_sources)
)


print(
    "Run edges:",
    len(run_sources)
)


print(
    "Headway edges:",
    len(headway_sources)
)


print(
    "Total edges:",
    num_edges
)


print(
    "\nMissing dwell durations:",
    int(
        len(dwell_duration_valids)
        - np.sum(dwell_duration_valids)
    )
)

print(
    "Missing run durations:",
    int(
        len(run_duration_valids)
        - np.sum(run_duration_valids)
    )
)


print(
    "\nX:",
    tuple(
        data.x.shape
    )
)


print(
    "Y_delay:",
    tuple(
        data.y_delay.shape
    )
)


print(
    "Y_class:",
    tuple(
        data.y_delayed.shape
    )
)


print(
    "Target mask:",
    tuple(
        data.target_mask.shape
    )
)


print(
    "\nDelay classification threshold:",
    DELAY_THRESHOLD_SEC,
    "seconds"
)


print(
    "Maximum headway edge:",
    MAX_HEADWAY_SEC,
    "seconds"
)


print(
    "\nGraph saved to:"
)

print(
    OUTPUT_GRAPH_PATH
)


print(
    "\nEvent-node table saved to:"
)

print(
    OUTPUT_NODE_PATH
)