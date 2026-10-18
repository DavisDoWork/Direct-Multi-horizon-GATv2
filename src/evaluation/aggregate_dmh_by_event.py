from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    ROOT
    / "data"
    / "processed"
    / "event_graph_arr_dep.pt"
)

NODE_ERROR_PATH = (
    ROOT
    / "results"
    / "dmh_validation_node_errors.csv"
)

HORIZON_MAE_PATH = (
    ROOT
    / "results"
    / "dmh_horizon_mae.csv"
)

OUTPUT_EVENT_PATH = (
    ROOT
    / "results"
    / "dmh_validation_event_aggregated.csv"
)

OUTPUT_SUMMARY_PATH = (
    ROOT
    / "results"
    / "dmh_validation_aggregation_summary.csv"
)

ALIGNMENT_TOLERANCE_SEC = 1e-3

MIN_ALIGNMENT_MATCH_RATE = 0.99


# ============================================================
# HELPER
# ============================================================

def to_numpy_metadata(value):
    """
    Convert graph metadata into a NumPy array.

    Handles:
        torch.Tensor
        np.ndarray
        Python list
    """

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return np.asarray(value)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading graph:")
print(GRAPH_PATH)

data = torch.load(
    GRAPH_PATH,
    weights_only=False,
)


# ============================================================
# LOAD DMH PREDICTIONS
# ============================================================

print("\nLoading DMH node predictions:")
print(NODE_ERROR_PATH)

pred_df = pd.read_csv(
    NODE_ERROR_PATH
)


# ============================================================
# LOAD HORIZON MAE
# ============================================================

print("\nLoading horizon MAE:")
print(HORIZON_MAE_PATH)

mae_df = pd.read_csv(
    HORIZON_MAE_PATH
)


# ============================================================
# FEATURE INDEXES
# ============================================================

feature_names = list(
    data.feature_names
)

if "node_delay_sec" not in feature_names:
    raise ValueError(
        "node_delay_sec not found in feature_names."
    )

if "current_delay_valid" not in feature_names:
    raise ValueError(
        "current_delay_valid not found in feature_names."
    )


delay_idx = feature_names.index(
    "node_delay_sec"
)

delay_valid_idx = feature_names.index(
    "current_delay_valid"
)


# ============================================================
# FULL GRAPH METADATA
# ============================================================

num_nodes = data.x.shape[0]

node_index = np.arange(
    num_nodes,
    dtype=np.int64,
)


trip_uid = to_numpy_metadata(
    data.trip_uid
)

event_time_ns = to_numpy_metadata(
    data.event_time_ns
)

stop_sequence = to_numpy_metadata(
    data.stop_sequence
)

event_type = to_numpy_metadata(
    data.event_type
)

node_id = to_numpy_metadata(
    data.node_id
)


# ============================================================
# CURRENT EVENT DELAY
# ============================================================

node_delay_sec = (
    data.x[
        :,
        delay_idx
    ]
    .float()
    .detach()
    .cpu()
    .numpy()
)

current_delay_valid = (
    data.x[
        :,
        delay_valid_idx
    ]
    .float()
    .detach()
    .cpu()
    .numpy()
    > 0.5
)


# ============================================================
# EVENT TABLE
# ============================================================

event_df = pd.DataFrame(
    {
        "node_index": node_index,
        "trip_uid": trip_uid,
        "event_time_ns": event_time_ns,
        "stop_sequence": stop_sequence,
        "event_type": event_type,
        "node_id": node_id,
        "node_delay_sec": node_delay_sec,
        "current_delay_valid": current_delay_valid,
    }
)


# ============================================================
# ORDER EVENTS WITHIN TRIP
# ============================================================

# ============================================================
# TRUE OPERATIONAL EVENT ORDER
# ============================================================

# IMPORTANT:
# graph_construction_final.py already sorted nodes by:
#
#   trip_uid
#   stop_sequence
#   event_order
#
# and then assigned node_idx sequentially.
#
# Therefore node_index already represents the exact ordering
# used to construct y_delay t+1 / t+2 / t+3.
#
# Do NOT reorder by event_time.

ordered = (
    event_df
    .sort_values(
        by=[
            "trip_uid",
            "node_index",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)


# ============================================================
# SOURCE -> FUTURE TARGET NODE MAP
# ============================================================

future_target = np.full(
    (
        num_nodes,
        3,
    ),
    -1,
    dtype=np.int64,
)


for _, group in ordered.groupby(
    "trip_uid",
    sort=False,
):

    idx = (
        group[
            "node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    n = len(idx)

    for k in range(
        1,
        4,
    ):

        if n <= k:
            continue

        source_nodes = idx[:-k]

        target_nodes = idx[k:]

        future_target[
            source_nodes,
            k - 1
        ] = target_nodes


# ============================================================
# ALIGNMENT CHECK
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "CHECKING FUTURE-EVENT ALIGNMENT"
)

print(
    "=" * 80
)


y_delay_sec = (
    data.y_delay
    .float()
    .detach()
    .cpu()
    .numpy()
)

target_mask = (
    data.target_mask
    .bool()
    .detach()
    .cpu()
    .numpy()
)

target_mask = (
    target_mask
    & np.isfinite(
        y_delay_sec
    )
)


alignment_rows = []


for k in range(3):

    candidate_target = (
        future_target[
            :,
            k
        ]
    )

    has_candidate = (
        candidate_target
        >= 0
    )

    comparable = (
        target_mask[
            :,
            k
        ]
        & has_candidate
    )

    source_idx = np.where(
        comparable
    )[0]

    target_idx = (
        candidate_target[
            source_idx
        ]
    )


    destination_delay = (
        node_delay_sec[
            target_idx
        ]
    )

    destination_valid = (
        current_delay_valid[
            target_idx
        ]
    )

    stored_target = (
        y_delay_sec[
            source_idx,
            k,
        ]
    )


    valid_compare = (
        destination_valid
        & np.isfinite(
            destination_delay
        )
        & np.isfinite(
            stored_target
        )
    )


    destination_delay_valid = (
        destination_delay[
            valid_compare
        ]
    )

    stored_target_valid = (
        stored_target[
            valid_compare
        ]
    )


    abs_difference = np.abs(
        destination_delay_valid
        - stored_target_valid
    )


    matched = (
        abs_difference
        <= ALIGNMENT_TOLERANCE_SEC
    )


    if len(matched) > 0:

        match_rate = float(
            matched.mean()
        )

        mean_abs_difference = float(
            abs_difference.mean()
        )

    else:

        match_rate = np.nan

        mean_abs_difference = np.nan


    alignment_rows.append(
        {
            "horizon":
                f"t+{k + 1}",

            "comparisons":
                len(matched),

            "matches":
                int(
                    matched.sum()
                ),

            "match_rate":
                match_rate,

            "mean_abs_difference_sec":
                mean_abs_difference,
        }
    )


    print(
        f"t+{k + 1} | "
        f"comparisons = {len(matched):7d} | "
        f"match rate = {match_rate:.6f} | "
        f"mean abs diff = "
        f"{mean_abs_difference:.6f} sec"
    )


alignment_df = pd.DataFrame(
    alignment_rows
)


bad_alignment = (
    alignment_df[
        "match_rate"
    ]
    .fillna(0.0)
    < MIN_ALIGNMENT_MATCH_RATE
)


if bad_alignment.any():

    print(
        "\nAlignment check FAILED."
    )

    print(
        "\nThe inferred event ordering does not "
        "reproduce data.y_delay reliably."
    )

    print(
        "\nDo NOT continue with horizon aggregation."
    )

    print(
        "\nAlignment summary:"
    )

    print(
        alignment_df.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Future-event mapping failed alignment check."
    )


print(
    "\nAlignment check PASSED."
)


# ============================================================
# HORIZON MAE
# ============================================================

required_mae_columns = {
    "horizon",
    "mae_seconds",
}

if not required_mae_columns.issubset(
    mae_df.columns
):

    raise ValueError(
        "dmh_horizon_mae.csv must contain "
        "'horizon' and 'mae_seconds'."
    )


mae_lookup = dict(
    zip(
        mae_df[
            "horizon"
        ],
        mae_df[
            "mae_seconds"
        ],
    )
)


for horizon in [
    "t+1",
    "t+2",
    "t+3",
]:

    if horizon not in mae_lookup:

        raise ValueError(
            f"{horizon} missing from "
            "dmh_horizon_mae.csv."
        )


mae_values = np.array(
    [
        float(
            mae_lookup[
                "t+1"
            ]
        ),
        float(
            mae_lookup[
                "t+2"
            ]
        ),
        float(
            mae_lookup[
                "t+3"
            ]
        ),
    ]
)


if not np.all(
    np.isfinite(
        mae_values
    )
):

    raise ValueError(
        "Non-finite horizon MAE found."
    )


if np.any(
    mae_values <= 0
):

    raise ValueError(
        "All horizon MAEs must be > 0."
    )


# ============================================================
# HORIZON WEIGHTS
# ============================================================

inverse_mae = (
    1.0
    / mae_values
)

inverse_mae_weights = (
    inverse_mae
    / inverse_mae.sum()
)

equal_weights = np.array(
    [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ]
)


print(
    "\n"
    + "=" * 80
)

print(
    "HORIZON WEIGHTS"
)

print(
    "=" * 80
)


for k in range(3):

    print(
        f"t+{k + 1} | "
        f"MAE = "
        f"{mae_values[k]:8.3f} sec | "
        f"weight = "
        f"{inverse_mae_weights[k]:.6f}"
    )


# ============================================================
# VALIDATE PREDICTION CSV
# ============================================================

required_prediction_columns = [
    "source_node_index",
    "prediction_t1_sec",
    "prediction_t2_sec",
    "prediction_t3_sec",
]


missing_columns = [
    column
    for column
    in required_prediction_columns
    if column
    not in pred_df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing prediction columns: "
        f"{missing_columns}"
    )


# ============================================================
# SOURCE NODES
# ============================================================

source_nodes = (
    pred_df[
        "source_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


source_predictions = np.column_stack(
    [
        pred_df[
            "prediction_t1_sec"
        ].to_numpy(
            dtype=float
        ),

        pred_df[
            "prediction_t2_sec"
        ].to_numpy(
            dtype=float
        ),

        pred_df[
            "prediction_t3_sec"
        ].to_numpy(
            dtype=float
        ),
    ]
)


validation_node_set = set(
    source_nodes.tolist()
)


# ============================================================
# DESTINATION EVENT RECORDS
# ============================================================

event_records = {}


def ensure_record(
    target_idx,
):

    if target_idx not in event_records:

        event_records[
            target_idx
        ] = {

            "target_node_index":
                int(
                    target_idx
                ),

            "node_id":
                node_id[
                    target_idx
                ],

            "trip_uid":
                trip_uid[
                    target_idx
                ],

            "event_time_ns":
                event_time_ns[
                    target_idx
                ],

            "stop_sequence":
                stop_sequence[
                    target_idx
                ],

            "event_type":
                event_type[
                    target_idx
                ],

            "true_delay_sec":
                node_delay_sec[
                    target_idx
                ],

            "current_delay_valid":
                bool(
                    current_delay_valid[
                        target_idx
                    ]
                ),

            "prediction_t1_sec":
                np.nan,

            "prediction_t2_sec":
                np.nan,

            "prediction_t3_sec":
                np.nan,

            "source_t1_node":
                np.nan,

            "source_t2_node":
                np.nan,

            "source_t3_node":
                np.nan,
        }

    return event_records[
        target_idx
    ]


# ============================================================
# SOURCE -> TARGET REORGANIZATION
# ============================================================

for row_idx, source_idx in enumerate(
    source_nodes
):

    for k in range(3):

        target_idx = (
            future_target[
                source_idx,
                k,
            ]
        )


        if target_idx < 0:
            continue


        # Keep only events inside validation period.
        if (
            target_idx
            not in validation_node_set
        ):
            continue


        record = ensure_record(
            target_idx
        )


        prediction_column = (
            f"prediction_t{k + 1}_sec"
        )

        source_column = (
            f"source_t{k + 1}_node"
        )


        if np.isfinite(
            record[
                prediction_column
            ]
        ):

            raise RuntimeError(
                f"Duplicate prediction found for "
                f"target node {target_idx}, "
                f"horizon t+{k + 1}."
            )


        record[
            prediction_column
        ] = (
            source_predictions[
                row_idx,
                k,
            ]
        )


        record[
            source_column
        ] = int(
            source_idx
        )


# ============================================================
# CREATE TARGET-EVENT DATAFRAME
# ============================================================

aggregated = pd.DataFrame(
    event_records.values()
)


if aggregated.empty:

    raise RuntimeError(
        "No destination-event forecasts created."
    )


aggregated = (
    aggregated
    .sort_values(
        by=[
            "event_time_ns",
            "target_node_index",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# FORECAST MATRIX
# ============================================================

prediction_columns = [
    "prediction_t1_sec",
    "prediction_t2_sec",
    "prediction_t3_sec",
]


prediction_matrix = (
    aggregated[
        prediction_columns
    ]
    .to_numpy(
        dtype=float
    )
)


available = np.isfinite(
    prediction_matrix
)


aggregated[
    "num_forecasts"
] = available.sum(
    axis=1
)


aggregated[
    "has_all_3_forecasts"
] = (
    aggregated[
        "num_forecasts"
    ]
    == 3
)


# ============================================================
# EQUAL-WEIGHT AGGREGATION
# ============================================================

equal_numerator = np.nansum(
    prediction_matrix
    * equal_weights.reshape(
        1,
        3,
    ),
    axis=1,
)


equal_denominator = np.sum(
    available
    * equal_weights.reshape(
        1,
        3,
    ),
    axis=1,
)


equal_final = np.divide(
    equal_numerator,
    equal_denominator,
    out=np.full(
        len(
            aggregated
        ),
        np.nan,
    ),
    where=(
        equal_denominator
        > 0
    ),
)


aggregated[
    "final_equal_weight_sec"
] = equal_final


# ============================================================
# INVERSE-MAE WEIGHTED AGGREGATION
# ============================================================

weighted_numerator = np.nansum(
    prediction_matrix
    * inverse_mae_weights.reshape(
        1,
        3,
    ),
    axis=1,
)


weighted_denominator = np.sum(
    available
    * inverse_mae_weights.reshape(
        1,
        3,
    ),
    axis=1,
)


inverse_mae_final = np.divide(
    weighted_numerator,
    weighted_denominator,
    out=np.full(
        len(
            aggregated
        ),
        np.nan,
    ),
    where=(
        weighted_denominator
        > 0
    ),
)


aggregated[
    "final_inverse_mae_sec"
] = inverse_mae_final


# ============================================================
# TRUE TARGET
# ============================================================

truth = (
    aggregated[
        "true_delay_sec"
    ]
    .to_numpy(
        dtype=float
    )
)


truth_valid = (
    aggregated[
        "current_delay_valid"
    ]
    .to_numpy(
        dtype=bool
    )
    & np.isfinite(
        truth
    )
)


# ============================================================
# ABSOLUTE ERROR
# ============================================================

aggregated[
    "abs_error_equal_sec"
] = np.where(

    truth_valid
    & np.isfinite(
        equal_final
    ),

    np.abs(
        equal_final
        - truth
    ),

    np.nan,
)


aggregated[
    "abs_error_inverse_mae_sec"
] = np.where(

    truth_valid
    & np.isfinite(
        inverse_mae_final
    ),

    np.abs(
        inverse_mae_final
        - truth
    ),

    np.nan,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


def calculate_summary(
    name,
    subset_mask,
):

    subset_mask = (
        subset_mask
        & truth_valid
    )


    equal_errors = (
        aggregated.loc[
            subset_mask,
            "abs_error_equal_sec",
        ]
        .dropna()
        .to_numpy()
    )


    inverse_errors = (
        aggregated.loc[
            subset_mask,
            "abs_error_inverse_mae_sec",
        ]
        .dropna()
        .to_numpy()
    )


    summary_rows.append(
        {
            "subset":
                name,

            "events":
                int(
                    subset_mask.sum()
                ),

            "equal_weight_mae_sec":
                (
                    float(
                        equal_errors.mean()
                    )
                    if len(
                        equal_errors
                    )
                    > 0
                    else np.nan
                ),

            "inverse_mae_weight_mae_sec":
                (
                    float(
                        inverse_errors.mean()
                    )
                    if len(
                        inverse_errors
                    )
                    > 0
                    else np.nan
                ),
        }
    )


# ============================================================
# ALL EVENTS WITH >= 1 FORECAST
# ============================================================

calculate_summary(

    "all_available",

    (
        aggregated[
            "num_forecasts"
        ]
        .to_numpy()
        >= 1
    ),
)


# ============================================================
# ONLY EVENTS WITH ALL 3 FORECASTS
# ============================================================

calculate_summary(

    "all_3_forecasts",

    aggregated[
        "has_all_3_forecasts"
    ]
    .to_numpy(
        dtype=bool
    ),
)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_EVENT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


aggregated.to_csv(
    OUTPUT_EVENT_PATH,
    index=False,
)


summary_df.to_csv(
    OUTPUT_SUMMARY_PATH,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "DESTINATION-EVENT AGGREGATION"
)

print(
    "=" * 80
)


print(
    "\nTarget events with >= 1 forecast:",
    int(
        (
            aggregated[
                "num_forecasts"
            ]
            >= 1
        ).sum()
    ),
)


print(
    "Target events with all 3 forecasts:",
    int(
        aggregated[
            "has_all_3_forecasts"
        ].sum()
    ),
)


print(
    "\nAggregation MAE:"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nSaved event-level aggregation:"
)

print(
    OUTPUT_EVENT_PATH
)


print(
    "\nSaved aggregation summary:"
)

print(
    OUTPUT_SUMMARY_PATH
)


print(
    "\nNOTE:"
)

print(
    "These are validation-set results. "
    "Use them to select the weighting rule, "
    "not as the final test-set comparison."
)