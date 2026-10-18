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

DMH_NODE_PATH = (
    ROOT
    / "results"
    / "dmh_validation_node_errors.csv"
)

AR_PREDICTION_PATH = (
    ROOT
    / "results"
    / "ar_validation_predictions.npy"
)

DMH_AGGREGATED_PATH = (
    ROOT
    / "results"
    / "dmh_validation_event_aggregated.csv"
)

OUTPUT_METRICS_PATH = (
    ROOT
    / "results"
    / "final_four_model_validation_metrics.csv"
)

OUTPUT_EVENTS_PATH = (
    ROOT
    / "results"
    / "final_four_model_validation_events.csv"
)


# ============================================================
# REGRESSION METRICS
# ============================================================

def calculate_regression_metrics(
    truth,
    prediction,
):

    error = (
        prediction
        - truth
    )

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    mae = float(
        absolute_error.mean()
    )

    rmse = float(
        np.sqrt(
            squared_error.mean()
        )
    )

    return {
        "mae_sec":
            mae,

        "mae_min":
            mae / 60.0,

        "rmse_sec":
            rmse,

        "rmse_min":
            rmse / 60.0,
    }


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_classification_metrics(
    truth_delay,
    prediction_delay,
    delay_threshold_sec,
):
    """
    Convert continuous delay predictions into:

        delayed = delay > threshold

    and calculate:

        Accuracy
        Precision
        Recall
        F1
    """

    true_class = (
        truth_delay
        > delay_threshold_sec
    )

    predicted_class = (
        prediction_delay
        > delay_threshold_sec
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tp = int(
        np.sum(
            predicted_class
            & true_class
        )
    )

    tn = int(
        np.sum(
            (~predicted_class)
            & (~true_class)
        )
    )

    fp = int(
        np.sum(
            predicted_class
            & (~true_class)
        )
    )

    fn = int(
        np.sum(
            (~predicted_class)
            & true_class
        )
    )


    total = (
        tp
        + tn
        + fp
        + fn
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = (
        (tp + tn)
        / total
        if total > 0
        else 0.0
    )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision = (
        tp
        / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall = (
        tp
        / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )


    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,
    }


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading graph:")
print(GRAPH_PATH)

data = torch.load(
    GRAPH_PATH,
    weights_only=False,
)


print(
    "\nGraph nodes:",
    data.x.shape[0],
)


# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================

#
# Use the graph's operational delay threshold.
#
# Current graph:
#
#     delay_threshold_sec = 60.0
#

if hasattr(
    data,
    "delay_threshold_sec"
):

    DELAY_THRESHOLD_SEC = float(
        data.delay_threshold_sec
    )

else:

    DELAY_THRESHOLD_SEC = 60.0


print(
    "\nClassification delay threshold:",
    DELAY_THRESHOLD_SEC,
    "sec",
)


# ============================================================
# CURRENT DELAY FEATURE
# ============================================================

feature_names = list(
    data.feature_names
)


if (
    "node_delay_sec"
    not in feature_names
):

    raise ValueError(
        "node_delay_sec not found."
    )


if (
    "current_delay_valid"
    not in feature_names
):

    raise ValueError(
        "current_delay_valid not found."
    )


delay_idx = (
    feature_names.index(
        "node_delay_sec"
    )
)


delay_valid_idx = (
    feature_names.index(
        "current_delay_valid"
    )
)


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
# GRAPH TARGETS
# ============================================================

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


# ============================================================
# LOAD VALIDATION SOURCE MAPPING
# ============================================================

print(
    "\nLoading validation source mapping:"
)

print(
    DMH_NODE_PATH
)


source_df = pd.read_csv(
    DMH_NODE_PATH
)


if (
    "source_node_index"
    not in source_df.columns
):

    raise ValueError(
        "source_node_index missing."
    )


source_nodes = (
    source_df[
        "source_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


print(
    "Validation source nodes:",
    len(
        source_nodes
    ),
)


# ============================================================
# LOAD AUTOREGRESSIVE PREDICTIONS
# ============================================================

print(
    "\nLoading AR predictions:"
)

print(
    AR_PREDICTION_PATH
)


ar_predictions = np.load(
    AR_PREDICTION_PATH
)


print(
    "AR prediction shape:",
    ar_predictions.shape,
)


if (
    ar_predictions.ndim != 2
    or ar_predictions.shape[1] != 3
):

    raise ValueError(
        "AR predictions must have "
        "shape [N, 3]."
    )


if (
    ar_predictions.shape[0]
    != len(source_nodes)
):

    raise ValueError(
        "AR rows do not match "
        "validation source nodes."
    )


# ============================================================
# SOURCE NODE -> AR ROW
# ============================================================

source_to_ar_row = {

    int(node):
        row

    for row, node
    in enumerate(
        source_nodes
    )
}


# ============================================================
# LOAD AGGREGATED DMH
# ============================================================

print(
    "\nLoading aggregated DMH predictions:"
)

print(
    DMH_AGGREGATED_PATH
)


dmh_df = pd.read_csv(
    DMH_AGGREGATED_PATH
)


required_columns = [

    "target_node_index",

    "true_delay_sec",

    "source_t3_node",

    "has_all_3_forecasts",

    "final_inverse_mae_sec",
]


missing_columns = [

    column

    for column
    in required_columns

    if column
    not in dmh_df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns: "
        f"{missing_columns}"
    )


# ============================================================
# STRICT DMH EVENT SET
# ============================================================

candidate = dmh_df[

    dmh_df[
        "has_all_3_forecasts"
    ].astype(bool)

].copy()


candidate = candidate[
    np.isfinite(
        candidate[
            "true_delay_sec"
        ]
    )
]


candidate = candidate[
    np.isfinite(
        candidate[
            "final_inverse_mae_sec"
        ]
    )
]


candidate = candidate[
    np.isfinite(
        candidate[
            "source_t3_node"
        ]
    )
]


candidate = candidate.reset_index(
    drop=True
)


print(
    "\nDMH events with all 3 forecasts:",
    len(candidate),
)


# ============================================================
# BUILD COMMON EVENT DATASET
# ============================================================

records = []


for _, row in candidate.iterrows():


    # --------------------------------------------------------
    # TARGET EVENT
    # --------------------------------------------------------

    target_node = int(
        row[
            "target_node_index"
        ]
    )


    # --------------------------------------------------------
    # t+3 ORIGIN EVENT
    # --------------------------------------------------------

    source_t3 = int(
        row[
            "source_t3_node"
        ]
    )


    # --------------------------------------------------------
    # AR MUST EXIST
    # --------------------------------------------------------

    if (
        source_t3
        not in source_to_ar_row
    ):

        continue


    ar_row = (
        source_to_ar_row[
            source_t3
        ]
    )


    # ========================================================
    # AUTOREGRESSIVE
    # ========================================================

    #
    # Native recursive t+3 prediction
    #

    ar_prediction = float(

        ar_predictions[
            ar_row,
            2
        ]
    )


    if not np.isfinite(
        ar_prediction
    ):

        continue


    # ========================================================
    # PERSISTENCE
    # ========================================================

    #
    # Native persistence:
    #
    # y_hat(t+3) = current delay at source_t3
    #

    if not current_delay_valid[
        source_t3
    ]:

        continue


    persistence_prediction = float(

        node_delay_sec[
            source_t3
        ]
    )


    if not np.isfinite(
        persistence_prediction
    ):

        continue


    # ========================================================
    # DMH FINAL
    # ========================================================

    dmh_prediction = float(

        row[
            "final_inverse_mae_sec"
        ]
    )


    if not np.isfinite(
        dmh_prediction
    ):

        continue


    # ========================================================
    # ZERO BASELINE
    # ========================================================

    zero_prediction = 0.0


    # ========================================================
    # TRUE TARGET
    # ========================================================

    truth = float(

        row[
            "true_delay_sec"
        ]
    )


    if not np.isfinite(
        truth
    ):

        continue


    # ========================================================
    # TARGET SANITY CHECK
    # ========================================================

    #
    # source_t3's t+3 target must equal
    # the DMH target event.
    #

    if not target_mask[
        source_t3,
        2
    ]:

        continue


    graph_t3_truth = float(

        y_delay_sec[
            source_t3,
            2
        ]
    )


    if (
        abs(
            graph_t3_truth
            - truth
        )
        > 1e-3
    ):

        raise RuntimeError(

            f"Target mismatch:\n"

            f"source={source_t3}\n"

            f"target={target_node}\n"

            f"graph t+3 truth="
            f"{graph_t3_truth}\n"

            f"event truth="
            f"{truth}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    records.append(
        {

            "target_node_index":
                target_node,

            "source_t3_node":
                source_t3,

            "true_delay_sec":
                truth,

            "zero_prediction_sec":
                zero_prediction,

            "persistence_prediction_sec":
                persistence_prediction,

            "ar_t3_prediction_sec":
                ar_prediction,

            "dmh_final_prediction_sec":
                dmh_prediction,
        }
    )


# ============================================================
# FINAL COMMON EVENT TABLE
# ============================================================

results = pd.DataFrame(
    records
)


if results.empty:

    raise RuntimeError(
        "No common evaluation events."
    )


print(
    "\n"
    + "=" * 80
)

print(
    "COMMON FINAL EVENT SET"
)

print(
    "=" * 80
)


print(
    "\nEvents used for all four models:",
    len(results),
)


# ============================================================
# TRUE VALUES
# ============================================================

truth = (

    results[
        "true_delay_sec"
    ]

    .to_numpy(
        dtype=float
    )
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

model_predictions = {

    "Zero":

        results[
            "zero_prediction_sec"
        ].to_numpy(
            dtype=float
        ),


    "Persistence":

        results[
            "persistence_prediction_sec"
        ].to_numpy(
            dtype=float
        ),


    "Autoregressive_GATv2":

        results[
            "ar_t3_prediction_sec"
        ].to_numpy(
            dtype=float
        ),


    "DMH_GATv2_Aggregated":

        results[
            "dmh_final_prediction_sec"
        ].to_numpy(
            dtype=float
        ),
}


# ============================================================
# TRUE CLASSIFICATION LABEL
# ============================================================

true_delayed = (
    truth
    > DELAY_THRESHOLD_SEC
)


results[
    "true_delayed"
] = true_delayed.astype(
    int
)


print(
    "\nTrue delayed events:",
    int(
        true_delayed.sum()
    ),
)


print(
    "True non-delayed events:",
    int(
        (
            ~true_delayed
        ).sum()
    ),
)


print(
    "Delayed prevalence:",
    f"{true_delayed.mean():.4f}",
)


# ============================================================
# METRIC CALCULATION
# ============================================================

metric_rows = []


for (
    model_name,
    prediction
) in model_predictions.items():


    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    regression_metrics = (
        calculate_regression_metrics(
            truth,
            prediction,
        )
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classification_metrics = (
        calculate_classification_metrics(
            truth,
            prediction,
            DELAY_THRESHOLD_SEC,
        )
    )


    metric_rows.append(
        {

            "model":
                model_name,

            "n_events":
                len(truth),

            "mae_sec":
                regression_metrics[
                    "mae_sec"
                ],

            "mae_min":
                regression_metrics[
                    "mae_min"
                ],

            "rmse_sec":
                regression_metrics[
                    "rmse_sec"
                ],

            "rmse_min":
                regression_metrics[
                    "rmse_min"
                ],

            "accuracy":
                classification_metrics[
                    "accuracy"
                ],

            "precision":
                classification_metrics[
                    "precision"
                ],

            "recall":
                classification_metrics[
                    "recall"
                ],

            "f1_score":
                classification_metrics[
                    "f1_score"
                ],

            "tp":
                classification_metrics[
                    "tp"
                ],

            "tn":
                classification_metrics[
                    "tn"
                ],

            "fp":
                classification_metrics[
                    "fp"
                ],

            "fn":
                classification_metrics[
                    "fn"
                ],
        }
    )


# ============================================================
# METRIC TABLE
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)


metrics_df = metrics_df[
    [

        "model",

        "n_events",

        "mae_sec",

        "mae_min",

        "rmse_sec",

        "rmse_min",

        "accuracy",

        "precision",

        "recall",

        "f1_score",

        "tp",

        "tn",

        "fp",

        "fn",
    ]
]


metrics_df = (

    metrics_df

    .sort_values(
        "mae_sec"
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# ADD EVENT-LEVEL CLASSIFICATION PREDICTIONS
# ============================================================

results[
    "zero_predicted_delayed"
] = (
    results[
        "zero_prediction_sec"
    ]
    > DELAY_THRESHOLD_SEC
).astype(int)


results[
    "persistence_predicted_delayed"
] = (
    results[
        "persistence_prediction_sec"
    ]
    > DELAY_THRESHOLD_SEC
).astype(int)


results[
    "ar_predicted_delayed"
] = (
    results[
        "ar_t3_prediction_sec"
    ]
    > DELAY_THRESHOLD_SEC
).astype(int)


results[
    "dmh_predicted_delayed"
] = (
    results[
        "dmh_final_prediction_sec"
    ]
    > DELAY_THRESHOLD_SEC
).astype(int)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics_df.to_csv(
    OUTPUT_METRICS_PATH,
    index=False,
)


results.to_csv(
    OUTPUT_EVENTS_PATH,
    index=False,
)


# ============================================================
# PAPER STYLE TABLE
# ============================================================

paper_table = (
    metrics_df[
        [

            "model",

            "mae_min",

            "rmse_min",

            "accuracy",

            "precision",

            "recall",

            "f1_score",
        ]
    ]
    .copy()
)


paper_table.columns = [

    "Model",

    "MAE",

    "RMSE",

    "Accuracy",

    "Precision",

    "Recall",

    "F1-Score",
]


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "OVERALL MODEL PERFORMANCE COMPARISON"
)

print(
    "=" * 95
)


print(
    "\n"
    + paper_table.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\n"
    + "=" * 95
)

print(
    "FULL METRICS + CONFUSION MATRIX"
)

print(
    "=" * 95
)


print(
    "\n"
    + metrics_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nSaved metrics:"
)

print(
    OUTPUT_METRICS_PATH
)


print(
    "\nSaved event-level results:"
)

print(
    OUTPUT_EVENTS_PATH
)


print(
    "\nClassification rule:"
)

print(
    f"  delayed = delay > "
    f"{DELAY_THRESHOLD_SEC:.1f} sec"
)


print(
    "\nAll models evaluated on "
    "the exact same target events."
)


print(
    "\nNOTE:"
)

print(
    "These are validation results, "
    "not final held-out test results."
)