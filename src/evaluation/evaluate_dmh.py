from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from torch_geometric.utils import subgraph

from src.models.gatv2_dmh import (
    GATv2DMHClassifier,
    GATv2DMHRegressor,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


GRAPH_PATH = (
    ROOT
    / "data"
    / "processed"
    / "event_graph_arr_dep.pt"
)


DMH_CHECKPOINT_PATH = (
    ROOT
    / "results"
    / "gatv2_dmh_hurdle.pt"
)


#
# AR predictions must have:
#
# shape = [N_validation, 3]
#
# units = seconds
#
AR_PREDICTION_PATH = (
    ROOT
    / "results"
    / "ar_validation_predictions.npy"
)


RESULT_PATH = (
    ROOT
    / "results"
    / "four_model_validation_results.csv"
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("Device:", device)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return mae, rmse


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading graph:")

print(GRAPH_PATH)


data = torch.load(
    GRAPH_PATH,
    weights_only=False,
)


print("\nGraph:")

print(data)


print(
    "\nx:",
    data.x.shape,
)

print(
    "edge_index:",
    data.edge_index.shape,
)

print(
    "y_delay:",
    data.y_delay.shape,
)


# ============================================================
# LOAD DMH CHECKPOINT
# ============================================================

print("\nLoading DMH checkpoint:")

print(DMH_CHECKPOINT_PATH)


checkpoint = torch.load(
    DMH_CHECKPOINT_PATH,
    weights_only=False,
)


feature_names = list(
    checkpoint[
        "feature_names"
    ]
)


feature_mean = checkpoint[
    "feature_mean"
].float()


feature_std = checkpoint[
    "feature_std"
].float()


cutoff_time = checkpoint[
    "cutoff_time"
]


classification_threshold = checkpoint.get(
    "classification_threshold",
    0.5,
)


# ============================================================
# FIND CURRENT DELAY FEATURES
# ============================================================

if (
    "node_delay_sec"
    not in feature_names
):

    raise ValueError(
        "node_delay_sec not found in feature_names."
    )


current_delay_idx = feature_names.index(
    "node_delay_sec"
)


if (
    "current_delay_valid"
    not in feature_names
):

    raise ValueError(
        "current_delay_valid not found in feature_names."
    )


current_delay_valid_idx = feature_names.index(
    "current_delay_valid"
)


# ============================================================
# RAW DATA
# ============================================================

x_raw = data.x.float()


y_delay_sec = data.y_delay.float()


target_mask = (
    data.target_mask.bool()
    & torch.isfinite(
        y_delay_sec
    )
)


event_time = (
    data.event_time_ns
    .cpu()
    .numpy()
)


# ============================================================
# SAME VALIDATION SPLIT
# ============================================================

val_nodes = torch.tensor(
    event_time > cutoff_time,
    dtype=torch.bool,
)


val_ids = torch.where(
    val_nodes
)[0]


print(
    "\nValidation nodes:",
    len(val_ids),
)


# ============================================================
# VALIDATION SUBGRAPH
# ============================================================

val_edge_index, _ = subgraph(

    subset=val_ids,

    edge_index=data.edge_index,

    relabel_nodes=True,

    num_nodes=data.x.shape[0],
)


val_x_raw = (
    x_raw[
        val_ids
    ]
    .clone()
)


val_y = (
    y_delay_sec[
        val_ids
    ]
    .clone()
)


val_target_mask = (
    target_mask[
        val_ids
    ]
    .clone()
)


print(
    "Validation edges:",
    val_edge_index.shape[1],
)


# ============================================================
# PERSISTENCE
# ============================================================

#
# Use RAW current delay.
#

val_current_delay = (
    val_x_raw[
        :,
        current_delay_idx
    ]
    .clone()
)


val_current_delay_valid = (
    val_x_raw[
        :,
        current_delay_valid_idx
    ] > 0.5
)


# ============================================================
# NORMALIZE FEATURES FOR DMH
# ============================================================

val_x = (
    val_x_raw
    - feature_mean
) / feature_std


# ============================================================
# BUILD DMH CLASSIFIER
# ============================================================

classifier = GATv2DMHClassifier(

    in_channels=checkpoint[
        "in_channels"
    ],

    hidden_channels=checkpoint[
        "hidden_channels"
    ],

    heads=checkpoint[
        "heads"
    ],

    dropout=checkpoint[
        "dropout"
    ],

    horizons=checkpoint.get(
        "horizons",
        3,
    ),
)


classifier.load_state_dict(
    checkpoint[
        "classifier_state_dict"
    ]
)


classifier = classifier.to(
    device
)


classifier.eval()


# ============================================================
# BUILD DMH REGRESSOR
# ============================================================

regressor = GATv2DMHRegressor(

    in_channels=checkpoint[
        "in_channels"
    ],

    hidden_channels=checkpoint[
        "hidden_channels"
    ],

    heads=checkpoint[
        "heads"
    ],

    dropout=checkpoint[
        "dropout"
    ],

    horizons=checkpoint.get(
        "horizons",
        3,
    ),
)


regressor.load_state_dict(
    checkpoint[
        "regressor_state_dict"
    ]
)


regressor = regressor.to(
    device
)


regressor.eval()


# ============================================================
# MOVE VALIDATION GRAPH TO DEVICE
# ============================================================

val_x = val_x.to(
    device
)


val_edge_index = val_edge_index.to(
    device
)


# ============================================================
# DMH TWO-STAGE PREDICTION
# ============================================================

print(
    "\nRunning DMH predictions..."
)


with torch.no_grad():

    # --------------------------------------------------------
    # Stage 1:
    # delayed / not delayed
    # --------------------------------------------------------

    logits = classifier(
        val_x,
        val_edge_index,
    )


    probabilities = torch.sigmoid(
        logits
    )


    predicted_delayed = (
        probabilities
        >= classification_threshold
    )


    # --------------------------------------------------------
    # Stage 2:
    # transformed delay magnitude
    # --------------------------------------------------------

    predicted_log_delay = regressor(
        val_x,
        val_edge_index,
    )


    #
    # Training target was:
    #
    # log1p(delay_minutes)
    #
    # therefore invert using expm1().
    #

    predicted_delay_min = torch.expm1(
        predicted_log_delay
    )


    predicted_delay_min = torch.clamp(
        predicted_delay_min,
        min=0.0,
    )


    # --------------------------------------------------------
    # Hurdle
    # --------------------------------------------------------

    dmh_prediction_min = torch.where(

        predicted_delayed,

        predicted_delay_min,

        torch.zeros_like(
            predicted_delay_min
        ),
    )


    #
    # minutes -> seconds
    #

    dmh_prediction_sec = (
        dmh_prediction_min
        * 60.0
    )


# ============================================================
# CPU / NUMPY
# ============================================================

dmh_prediction_np = (
    dmh_prediction_sec
    .cpu()
    .numpy()
)


y_true_all = (
    val_y
    .cpu()
    .numpy()
)


target_mask_np = (
    val_target_mask
    .cpu()
    .numpy()
)


current_delay_np = (
    val_current_delay
    .cpu()
    .numpy()
)


current_delay_valid_np = (
    val_current_delay_valid
    .cpu()
    .numpy()
)


# ============================================================
# LOAD AUTOREGRESSIVE PREDICTIONS
# ============================================================

if not AR_PREDICTION_PATH.exists():

    raise FileNotFoundError(
        "\nAR predictions not found:\n"
        f"{AR_PREDICTION_PATH}\n\n"
        "Expected a NumPy array with shape "
        f"({len(val_ids)}, 3) in SECONDS."
    )


print(
    "\nLoading AR predictions:"
)

print(
    AR_PREDICTION_PATH
)


ar_prediction_np = np.load(
    AR_PREDICTION_PATH
)


# ============================================================
# AR SHAPE CHECK
# ============================================================

expected_shape = (
    len(val_ids),
    3,
)


if (
    ar_prediction_np.shape
    != expected_shape
):

    raise ValueError(
        "\nAR prediction shape mismatch.\n"
        f"Expected: {expected_shape}\n"
        f"Received: {ar_prediction_np.shape}"
    )


# ============================================================
# FAIR FOUR-MODEL COMPARISON
# ============================================================

results = []

DELAY_THRESHOLD_SEC = 60.0

for k in range(3):

    # --------------------------------------------------------
    # SAME OBSERVATIONS FOR EVERY MODEL
    # --------------------------------------------------------

    common_mask = (
        target_mask_np[:, k]
        & current_delay_valid_np
        & np.isfinite(y_true_all[:, k])
        & np.isfinite(current_delay_np)
        & np.isfinite(dmh_prediction_np[:, k])
        & np.isfinite(ar_prediction_np[:, k])
    )

    # True future delay for horizon k
    y_true = y_true_all[
        common_mask,
        k
    ]

    # --------------------------------------------------------
    # ZERO DELAY
    # --------------------------------------------------------

    zero_pred = np.zeros(
        len(y_true),
        dtype=float,
    )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    persistence_pred = current_delay_np[
        common_mask
    ]

    # --------------------------------------------------------
    # AUTOREGRESSIVE GATv2
    # --------------------------------------------------------

    ar_pred = ar_prediction_np[
        common_mask,
        k
    ]

    # --------------------------------------------------------
    # DMH GATv2
    # --------------------------------------------------------

    dmh_pred = dmh_prediction_np[
        common_mask,
        k
    ]

    # --------------------------------------------------------
    # MODEL DICTIONARY
    # --------------------------------------------------------

    model_predictions = {
        "Zero-delay": zero_pred,
        "Persistence": persistence_pred,
        "AR-GATv2": ar_pred,
        "DMH-GATv2": dmh_pred,
    }

    # --------------------------------------------------------
    # TRUE CLASS
    # --------------------------------------------------------

    y_true_class = (
        y_true >= DELAY_THRESHOLD_SEC
    ).astype(int)

    # --------------------------------------------------------
    # METRICS FOR EACH MODEL AT THIS HORIZON
    # --------------------------------------------------------

    for model_name, prediction in model_predictions.items():

        # Regression
        mae, rmse = calculate_metrics(
            y_true,
            prediction,
        )

        # Classification
        prediction_class = (
            prediction >= DELAY_THRESHOLD_SEC
        ).astype(int)

        accuracy = np.mean(
            prediction_class == y_true_class
        )

        results.append(
            {
                "model": model_name,
                "horizon": f"t+{k + 1}",
                "mae_sec": mae,
                "rmse_sec": rmse,
                "mae_min": mae / 60.0,
                "rmse_min": rmse / 60.0,
                "accuracy": accuracy,
                "n": len(y_true),
            }
        )

# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT LONG FORMAT
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "FOUR-MODEL COMPARISON"
)


print(
    "=" * 100
)


print(

    results_df[
        [
            "model",
            "horizon",
            "mae_sec",
            "rmse_sec",
            "mae_min",
            "rmse_min",
            "accuracy",
            "n",
        ]
    ]

    .to_string(

        index=False,

        float_format=lambda x:
        f"{x:.3f}",
    )
)


# ============================================================
# MAE TABLE
# ============================================================

mae_table = results_df.pivot(

    index="model",

    columns="horizon",

    values="mae_min",
)


mae_table = mae_table.reindex(
    [
        "Zero-delay",
        "Persistence",
        "AR-GATv2",
        "DMH-GATv2",
    ]
)


print(
    "\n"
    + "=" * 70
)


print(
    "MAE (MINUTES)"
)


print(
    "=" * 70
)


print(
    mae_table.to_string(
        float_format=lambda x:
        f"{x:.3f}"
    )
)


# ============================================================
# RMSE TABLE
# ============================================================

rmse_table = results_df.pivot(

    index="model",

    columns="horizon",

    values="rmse_min",
)


rmse_table = rmse_table.reindex(
    [
        "Zero-delay",
        "Persistence",
        "AR-GATv2",
        "DMH-GATv2",
    ]
)


print(
    "\n"
    + "=" * 70
)


print(
    "RMSE (MINUTES)"
)


print(
    "=" * 70
)


print(
    rmse_table.to_string(
        float_format=lambda x:
        f"{x:.3f}"
    )
)

# ============================================================
# ACCURACY TABLE
# ============================================================

accuracy_table = results_df.pivot(
    index="model",
    columns="horizon",
    values="accuracy",
)

accuracy_table = accuracy_table.reindex(
    [
        "Zero-delay",
        "Persistence",
        "AR-GATv2",
        "DMH-GATv2",
    ]
)

print(
    "\n"
    + "=" * 70
)

print(
    "ACCURACY"
)

print(
    "=" * 70
)

print(
    accuracy_table.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)
# ============================================================
# DMH VS AR IMPROVEMENT
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "DMH IMPROVEMENT RELATIVE TO AR-GATv2"
)


print(
    "=" * 70
)


for k in range(3):

    horizon = f"t+{k + 1}"


    ar_row = results_df[
        (
            results_df["model"]
            == "AR-GATv2"
        )
        &
        (
            results_df["horizon"]
            == horizon
        )
    ].iloc[0]


    dmh_row = results_df[
        (
            results_df["model"]
            == "DMH-GATv2"
        )
        &
        (
            results_df["horizon"]
            == horizon
        )
    ].iloc[0]


    mae_improvement = (
        (
            ar_row["mae_sec"]
            - dmh_row["mae_sec"]
        )
        / ar_row["mae_sec"]
        * 100.0
    )


    rmse_improvement = (
        (
            ar_row["rmse_sec"]
            - dmh_row["rmse_sec"]
        )
        / ar_row["rmse_sec"]
        * 100.0
    )


    print(
        f"{horizon}: "
        f"MAE improvement = "
        f"{mae_improvement:+.2f}% | "
        f"RMSE improvement = "
        f"{rmse_improvement:+.2f}%"
    )

# ============================================================
# SAVE DMH NODE-LEVEL PREDICTIONS
# ============================================================

DMH_NODE_PATH = (
    ROOT
    / "results"
    / "dmh_validation_node_errors.csv"
)

DMH_HORIZON_MAE_PATH = (
    ROOT
    / "results"
    / "dmh_horizon_mae.csv"
)


val_ids_np = (
    val_ids
    .cpu()
    .numpy()
)


node_prediction_df = pd.DataFrame(
    {
        "source_node_index":
            val_ids_np,

        "prediction_t1_sec":
            dmh_prediction_np[:, 0],

        "prediction_t2_sec":
            dmh_prediction_np[:, 1],

        "prediction_t3_sec":
            dmh_prediction_np[:, 2],
    }
)


node_prediction_df.to_csv(
    DMH_NODE_PATH,
    index=False,
)


# ============================================================
# DMH HORIZON MAE
# ============================================================

horizon_rows = []


for k in range(3):

    valid = (
        target_mask_np[:, k]
        & np.isfinite(
            y_true_all[:, k]
        )
        & np.isfinite(
            dmh_prediction_np[:, k]
        )
    )

    mae = np.mean(
        np.abs(
            dmh_prediction_np[
                valid,
                k,
            ]
            -
            y_true_all[
                valid,
                k,
            ]
        )
    )

    horizon_rows.append(
        {
            "horizon":
                f"t+{k + 1}",

            "mae_seconds":
                float(mae),
        }
    )


horizon_mae_df = pd.DataFrame(
    horizon_rows
)


horizon_mae_df.to_csv(
    DMH_HORIZON_MAE_PATH,
    index=False,
)


print(
    "\nSaved DMH node predictions:"
)
print(DMH_NODE_PATH)

print(
    "\nSaved DMH horizon MAE:"
)
print(DMH_HORIZON_MAE_PATH)

# ============================================================
# HORIZON-WISE ACCURACY TABLE
# ============================================================

selected_models = [
    "Persistence",
    "AR-GATv2",
    "DMH-GATv2",
]

accuracy_table = results_df.pivot(
    index="model",
    columns="horizon",
    values="accuracy",
)

accuracy_table = accuracy_table.reindex(
    selected_models
)

print(
    "\n"
    + "=" * 70
)

print(
    "HORIZON-WISE ACCURACY"
)

print(
    "=" * 70
)

print(
    accuracy_table.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    RESULT_PATH,
    index=False,
)


print(
    "\nSaved:"
)

print(
    RESULT_PATH
)