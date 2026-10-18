from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import subgraph

from src.models.gatv2_dmh import (
    GATv2DMHClassifier,
    GATv2DMHRegressor,
)


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

CHECKPOINT_PATH = (
    ROOT
    / "results"
    / "gatv2_dmh_hurdle.pt"
)

NODE_ERROR_PATH = (
    ROOT
    / "results"
    / "dmh_validation_node_errors.csv"
)

SUMMARY_PATH = (
    ROOT
    / "results"
    / "dmh_horizon_mae.csv"
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
# LOAD GRAPH + CHECKPOINT
# ============================================================

print("\nLoading graph:")
print(GRAPH_PATH)

data = torch.load(
    GRAPH_PATH,
    weights_only=False,
)

print("\nLoading checkpoint:")
print(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)


# ============================================================
# TARGETS
# ============================================================

y_delay_sec = data.y_delay.float()

target_mask = (
    data.target_mask.bool()
    & torch.isfinite(y_delay_sec)
)


# ============================================================
# REBUILD EXACT SAME CHRONOLOGICAL VALIDATION SPLIT
# ============================================================

event_time = (
    data.event_time_ns
    .cpu()
    .numpy()
)

cutoff_time = float(
    checkpoint["cutoff_time"]
)

val_nodes = torch.tensor(
    event_time > cutoff_time,
    dtype=torch.bool,
)

val_ids = torch.where(
    val_nodes
)[0]

print("\nValidation nodes:", len(val_ids))
print("Cutoff timestamp:", int(cutoff_time))


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
    data.x.float()[
        val_ids
    ].clone()
)

val_y_sec = (
    y_delay_sec[
        val_ids
    ].clone()
)

val_mask = (
    target_mask[
        val_ids
    ].clone()
)


# ============================================================
# NORMALIZATION
# ============================================================

feature_mean = (
    checkpoint["feature_mean"]
    .float()
)

feature_std = (
    checkpoint["feature_std"]
    .float()
)

val_x = (
    val_x_raw
    - feature_mean
) / feature_std


# ============================================================
# REBUILD MODELS
# ============================================================

classifier = GATv2DMHClassifier(
    in_channels=checkpoint["in_channels"],
    hidden_channels=checkpoint["hidden_channels"],
    heads=checkpoint["heads"],
    dropout=checkpoint["dropout"],
    horizons=checkpoint["horizons"],
).to(device)

regressor = GATv2DMHRegressor(
    in_channels=checkpoint["in_channels"],
    hidden_channels=checkpoint["hidden_channels"],
    heads=checkpoint["heads"],
    dropout=checkpoint["dropout"],
    horizons=checkpoint["horizons"],
).to(device)

classifier.load_state_dict(
    checkpoint["classifier_state_dict"]
)

regressor.load_state_dict(
    checkpoint["regressor_state_dict"]
)

classifier.eval()
regressor.eval()


# ============================================================
# MOVE VALIDATION GRAPH TO DEVICE
# ============================================================

val_x = val_x.to(device)
val_edge_index = val_edge_index.to(device)


# ============================================================
# DMH HURDLE PREDICTION
# ============================================================

classification_threshold = float(
    checkpoint.get(
        "classification_threshold",
        0.5,
    )
)

with torch.no_grad():

    # Stage 1: delayed / not delayed
    logits = classifier(
        val_x,
        val_edge_index,
    )

    probability = torch.sigmoid(
        logits
    )

    predicted_delayed = (
        probability
        >= classification_threshold
    )

    # Stage 2: positive delay magnitude
    predicted_log_delay = regressor(
        val_x,
        val_edge_index,
    )

# ------------------------------------------------------------
# RAW REGRESSION OUTPUT
# ------------------------------------------------------------

predicted_delay_min_raw = torch.expm1(
    predicted_log_delay
)

predicted_delay_min_raw = torch.clamp(
    predicted_delay_min_raw,
    min=0.0,
)

# Preserve raw regressor prediction BEFORE hurdle
raw_regressor_delay_sec = (
    predicted_delay_min_raw
    * 60.0
)


# ------------------------------------------------------------
# HARD HURDLE OUTPUT
# ------------------------------------------------------------

predicted_delay_min = torch.where(
    predicted_delayed,
    predicted_delay_min_raw,
    torch.zeros_like(
        predicted_delay_min_raw
    ),
)

predicted_delay_sec = (
    predicted_delay_min
    * 60.0
)


pred_sec = predicted_delay_sec.cpu()

raw_pred_sec = (
    raw_regressor_delay_sec.cpu()
)

true_sec = val_y_sec.cpu()

mask = val_mask.cpu()


# ============================================================
# NODE-LEVEL ABSOLUTE ERRORS
# ============================================================

absolute_error = torch.full_like(
    true_sec,
    float("nan"),
)

absolute_error[mask] = torch.abs(
    pred_sec[mask]
    - true_sec[mask]
)
# ============================================================
# RAW REGRESSOR ABSOLUTE ERRORS
# ============================================================

raw_absolute_error = torch.full_like(
    true_sec,
    float("nan"),
)

raw_absolute_error[mask] = torch.abs(
    raw_pred_sec[mask]
    - true_sec[mask]
)


# ============================================================
# HORIZON MAE
# ============================================================

print("\n" + "=" * 80)
print("DMH VALIDATION HORIZON MAE")
print("=" * 80)

summary_rows = []

for k in range(3):

    horizon_mask = mask[:, k]

    # --------------------------------------------------------
    # Final hurdle prediction error
    # --------------------------------------------------------

    final_errors_k = (
        absolute_error[
            horizon_mask,
            k,
        ]
    )

    final_mae_k = (
        final_errors_k
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # Raw regressor prediction error
    # --------------------------------------------------------

    raw_errors_k = (
        raw_absolute_error[
            horizon_mask,
            k,
        ]
    )

    raw_mae_k = (
        raw_errors_k
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # Raw regression MAE only on truly delayed events
    # > 60 sec, matching our final evaluation definition
    # --------------------------------------------------------

    true_delayed_mask = (
        horizon_mask
        &
        (true_sec[:, k] > 60.0)
    )

    raw_delayed_errors_k = (
        raw_absolute_error[
            true_delayed_mask,
            k,
        ]
    )

    if len(raw_delayed_errors_k) > 0:
        raw_delayed_mae_k = (
            raw_delayed_errors_k
            .mean()
            .item()
        )
    else:
        raw_delayed_mae_k = float("nan")


    # --------------------------------------------------------
    # Save
    #
    # Keep "mae_seconds" = final hurdle MAE because
    # aggregate_dmh_by_event.py already expects that column.
    # --------------------------------------------------------

    summary_rows.append(
        {
            "horizon": f"t+{k + 1}",
            "valid_events": int(
                horizon_mask.sum().item()
            ),

            "mae_seconds":
                final_mae_k,

            "mae_minutes":
                final_mae_k / 60.0,

            "raw_regressor_mae_seconds":
                raw_mae_k,

            "raw_regressor_delayed_mae_seconds":
                raw_delayed_mae_k,
        }
    )


    print(
        f"\nt+{k + 1}"
    )

    print(
        f"N = "
        f"{horizon_mask.sum().item()}"
    )

    print(
        f"RAW regressor MAE, all events = "
        f"{raw_mae_k:.3f} sec"
    )

    print(
        f"RAW regressor MAE, true delay >60s = "
        f"{raw_delayed_mae_k:.3f} sec"
    )

    print(
        f"FINAL hurdle MAE = "
        f"{final_mae_k:.3f} sec"
    )

    print(
        f"Hurdle change = "
        f"{final_mae_k - raw_mae_k:+.3f} sec"
    )

# ============================================================
# SAVE PER-NODE RESULTS
# ============================================================

def get_metadata(name):
    """
    Return validation metadata if the graph contains it.
    Otherwise return NaN.
    """
    if not hasattr(data, name):
        return np.full(
            len(val_ids),
            np.nan,
        )

    value = getattr(data, name)

    if torch.is_tensor(value):
        return (
            value[val_ids]
            .cpu()
            .numpy()
        )

    value = np.asarray(value)

    return value[
        val_ids.cpu().numpy()
    ]


results = pd.DataFrame(
    {
        "source_node_index": (
            val_ids.cpu().numpy()
        ),
        "node_id": get_metadata(
            "node_id"
        ),
        "original_event_id": get_metadata(
            "original_event_id"
        ),
        "trip_uid": get_metadata(
            "trip_uid"
        ),
        "stop_id": get_metadata(
            "stop_id"
        ),
        "stop_sequence": get_metadata(
            "stop_sequence"
        ),
        "event_time_ns": get_metadata(
            "event_time_ns"
        ),
    }
)

for k in range(3):

    results[
        f"prediction_t{k + 1}_sec"
    ] = pred_sec[:, k].numpy()

    results[
        f"true_t{k + 1}_sec"
    ] = true_sec[:, k].numpy()

    results[
        f"absolute_error_t{k + 1}_sec"
    ] = absolute_error[:, k].numpy()

    results[
        f"valid_t{k + 1}"
    ] = mask[:, k].numpy()


# ============================================================
# SAVE OUTPUTS
# ============================================================

NODE_ERROR_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    NODE_ERROR_PATH,
    index=False,
)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


print("\nSaved per-node errors:")
print(NODE_ERROR_PATH)

print("\nSaved horizon MAE summary:")
print(SUMMARY_PATH)
