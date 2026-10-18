from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.utils import subgraph

from src.models.gatv2_ar import (
    GATv2ARClassifier,
    GATv2ARRegressor,
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
    / "gatv2_ar_hurdle.pt"
)


PREDICTION_PATH = (
    ROOT
    / "results"
    / "ar_validation_predictions.npy"
)


SEED = 42

TRAIN_RATIO = 0.80

EPOCHS = 200

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

HIDDEN_CHANNELS = 32

# Keep identical to DMH.
HEADS = 4

DROPOUT = 0.1

PATIENCE = 25

CLASSIFICATION_THRESHOLD = 0.5


torch.manual_seed(SEED)

np.random.seed(SEED)


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
# MASKED BCE
# ============================================================

def masked_bce_loss(
    logits,
    target,
    mask,
):

    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )


    masked_loss = torch.where(
        mask,
        loss,
        torch.zeros_like(loss),
    )


    return (
        masked_loss.sum()
        / mask.sum().clamp(min=1)
    )


# ============================================================
# MASKED MSE
# ============================================================

def masked_mse_loss(
    prediction,
    target,
    mask,
):

    safe_target = torch.where(
        mask,
        target,
        torch.zeros_like(target),
    )


    squared_error = (
        prediction
        - safe_target
    ) ** 2


    masked_error = torch.where(
        mask,
        squared_error,
        torch.zeros_like(
            squared_error
        ),
    )


    return (
        masked_error.sum()
        / mask.sum().clamp(min=1)
    )


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
# FEATURES
# ============================================================

feature_names = list(
    data.feature_names
)


x_raw = data.x.float()


# ============================================================
# TARGETS
# ============================================================

y_delay_sec = (
    data.y_delay.float()
)


target_mask = (
    data.target_mask.bool()
    & torch.isfinite(
        y_delay_sec
    )
)


# ============================================================
# IMPORTANT:
# AR MODEL TRAINS ONE-STEP ONLY
# ============================================================

#
# t+1 is column 0.
#

one_step_y_sec = (
    y_delay_sec[
        :,
        0
    ]
)


one_step_mask = (
    target_mask[
        :,
        0
    ]
)


# ============================================================
# CLASSIFICATION TARGET
# ============================================================

#
# Same hurdle definition as DMH:
#
# delayed = delay > 0
#

classifier_target = (
    data.y_delayed[
        :,
        0
    ].float()
)


# ============================================================
# REGRESSION TARGET
# ============================================================

#
# seconds → minutes
#

one_step_y_min = (
    one_step_y_sec / 60.0
)


positive_delay_mask = (
    one_step_mask
    & data.y_delayed[
        :,
        0
    ].bool()
)


regression_target = torch.zeros_like(
    one_step_y_min
)


regression_target[
    positive_delay_mask
] = torch.log1p(
    one_step_y_min[
        positive_delay_mask
    ]
)


# ============================================================
# EVENT TIME
# ============================================================

event_time = (
    data.event_time_ns
    .cpu()
    .numpy()
)


# ============================================================
# FEATURE CHECKS
# ============================================================

if torch.isnan(
    x_raw
).any():

    raise ValueError(
        "data.x contains NaN."
    )


if torch.isinf(
    x_raw
).any():

    raise ValueError(
        "data.x contains Inf."
    )


# ============================================================
# FIND CURRENT DELAY FEATURE
# ============================================================

if (
    "node_delay_sec"
    not in feature_names
):

    raise ValueError(
        "node_delay_sec not found."
    )


current_delay_idx = (
    feature_names.index(
        "node_delay_sec"
    )
)


if (
    "current_delay_valid"
    not in feature_names
):

    raise ValueError(
        "current_delay_valid not found."
    )


current_delay_valid_idx = (
    feature_names.index(
        "current_delay_valid"
    )
)


print(
    "\nCurrent delay feature index:",
    current_delay_idx,
)


print(
    "Current delay validity index:",
    current_delay_valid_idx,
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

valid_times = event_time[
    np.isfinite(
        event_time
    )
]


cutoff_time = np.quantile(
    valid_times,
    TRAIN_RATIO,
)


train_nodes = torch.tensor(
    event_time <= cutoff_time,
    dtype=torch.bool,
)


val_nodes = torch.tensor(
    event_time > cutoff_time,
    dtype=torch.bool,
)


train_ids = torch.where(
    train_nodes
)[0]


val_ids = torch.where(
    val_nodes
)[0]


print(
    "\nChronological split"
)

print(
    "=" * 60
)


print(
    "Train nodes:",
    len(train_ids),
)


print(
    "Validation nodes:",
    len(val_ids),
)


print(
    "Cutoff timestamp:",
    int(cutoff_time),
)


# ============================================================
# TRAIN SUBGRAPH
# ============================================================

train_edge_index, _ = subgraph(

    subset=train_ids,

    edge_index=data.edge_index,

    relabel_nodes=True,

    num_nodes=data.x.shape[0],
)


train_x_raw = (
    x_raw[
        train_ids
    ]
    .clone()
)


train_classifier_y = (
    classifier_target[
        train_ids
    ]
    .clone()
    .unsqueeze(-1)
)


train_regression_y = (
    regression_target[
        train_ids
    ]
    .clone()
    .unsqueeze(-1)
)


train_target_mask = (
    one_step_mask[
        train_ids
    ]
    .clone()
    .unsqueeze(-1)
)


train_regression_mask = (
    positive_delay_mask[
        train_ids
    ]
    .clone()
    .unsqueeze(-1)
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


val_classifier_y = (
    classifier_target[
        val_ids
    ]
    .clone()
    .unsqueeze(-1)
)


val_regression_y = (
    regression_target[
        val_ids
    ]
    .clone()
    .unsqueeze(-1)
)


val_target_mask = (
    one_step_mask[
        val_ids
    ]
    .clone()
    .unsqueeze(-1)
)


val_regression_mask = (
    positive_delay_mask[
        val_ids
    ]
    .clone()
    .unsqueeze(-1)
)


print(
    "\nTrain graph:",
    len(train_ids),
    "nodes |",
    train_edge_index.shape[1],
    "edges",
)


print(
    "Validation graph:",
    len(val_ids),
    "nodes |",
    val_edge_index.shape[1],
    "edges",
)


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

#
# Training statistics ONLY.
#

feature_mean = train_x_raw.mean(
    dim=0,
    keepdim=True,
)


feature_std = train_x_raw.std(
    dim=0,
    keepdim=True,
)


feature_std = torch.where(

    feature_std < 1e-8,

    torch.ones_like(
        feature_std
    ),

    feature_std,
)


train_x = (
    train_x_raw
    - feature_mean
) / feature_std


val_x = (
    val_x_raw
    - feature_mean
) / feature_std


# ============================================================
# DEVICE
# ============================================================

train_x = train_x.to(
    device
)


train_edge_index = train_edge_index.to(
    device
)


train_classifier_y = train_classifier_y.to(
    device
)


train_regression_y = train_regression_y.to(
    device
)


train_target_mask = train_target_mask.to(
    device
)


train_regression_mask = train_regression_mask.to(
    device
)


val_x = val_x.to(
    device
)


val_edge_index_device = val_edge_index.to(
    device
)


val_classifier_y = val_classifier_y.to(
    device
)


val_regression_y = val_regression_y.to(
    device
)


val_target_mask = val_target_mask.to(
    device
)


val_regression_mask = val_regression_mask.to(
    device
)


# ============================================================
# MODELS
# ============================================================

classifier = GATv2ARClassifier(

    in_channels=train_x.shape[1],

    hidden_channels=HIDDEN_CHANNELS,

    heads=HEADS,

    dropout=DROPOUT,

).to(device)


regressor = GATv2ARRegressor(

    in_channels=train_x.shape[1],

    hidden_channels=HIDDEN_CHANNELS,

    heads=HEADS,

    dropout=DROPOUT,

).to(device)


print(
    "\nAR Classifier:"
)

print(
    classifier
)


print(
    "\nAR Regressor:"
)

print(
    regressor
)


# ============================================================
# OPTIMIZERS
# ============================================================

classifier_optimizer = torch.optim.Adam(

    classifier.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,
)


regressor_optimizer = torch.optim.Adam(

    regressor.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 1: AR GATv2 CLASSIFIER"
)

print(
    "=" * 80
)


best_classifier_loss = float(
    "inf"
)

best_classifier_state = None

best_classifier_epoch = None

patience_counter = 0


for epoch in range(
    1,
    EPOCHS + 1,
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    classifier.train()

    classifier_optimizer.zero_grad()


    train_logits = classifier(
        train_x,
        train_edge_index,
    )


    train_loss = masked_bce_loss(
        train_logits,
        train_classifier_y,
        train_target_mask,
    )


    if not torch.isfinite(
        train_loss
    ):

        raise RuntimeError(
            f"Non-finite classifier "
            f"loss at epoch {epoch}."
        )


    train_loss.backward()

    classifier_optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    classifier.eval()


    with torch.no_grad():

        val_logits = classifier(
            val_x,
            val_edge_index_device,
        )


        val_loss = masked_bce_loss(
            val_logits,
            val_classifier_y,
            val_target_mask,
        )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        val_loss.item()
        < best_classifier_loss
    ):

        best_classifier_loss = (
            val_loss.item()
        )

        best_classifier_epoch = epoch

        best_classifier_state = (
            copy.deepcopy(
                classifier.state_dict()
            )
        )

        patience_counter = 0

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch:3d} | "
            f"Train BCE: "
            f"{train_loss.item():8.4f} | "
            f"Val BCE: "
            f"{val_loss.item():8.4f}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        patience_counter
        >= PATIENCE
    ):

        print(
            f"\nClassifier early stopping "
            f"at epoch {epoch}"
        )

        break


if (
    best_classifier_state
    is None
):

    raise RuntimeError(
        "No classifier checkpoint."
    )


classifier.load_state_dict(
    best_classifier_state
)


# ============================================================
# TRAIN REGRESSOR
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 2: AR GATv2 REGRESSOR"
)

print(
    "=" * 80
)


best_regression_loss = float(
    "inf"
)

best_regressor_state = None

best_regressor_epoch = None

patience_counter = 0


for epoch in range(
    1,
    EPOCHS + 1,
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    regressor.train()

    regressor_optimizer.zero_grad()


    train_prediction = regressor(
        train_x,
        train_edge_index,
    )


    train_loss = masked_mse_loss(
        train_prediction,
        train_regression_y,
        train_regression_mask,
    )


    if not torch.isfinite(
        train_loss
    ):

        raise RuntimeError(
            f"Non-finite regressor "
            f"loss at epoch {epoch}."
        )


    train_loss.backward()

    regressor_optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    regressor.eval()


    with torch.no_grad():

        val_prediction = regressor(
            val_x,
            val_edge_index_device,
        )


        val_loss = masked_mse_loss(
            val_prediction,
            val_regression_y,
            val_regression_mask,
        )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        val_loss.item()
        < best_regression_loss
    ):

        best_regression_loss = (
            val_loss.item()
        )

        best_regressor_epoch = epoch

        best_regressor_state = (
            copy.deepcopy(
                regressor.state_dict()
            )
        )

        patience_counter = 0

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch:3d} | "
            f"Train MSE: "
            f"{train_loss.item():8.4f} | "
            f"Val MSE: "
            f"{val_loss.item():8.4f}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        patience_counter
        >= PATIENCE
    ):

        print(
            f"\nRegressor early stopping "
            f"at epoch {epoch}"
        )

        break


if (
    best_regressor_state
    is None
):

    raise RuntimeError(
        "No regressor checkpoint."
    )


regressor.load_state_dict(
    best_regressor_state
)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint = {

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    "classifier_state_dict":
        best_classifier_state,

    "classifier_best_epoch":
        best_classifier_epoch,

    "classifier_best_val_loss":
        best_classifier_loss,


    # --------------------------------------------------------
    # Regressor
    # --------------------------------------------------------

    "regressor_state_dict":
        best_regressor_state,

    "regressor_best_epoch":
        best_regressor_epoch,

    "regressor_best_val_loss":
        best_regression_loss,


    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    "in_channels":
        train_x.shape[1],

    "hidden_channels":
        HIDDEN_CHANNELS,

    "heads":
        HEADS,

    "dropout":
        DROPOUT,


    # --------------------------------------------------------
    # Hurdle
    # --------------------------------------------------------

    "classification_threshold":
        CLASSIFICATION_THRESHOLD,

    "delay_definition":
        f"delay_sec > {float(data.delay_threshold_sec)}",

    "delay_threshold_sec":
        float(data.delay_threshold_sec),

    "regression_transform":
        "log1p(delay_minutes)",


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    "feature_mean":
        feature_mean.cpu(),

    "feature_std":
        feature_std.cpu(),

    "feature_names":
        feature_names,


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    "cutoff_time":
        float(cutoff_time),

    "train_ratio":
        TRAIN_RATIO,
}


torch.save(
    checkpoint,
    CHECKPOINT_PATH,
)


# ============================================================
# AUTOREGRESSIVE VALIDATION ROLLOUT
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "AUTOREGRESSIVE 3-STEP ROLLOUT"
)


print(
    "=" * 80
)


classifier.eval()

regressor.eval()


#
# IMPORTANT:
#
# Start from RAW validation features.
#
# We need raw values because the predicted delay
# must be inserted back into node_delay_sec.
#

rollout_x_raw = (
    val_x_raw.clone()
)


ar_predictions = []


for k in range(3):

    print(
        f"\nPredicting t+{k + 1}..."
    )


    # --------------------------------------------------------
    # Normalize CURRENT graph state
    # --------------------------------------------------------

    rollout_x = (
        rollout_x_raw
        - feature_mean
    ) / feature_std


    rollout_x = rollout_x.to(
        device
    )


    # --------------------------------------------------------
    # ONE-STEP PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = classifier(
            rollout_x,
            val_edge_index_device,
        )


        probability = torch.sigmoid(
            logits
        )


        predicted_delayed = (
            probability
            >= CLASSIFICATION_THRESHOLD
        )


        predicted_log_delay = (
            regressor(
                rollout_x,
                val_edge_index_device,
            )
        )


        predicted_delay_min = (
            torch.expm1(
                predicted_log_delay
            )
        )


        predicted_delay_min = torch.clamp(
            predicted_delay_min,
            min=0.0,
        )


        predicted_delay_min = torch.where(

            predicted_delayed,

            predicted_delay_min,

            torch.zeros_like(
                predicted_delay_min
            ),
        )


        predicted_delay_sec = (
            predicted_delay_min
            * 60.0
        )


    # --------------------------------------------------------
    # [N, 1] -> [N]
    # --------------------------------------------------------

    predicted_delay_sec = (
        predicted_delay_sec
        .squeeze(-1)
        .cpu()
    )


    # --------------------------------------------------------
    # STORE HORIZON
    # --------------------------------------------------------

    ar_predictions.append(
        predicted_delay_sec.numpy()
    )


    # --------------------------------------------------------
    # AUTOREGRESSIVE UPDATE
    # --------------------------------------------------------
    #
    # This is THE key AR step.
    #
    # Prediction at t+k becomes the
    # current-delay feature for t+k+1.
    #

    rollout_x_raw[
        :,
        current_delay_idx
    ] = predicted_delay_sec


    #
    # Prediction now exists for every node.
    #

    rollout_x_raw[
        :,
        current_delay_valid_idx
    ] = 1.0


# ============================================================
# STACK PREDICTIONS
# ============================================================

#
# list:
#
# 3 × [N]
#
# becomes:
#
# [N, 3]
#

ar_predictions = np.stack(
    ar_predictions,
    axis=1,
)


print(
    "\nAR prediction shape:",
    ar_predictions.shape,
)


# ============================================================
# SAVE AR PREDICTIONS
# ============================================================

np.save(
    PREDICTION_PATH,
    ar_predictions,
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "AR TRAINING COMPLETE"
)


print(
    "=" * 80
)


print(
    "\nClassifier best epoch:",
    best_classifier_epoch,
)


print(
    "Classifier best val BCE:",
    best_classifier_loss,
)


print(
    "\nRegressor best epoch:",
    best_regressor_epoch,
)


print(
    "Regressor best val MSE:",
    best_regression_loss,
)


print(
    "\nSaved checkpoint:"
)


print(
    CHECKPOINT_PATH
)


print(
    "\nSaved validation predictions:"
)


print(
    PREDICTION_PATH
)


print(
    "\nPrediction shape:",
    ar_predictions.shape,
)