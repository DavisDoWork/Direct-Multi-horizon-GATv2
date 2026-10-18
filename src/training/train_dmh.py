from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn.functional as F

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


SEED = 42

TRAIN_RATIO = 0.80

EPOCHS = 200

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

HIDDEN_CHANNELS = 32

HEADS = 4

DROPOUT = 0.1

PATIENCE = 25

HORIZONS = 3


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
# MASKED BCE LOSS
# ============================================================

def masked_bce_loss(
    logits,
    target,
    mask,
):
    """
    Binary classification loss.

    logits:
        [N, 3]

    target:
        [N, 3]
        0 = not delayed
        1 = delayed

    mask:
        [N, 3]
    """

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
# MASKED MSE LOSS
# ============================================================

def masked_mse_loss(
    prediction,
    target,
    mask,
):
    """
    Regression loss.

    Only TRUE delayed events contribute.

    prediction:
        [N, 3]

    target:
        [N, 3]

    mask:
        [N, 3]
    """

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

print(
    "target_mask:",
    data.target_mask.shape,
)


# ============================================================
# FEATURES
# ============================================================

feature_names = list(
    data.feature_names
)


x_raw = data.x.float()


# ============================================================
# RAW TARGET
# ============================================================

#
# Expected:
#
# data.y_delay:
#
# [N, 3]
#
# delay in SECONDS
#

y_delay_sec = (
    data.y_delay.float()
)


# ============================================================
# VALID TARGET MASK
# ============================================================

target_mask = (
    data.target_mask.bool()
    & torch.isfinite(
        y_delay_sec
    )
)


# ============================================================
# STAGE 1 TARGET
# ============================================================

#
# Anchor hurdle definition:
#
# delayed = delay > 0
#
# NOT 30 minutes.
#

classifier_target = (
    data.y_delayed.float()
)


# ============================================================
# STAGE 2 TARGET
# ============================================================

#
# Convert seconds → minutes first.
#

y_delay_min = (
    y_delay_sec / 60.0
)


#
# log1p transformation:
#
# log(1 + delay_minutes)
#
# Safe because the regressor is only trained
# on positive-delay observations.
#

regression_target = torch.zeros_like(
    y_delay_min
)


positive_delay_mask = (
    target_mask
    & data.y_delayed.bool()
)


regression_target[
    positive_delay_mask
] = torch.log1p(
    y_delay_min[
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
# INPUT CHECK
# ============================================================

if torch.isnan(
    x_raw
).any():

    raise ValueError(
        "data.x contains NaN values."
    )


if torch.isinf(
    x_raw
).any():

    raise ValueError(
        "data.x contains Inf values."
    )


print("\nValid targets:")

for k in range(HORIZONS):

    print(
        f"t+{k + 1}: "
        f"{target_mask[:, k].sum().item()}"
    )


print("\nPositive-delay regression targets:")

for k in range(HORIZONS):

    print(
        f"t+{k + 1}: "
        f"{positive_delay_mask[:, k].sum().item()}"
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

valid_times = event_time[
    np.isfinite(event_time)
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


print("\nChronological split")
print("=" * 60)

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


train_x = (
    x_raw[
        train_ids
    ].clone()
)


train_classifier_y = (
    classifier_target[
        train_ids
    ].clone()
)


train_regression_y = (
    regression_target[
        train_ids
    ].clone()
)


train_target_mask = (
    target_mask[
        train_ids
    ].clone()
)


train_regression_mask = (
    positive_delay_mask[
        train_ids
    ].clone()
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


val_x = (
    x_raw[
        val_ids
    ].clone()
)


val_classifier_y = (
    classifier_target[
        val_ids
    ].clone()
)


val_regression_y = (
    regression_target[
        val_ids
    ].clone()
)


val_target_mask = (
    target_mask[
        val_ids
    ].clone()
)


val_regression_mask = (
    positive_delay_mask[
        val_ids
    ].clone()
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
# CRITICAL:
#
# Compute preprocessing statistics ONLY
# from training data.
#

feature_mean = train_x.mean(
    dim=0,
    keepdim=True,
)


feature_std = train_x.std(
    dim=0,
    keepdim=True,
)


#
# Avoid division by zero.
#

feature_std = torch.where(
    feature_std < 1e-8,
    torch.ones_like(
        feature_std
    ),
    feature_std,
)


train_x = (
    train_x
    - feature_mean
) / feature_std


val_x = (
    val_x
    - feature_mean
) / feature_std


# ============================================================
# MOVE TO DEVICE
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

val_edge_index = val_edge_index.to(
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

classifier = GATv2DMHClassifier(
    in_channels=train_x.shape[1],
    hidden_channels=HIDDEN_CHANNELS,
    heads=HEADS,
    dropout=DROPOUT,
    horizons=HORIZONS,
).to(device)


regressor = GATv2DMHRegressor(
    in_channels=train_x.shape[1],
    hidden_channels=HIDDEN_CHANNELS,
    heads=HEADS,
    dropout=DROPOUT,
    horizons=HORIZONS,
).to(device)


print("\nClassifier:")
print(classifier)

print("\nRegressor:")
print(regressor)


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

print("\n" + "=" * 80)
print("STAGE 1: DMH GATv2 CLASSIFIER")
print("=" * 80)


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
            f"training loss at epoch {epoch}."
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
            val_edge_index,
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

        best_classifier_state = copy.deepcopy(
            classifier.state_dict()
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
    # EARLY STOP
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


if best_classifier_state is None:

    raise RuntimeError(
        "No classifier checkpoint created."
    )


classifier.load_state_dict(
    best_classifier_state
)


# ============================================================
# TRAIN REGRESSOR
# ============================================================

print("\n" + "=" * 80)
print("STAGE 2: DMH GATv2 REGRESSOR")
print("=" * 80)


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
            f"training loss at epoch {epoch}."
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
            val_edge_index,
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

        best_regressor_state = copy.deepcopy(
            regressor.state_dict()
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
    # EARLY STOP
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


if best_regressor_state is None:

    raise RuntimeError(
        "No regressor checkpoint created."
    )


regressor.load_state_dict(
    best_regressor_state
)


# ============================================================
# QUICK HURDLE VALIDATION PREDICTION
# ============================================================

classifier.eval()

regressor.eval()


with torch.no_grad():

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    val_logits = classifier(
        val_x,
        val_edge_index,
    )

    val_probability = torch.sigmoid(
        val_logits
    )


    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    val_log_delay = regressor(
        val_x,
        val_edge_index,
    )


    #
    # Convert:
    #
    # log(1 + delay_minutes)
    #
    # back into minutes.
    #

    val_delay_magnitude = torch.expm1(
        val_log_delay
    )


    #
    # Delay magnitude cannot be negative.
    #

    val_delay_magnitude = torch.clamp(
        val_delay_magnitude,
        min=0.0,
    )


    # --------------------------------------------------------
    # Hurdle
    # --------------------------------------------------------

    #
    # 0.5 is the default probability threshold.
    #
    # This is NOT the "30 minute" rule.
    #

    predicted_delayed = (
        val_probability >= 0.5
    )


    final_val_prediction_min = torch.where(
        predicted_delayed,
        val_delay_magnitude,
        torch.zeros_like(
            val_delay_magnitude
        ),
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

    "horizons":
        HORIZONS,


    # --------------------------------------------------------
    # Hurdle setup
    # --------------------------------------------------------

    "delay_definition":
        f"delay_sec > {float(data.delay_threshold_sec)}",

    "delay_threshold_sec":
        float(data.delay_threshold_sec),

    "classification_threshold":
        0.5,

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
# COMPLETE
# ============================================================

print("\n" + "=" * 80)

print(
    "DMH TWO-STAGE TRAINING COMPLETE"
)

print("=" * 80)


print(
    "\nClassifier best epoch:",
    best_classifier_epoch,
)

print(
    "Classifier best validation BCE:",
    best_classifier_loss,
)


print(
    "\nRegressor best epoch:",
    best_regressor_epoch,
)

print(
    "Regressor best validation MSE:",
    best_regression_loss,
)


print(
    "\nSaved checkpoint:"
)

print(
    CHECKPOINT_PATH
)