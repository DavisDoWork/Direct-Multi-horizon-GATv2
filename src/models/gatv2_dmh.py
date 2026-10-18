import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


# ============================================================
# GATV2 BACKBONE
# ============================================================

class GATv2Backbone(nn.Module):
    """
    3-layer GATv2 backbone.

    Structure
    ---------
    x [N, F]
        ↓
    Linear projection
        ↓
    [N, hidden_channels]
        ↓
    GATv2 + ReLU + Dropout + Residual + LayerNorm
        ↓
    GATv2 + ReLU + Dropout + Residual + LayerNorm
        ↓
    GATv2 + ReLU + Dropout + Residual + LayerNorm
        ↓
    [N, hidden_channels]

    Notes
    -----
    heads=32 follows the anchor-paper architecture.

    concat=False keeps the output dimension equal to
    hidden_channels instead of hidden_channels * heads.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=32,
        dropout=0.1,
    ):
        super().__init__()

        # Raw features F -> hidden dimension
        self.input_projection = nn.Linear(
            in_channels,
            hidden_channels,
        )

        # ----------------------------------------------------
        # GATv2 layers
        # ----------------------------------------------------

        self.gat1 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

        self.gat3 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Layer normalization
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            hidden_channels
        )

        self.norm3 = nn.LayerNorm(
            hidden_channels
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        # ====================================================
        # INPUT PROJECTION
        # ====================================================

        x = self.input_projection(x)

        # ====================================================
        # GAT LAYER 1
        # ====================================================

        residual = x

        x = self.gat1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.norm1(
            x + residual
        )

        # ====================================================
        # GAT LAYER 2
        # ====================================================

        residual = x

        x = self.gat2(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.norm2(
            x + residual
        )

        # ====================================================
        # GAT LAYER 3
        # ====================================================

        residual = x

        x = self.gat3(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.norm3(
            x + residual
        )

        return x


# ============================================================
# STAGE 1: CLASSIFIER
# ============================================================

class GATv2DMHClassifier(nn.Module):
    """
    Stage 1 of the hurdle model.

    Predicts whether each future event is delayed.

    Output
    ------
    logits:
        [N, 3]

        column 0 -> P(delay at t+1)
        column 1 -> P(delay at t+2)
        column 2 -> P(delay at t+3)

    Important
    ---------
    forward() returns logits, NOT probabilities.

    During inference:

        probability = sigmoid(logits)
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=32,
        dropout=0.1,
        horizons=3,
    ):
        super().__init__()

        self.backbone = GATv2Backbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            hidden_channels,
            horizons,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        h = self.backbone(
            x,
            edge_index,
        )

        logits = self.classifier(h)

        return logits


# ============================================================
# STAGE 2: REGRESSOR
# ============================================================

class GATv2DMHRegressor(nn.Module):
    """
    Stage 2 of the hurdle model.

    Predicts future delay magnitude in transformed log-space.

    Output
    ------
    [N, 3]

        column 0 -> log delay t+1
        column 1 -> log delay t+2
        column 2 -> log delay t+3

    The raw network output is constrained using:

        5 * tanh(raw_output)

    This keeps the transformed prediction bounded.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=32,
        dropout=0.1,
        horizons=3,
    ):
        super().__init__()

        self.backbone = GATv2Backbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
        )

        self.regressor = nn.Linear(
            hidden_channels,
            horizons,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        h = self.backbone(
            x,
            edge_index,
        )

        raw_prediction = self.regressor(h)

        log_delay_prediction = (
            5.0
            * torch.tanh(
                raw_prediction
            )
        )

        return log_delay_prediction