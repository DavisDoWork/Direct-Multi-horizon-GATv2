import torch
import torch.nn as nn

from src.models.gatv2_dmh import GATv2Backbone


# ============================================================
# STAGE 1: AUTOREGRESSIVE CLASSIFIER
# ============================================================

class GATv2ARClassifier(nn.Module):
    """
    One-step autoregressive delay classifier.

    Input
    -----
    x:
        [N, F]

    edge_index:
        [2, E]

    Output
    ------
    logits:
        [N, 1]

    The same model is recursively reused for:

        t+1
        t+2
        t+3
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=4,
        dropout=0.1,
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
            1,
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

        logits = self.classifier(
            h
        )

        return logits


# ============================================================
# STAGE 2: AUTOREGRESSIVE REGRESSOR
# ============================================================

class GATv2ARRegressor(nn.Module):
    """
    One-step autoregressive delay magnitude regressor.

    Output:
        [N, 1]

    Delay is predicted in:

        log1p(delay_minutes)

    space.

    The network output is constrained using:

        5 * tanh(raw_prediction)
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=4,
        dropout=0.1,
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
            1,
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

        raw_prediction = self.regressor(
            h
        )

        log_delay_prediction = (
            5.0
            * torch.tanh(
                raw_prediction
            )
        )

        return log_delay_prediction