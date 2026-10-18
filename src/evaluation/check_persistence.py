from pathlib import Path

import torch


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


# ============================================================
# LOAD GRAPH
# ============================================================

print("Loading graph:")
print(GRAPH_PATH)

data = torch.load(
    GRAPH_PATH,
    weights_only=False,
)

print("Graph nodes:", data.x.shape[0])


# ============================================================
# GET CURRENT DELAY
# ============================================================

delay_idx = data.feature_names.index(
    "node_delay_sec"
)

current_delay = (
    data.x[:, delay_idx]
    .float()
)

y = data.y_delay.float()

mask = data.target_mask.bool()


# ============================================================
# PERSISTENCE DIAGNOSTIC
# ============================================================

print("\n" + "=" * 80)
print("RAW PERSISTENCE DIAGNOSTIC")
print("=" * 80)

for h in range(3):

    valid = (
        mask[:, h]
        & torch.isfinite(current_delay)
        & torch.isfinite(y[:, h])
    )

    error = torch.abs(
        current_delay[valid]
        - y[valid, h]
    )

    print(
        f"\nt+{h+1}"
    )

    print(
        "N      =",
        len(error)
    )

    print(
        "MAE    =",
        error.mean().item()
    )

    print(
        "Median =",
        error.median().item()
    )

    print(
        "P90    =",
        torch.quantile(
            error,
            0.90
        ).item()
    )

    print(
        "P95    =",
        torch.quantile(
            error,
            0.95
        ).item()
    )

    print(
        "P99    =",
        torch.quantile(
            error,
            0.99
        ).item()
    )

    print(
        "Max    =",
        error.max().item()
    )

print("\n" + "=" * 80)
print("PERSISTENCE OUTLIER ANALYSIS")
print("=" * 80)

for h in range(3):

    valid = (
        mask[:, h]
        & torch.isfinite(current_delay)
        & torch.isfinite(y[:, h])
    )

    error = torch.abs(
        current_delay[valid]
        - y[valid, h]
    )

    print(f"\nt+{h+1}")

    for threshold in [
        300,      # 5 min
        600,      # 10 min
        1800,     # 30 min
        3600,     # 1 hour
    ]:

        count = (
            error > threshold
        ).sum().item()

        percent = (
            count
            / len(error)
            * 100
        )

        print(
            f"> {threshold:4d}s: "
            f"{count:6d} "
            f"({percent:.4f}%)"
        )

    # Trimmed MAEs for diagnosis only
    for threshold in [
        300,
        600,
        1800,
        3600,
    ]:

        kept = (
            error <= threshold
        )

        trimmed_mae = (
            error[kept]
            .mean()
            .item()
        )

        print(
            f"MAE if errors <= {threshold:4d}s: "
            f"{trimmed_mae:.3f}s"
        )